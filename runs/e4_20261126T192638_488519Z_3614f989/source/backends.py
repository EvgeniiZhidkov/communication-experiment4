"""Codex CLI adapter with explicit tool policy and streamed diagnostics."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

from .config import PlayerConfig, RunConfig
from .core import ACTIONS, ALICE, ANSWERS, BOB, View
from .logging import RunLog, utc_now
from .prompts import alice_tracks_second_pass, calculation_instructions, render_followup, render_prompt

# No shell: Python is provided by the single, isolated MCP tool below.
DISABLED_FEATURES = ("shell_tool", "apps", "plugins", "browser_use", "browser_use_external",
                     "browser_use_full_cdp_access", "computer_use", "image_generation", "view_image",
                     "multi_agent", "skill_search", "hooks", "memories", "goals", "tool_suggest")
CALCULATION_REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "calculation": {"type": "string"},
        "action": {"type": "string", "enum": sorted(ACTIONS | ANSWERS)},
    },
    "required": ["calculation", "action"],
    "additionalProperties": False,
}
ACCUMULATOR_FIELDS = ("accumulator_before", "accumulator_after")
ACCUMULATOR_REPLY_SCHEMA = {
    **CALCULATION_REPLY_SCHEMA,
    "properties": {
        **{field: {"type": "integer", "enum": [0, 1]} for field in ACCUMULATOR_FIELDS},
        **CALCULATION_REPLY_SCHEMA["properties"],
    },
    "required": [*ACCUMULATOR_FIELDS, *CALCULATION_REPLY_SCHEMA["required"]],
}
POSITIONS_REPLY_SCHEMA = {
    **CALCULATION_REPLY_SCHEMA,
    "properties": {
        "remaining_positions": {"type": "array", "items": {"type": "integer"}},
        **CALCULATION_REPLY_SCHEMA["properties"],
    },
    "required": ["remaining_positions", *CALCULATION_REPLY_SCHEMA["required"]],
}


def tool_policy(allow_python: bool) -> dict:
    return {"allowed_tools": ["local_python.python"] if allow_python else [],
            "web_search": "disabled", "command_network_access": False,
            "shell": False, "apps": False, "plugins": False,
            "python_isolation": "bubblewrap --unshare-all; fresh filesystem; no host /home",
            "disabled_features": list(DISABLED_FEATURES)}


def command_for(executable: str, player: PlayerConfig, *, workspace: Path, reply_path: Path,
                python_log: Path, session_id: str | None, persistent: bool,
                fork: bool = False, output_schema: Path | None = None) -> list[str]:
    if fork and (not session_id or not persistent):
        raise RuntimeError("Для отдельной ветки нужна сохранённая сессия.")
    command = [executable, "--ask-for-approval", "never", "exec", "--sandbox", "read-only"]
    if session_id:
        command.append("fork" if fork else "resume")
    command += ["--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "--json",
                "--output-last-message", str(reply_path)]
    if output_schema is not None:
        command += ["--output-schema", str(output_schema)]
    for name in DISABLED_FEATURES:
        command += ["--disable", name]
    settings = {
        "web_search": "disabled", "sandbox_workspace_write.network_access": False,
        "project_doc_max_bytes": 0, "skills.bundled.enabled": False,
        "skills.include_instructions": False, "features.skip_host_skill_discovery": True,
        "suppress_unstable_features_warning": True,
    }
    if player.reasoning_effort:
        settings["model_reasoning_effort"] = player.reasoning_effort
    if player.allow_python:
        settings.update({
            "mcp_servers.local_python.command": sys.executable,
            "mcp_servers.local_python.args": [str(Path(__file__).with_name("python_tool.py")),
                                               "--log", str(python_log)],
            "mcp_servers.local_python.required": True,
            "mcp_servers.local_python.enabled_tools": ["python"],
            "mcp_servers.local_python.tools.python.approval_mode": "approve",
        })
    for key, value in settings.items():
        command += ["-c", f"{key}={json.dumps(value, ensure_ascii=False)}"]
    if player.model:
        command += ["--model", player.model]
    if not persistent:
        command.append("--ephemeral")
    if session_id:
        command.append(session_id)
    command.append("-")
    return command


def audit_event(event: dict, allow_python: bool) -> dict | None:
    item = event.get("item") or {}
    kind = item.get("type", "")
    if kind == "mcp_tool_call":
        server, tool = item.get("server"), item.get("tool")
        allowed = allow_python and server == "local_python" and tool == "python"
        return {"tool": f"{server}.{tool}", "allowed": allowed, "status": item.get("status")}
    if kind in {"command_execution", "web_search", "image_generation", "file_change", "collab_tool_call"}:
        return {"tool": kind, "allowed": False, "status": item.get("status")}
    if kind and kind not in {"agent_message", "reasoning", "todo_list", "error"}:
        return {"tool": f"unknown_item:{kind}", "allowed": False, "status": item.get("status")}
    return None


def _kill_process_group(process: subprocess.Popen) -> None:
    # The CLI may have exited while a descendant still holds its output pipes.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class CodexPlayer:
    def __init__(self, config: RunConfig, player: PlayerConfig, role: str, log: RunLog, *, persistent: bool,
                 episode_context: dict[str, int] | None = None):
        self.config, self.player, self.role, self.log = config, player, role, log
        self.persistent = persistent
        self.episode_context = dict(episode_context or {})
        self.session_id: str | None = None
        self.executable = shutil.which(config.codex_executable)
        if self.executable is None:
            raise RuntimeError("Codex CLI не найден. Установите CLI и выполните codex login.")
        self.temp = tempfile.TemporaryDirectory(prefix=f"halfduplex-{role.lower()}-")
        self.root = Path(self.temp.name)
        self.workspace = self.root / "work"
        self.workspace.mkdir(mode=0o700)
        self.home = self.root / "codex"
        self.home.mkdir(mode=0o700)
        self.calculation_schema = self.root / "calculation.schema.json"
        self.accumulator_fields = ACCUMULATOR_FIELDS if config.track_bob_accumulator and role == BOB else ()
        self.calculation_reply_schema = ACCUMULATOR_REPLY_SCHEMA if self.accumulator_fields else CALCULATION_REPLY_SCHEMA
        self.positions_schema = self.root / "positions.schema.json"
        if config.record_calculations:
            self.calculation_schema.write_text(json.dumps(self.calculation_reply_schema), encoding="utf-8")
        if config.track_alice_second_pass and role == ALICE:
            self.positions_schema.write_text(json.dumps(POSITIONS_REPLY_SCHEMA), encoding="utf-8")
        # Carry authentication only; never inherit plugins, skills, sessions or user config.
        auth_source = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
        self.auth_copied = auth_source.is_file()
        if self.auth_copied:
            auth_target = self.home / "auth.json"
            shutil.copyfile(auth_source, auth_target)
            auth_target.chmod(0o600)
        self.env = {key: value for key, value in os.environ.items()
                    if key in {"PATH", "HOME", "LANG", "LC_ALL", "TERM", "SYSTEMROOT", "TMPDIR"}}
        self.env["CODEX_HOME"] = str(self.home)
        self.env["NO_COLOR"] = "1"
        self.env["RUST_LOG"] = "warn"
        self._emit("player.created", role=role, config=player.__dict__,
                   policy=tool_policy(player.allow_python), auth_file_available=self.auth_copied)

    def _emit(self, kind: str, **fields) -> None:
        self.log.events.emit(kind, **(self.episode_context | fields))

    def __call__(self, view: View) -> str:
        if view.role != self.role:
            raise ValueError("Игрок получил чужую роль.")
        prompt = (render_followup(self.config, view) if self.persistent and self.session_id
                  else render_prompt(self.config, view, persistent=self.persistent))
        output_schema = None
        if alice_tracks_second_pass(self.config, view):
            output_schema = self.positions_schema
        elif calculation_instructions(self.config, view) is not None:
            output_schema = self.calculation_schema
        return self.turn(prompt, position=view.round_number, output_schema=output_schema)

    def _calculation_action(self, reply: str, *, call_id: str, position: int, output_schema: Path) -> str:
        try:
            body = json.loads(reply)
        except json.JSONDecodeError:
            body = None
        fields = {**self.episode_context, "call_id": call_id, "role": self.role, "round": position}
        positions_reply = output_schema == self.positions_schema
        schema = POSITIONS_REPLY_SCHEMA if positions_reply else self.calculation_reply_schema
        if isinstance(body, dict) and list(body) == schema["required"]:
            calculation, action = body["calculation"], body["action"]
            if positions_reply:
                remaining = body["remaining_positions"]
                state = {"remaining_positions": remaining}
                valid_state = isinstance(remaining, list) and all(type(index) is int for index in remaining)
            else:
                state = {name: body[name] for name in self.accumulator_fields}
                valid_state = all(type(bit) is int and bit in (0, 1) for bit in state.values())
            if (isinstance(calculation, str) and calculation.strip()
                    and isinstance(action, str) and action in ACTIONS | ANSWERS
                    and valid_state):
                if self.log.calculations is not None:
                    self.log.calculations.emit("decision.calculation", **fields,
                                               **state, calculation=calculation, action=action)
                return action
        if self.log.calculations is not None:
            self.log.calculations.emit("decision.calculation_invalid", **fields, raw_reply=reply)
        return "[Некорректный формат расчёта]"

    def turn(self, prompt: str, *, position: int = 0, fork: bool = False,
             output_schema: Path | None = None) -> str:
        session_id = self.session_id
        call_id = f"{self.role.lower()}_{uuid4().hex}"
        call_dir = self.log.directory / "calls" / call_id
        call_dir.mkdir()
        prompt_path, reply_path = call_dir / "prompt.txt", call_dir / "reply.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        command = command_for(self.executable, self.player, workspace=self.workspace,
                              reply_path=reply_path, python_log=call_dir / "python.jsonl",
                              session_id=session_id, persistent=self.persistent, fork=fork,
                              output_schema=output_schema)
        metadata = {**self.episode_context, "call_id": call_id, "role": self.role, "position": position,
                    "session_id_before": session_id, "command": command, "cwd": str(self.workspace),
                    "requested_model": self.player.model, "policy": tool_policy(self.player.allow_python),
                    "started_utc": utc_now()}
        if fork:
            metadata["purpose"] = "deviation_explanation"
        if output_schema is not None:
            metadata["response_format"] = "calculation"
        (call_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self._emit("model.started", **metadata)
        started = time.monotonic()
        violations, reader_errors = [], []
        process = None

        def consume(pipe, name):
            nonlocal session_id
            try:
                with (call_dir / name).open("w", encoding="utf-8") as stream:
                    for line in pipe:
                        stream.write(line)
                        stream.flush()
                        if name != "stdout.jsonl":
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            self._emit("cli.unparsed_stdout", call_id=call_id, line=line)
                            continue
                        if event.get("type") == "thread.started" and self.persistent:
                            session_id = event.get("thread_id")
                            if not fork:
                                self.session_id = session_id
                        audit = audit_event(event, self.player.allow_python)
                        if audit:
                            self._emit("tool.observed", call_id=call_id, role=self.role, **audit)
                            if not audit["allowed"]:
                                violations.append(audit)
                                if process:
                                    _kill_process_group(process)
                        self._emit("cli.event", call_id=call_id, role=self.role, event=event)
            except Exception as exc:
                reader_errors.append(str(exc))

        threads = []
        outcome = "error"
        def feed_prompt():
            try:
                process.stdin.write(prompt)
                process.stdin.close()
            except (BrokenPipeError, OSError):
                reader_errors.append("CLI закрыл stdin до передачи полного промпта.")

        try:
            process = subprocess.Popen(command, cwd=self.workspace, env=self.env,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, encoding="utf-8", errors="replace", start_new_session=True)
            for pipe, name in ((process.stdout, "stdout.jsonl"), (process.stderr, "stderr.log")):
                thread = threading.Thread(target=consume, args=(pipe, name), daemon=True)
                thread.start()
                threads.append(thread)
            feeder = threading.Thread(target=feed_prompt, daemon=True)
            feeder.start()
            threads.append(feeder)
            deadline = started + self.player.timeout_seconds
            process.wait(timeout=max(0, deadline - time.monotonic()))
            for thread in threads:
                thread.join(timeout=max(0, deadline - time.monotonic()))
            if violations:
                raise RuntimeError(f"Модель вызвала запрещённый инструмент: {violations}")
            if any(t.is_alive() for t in threads):
                raise subprocess.TimeoutExpired(command, self.player.timeout_seconds)
            if reader_errors:
                raise RuntimeError(f"Не удалось полностью сохранить вывод CLI: {reader_errors}")
            if process.returncode:
                details = (call_dir / "stderr.log").read_text(encoding="utf-8")[-1800:]
                raise RuntimeError(f"Codex CLI: код {process.returncode}. Диагностика: {call_dir}\n{details}")
            if not reply_path.is_file():
                raise RuntimeError(f"CLI не сохранил итоговый ответ. Диагностика: {call_dir}")
            if self.persistent and not session_id:
                raise RuntimeError("CLI не сообщил идентификатор беседы.")
            if fork and session_id == self.session_id:
                raise RuntimeError("CLI не сообщил идентификатор отдельной ветки.")
            reply = reply_path.read_text(encoding="utf-8")
            if output_schema is not None:
                reply = self._calculation_action(reply, call_id=call_id, position=position, output_schema=output_schema)
            outcome = "completed"
            return reply
        except BaseException as exc:
            self._emit("model.error", call_id=call_id, role=self.role, error=type(exc).__name__, message=str(exc))
            raise
        finally:
            if process:
                _kill_process_group(process)
                process.wait()
            for thread in threads:
                thread.join(timeout=5)
            if process:
                for pipe in (process.stdout, process.stderr, process.stdin):
                    if pipe and not pipe.closed:
                        pipe.close()
            metadata.update(outcome=outcome, returncode=process.returncode if process else None,
                            elapsed_seconds=time.monotonic() - started, session_id_after=session_id)
            (call_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            self._emit("model.finished", **metadata)

    def close(self):
        try:
            sessions = self.home / "sessions"
            if sessions.exists():
                resolved = set()
                for path in sessions.rglob("*.jsonl"):
                    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        payload = row.get("payload")
                        model = payload.get("model") if isinstance(payload, dict) else None
                        if row.get("type") == "turn_context" and isinstance(model, str):
                            resolved.add(model)
                self._emit("player.resolved_models", role=self.role, models=sorted(resolved),
                           requested_model=self.player.model)
                destination = self.log.directory / "sessions" / f"{self.role.lower()}_{self.root.name}"
                destination.parent.mkdir(exist_ok=True)
                shutil.copytree(sessions, destination)
        finally:
            self.temp.cleanup()  # In particular, the temporary auth copy is removed.
