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
from .core import View
from .logging import RunLog, utc_now
from .prompts import render_followup, render_prompt

# No shell: Python is provided by the single, isolated MCP tool below.
DISABLED_FEATURES = ("shell_tool", "apps", "plugins", "browser_use", "browser_use_external",
                     "browser_use_full_cdp_access", "computer_use", "image_generation", "view_image",
                     "multi_agent", "skill_search", "hooks", "memories", "goals", "tool_suggest")


def tool_policy(allow_python: bool) -> dict:
    return {"allowed_tools": ["local_python.python"] if allow_python else [],
            "web_search": "disabled", "command_network_access": False,
            "shell": False, "apps": False, "plugins": False,
            "python_isolation": "bubblewrap --unshare-all; fresh filesystem; no host /home",
            "disabled_features": list(DISABLED_FEATURES)}


def command_for(executable: str, player: PlayerConfig, *, workspace: Path, reply_path: Path,
                python_log: Path, session_id: str | None, persistent: bool) -> list[str]:
    command = [executable, "--ask-for-approval", "never", "exec", "--sandbox", "read-only"]
    if session_id:
        command.append("resume")
    command += ["--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "--json",
                "--output-last-message", str(reply_path)]
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
    def __init__(self, config: RunConfig, player: PlayerConfig, role: str, log: RunLog, *, persistent: bool):
        self.config, self.player, self.role, self.log = config, player, role, log
        self.persistent = persistent
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
        self.log.events.emit("player.created", role=role, config=player.__dict__,
                             policy=tool_policy(player.allow_python), auth_file_available=self.auth_copied)

    def __call__(self, view: View) -> str:
        if view.role != self.role:
            raise ValueError("Игрок получил чужую роль.")
        prompt = (render_followup(self.config, view) if self.persistent and self.session_id
                  else render_prompt(self.config, view, persistent=self.persistent))
        return self.turn(prompt, position=view.round_number)

    def turn(self, prompt: str, *, position: int = 0) -> str:
        call_id = f"{self.role.lower()}_{uuid4().hex}"
        call_dir = self.log.directory / "calls" / call_id
        call_dir.mkdir()
        prompt_path, reply_path = call_dir / "prompt.txt", call_dir / "reply.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        command = command_for(self.executable, self.player, workspace=self.workspace,
                              reply_path=reply_path, python_log=call_dir / "python.jsonl",
                              session_id=self.session_id, persistent=self.persistent)
        metadata = {"call_id": call_id, "role": self.role, "position": position,
                    "session_id_before": self.session_id, "command": command, "cwd": str(self.workspace),
                    "requested_model": self.player.model, "policy": tool_policy(self.player.allow_python),
                    "started_utc": utc_now()}
        (call_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self.log.events.emit("model.started", **metadata)
        started = time.monotonic()
        violations, reader_errors = [], []
        process = None

        def consume(pipe, name):
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
                            self.log.events.emit("cli.unparsed_stdout", call_id=call_id, line=line)
                            continue
                        if event.get("type") == "thread.started" and self.persistent:
                            self.session_id = event.get("thread_id")
                        audit = audit_event(event, self.player.allow_python)
                        if audit:
                            self.log.events.emit("tool.observed", call_id=call_id, role=self.role, **audit)
                            if not audit["allowed"]:
                                violations.append(audit)
                                if process:
                                    _kill_process_group(process)
                        self.log.events.emit("cli.event", call_id=call_id, role=self.role, event=event)
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
            if self.persistent and not self.session_id:
                raise RuntimeError("CLI не сообщил идентификатор беседы.")
            outcome = "completed"
            return reply_path.read_text(encoding="utf-8")
        except BaseException as exc:
            self.log.events.emit("model.error", call_id=call_id, role=self.role,
                                 error=type(exc).__name__, message=str(exc))
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
                            elapsed_seconds=time.monotonic() - started, session_id_after=self.session_id)
            (call_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            self.log.events.emit("model.finished", **metadata)

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
                self.log.events.emit("player.resolved_models", role=self.role, models=sorted(resolved),
                                     requested_model=self.player.model)
                destination = self.log.directory / "sessions" / f"{self.role.lower()}_{self.root.name}"
                destination.parent.mkdir(exist_ok=True)
                shutil.copytree(sessions, destination)
        finally:
            self.temp.cleanup()  # In particular, the temporary auth copy is removed.
