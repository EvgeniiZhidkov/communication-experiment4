"""One configurable runner for E4 episodes and decisions on reference histories."""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import traceback
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

from .analysis import analyze_file, reference_player
from .backends import CodexPlayer, tool_policy
from .config import PlayerConfig, RunConfig
from .core import ALICE, BOB, View, play
from .logging import RunLog, source_manifest
from .prompts import render_prompt
from .python_tool import PythonSandbox


def input_pairs(config: RunConfig) -> list[tuple[str, str]]:
    if config.case_limit is None:
        return [tuple(config.case)]
    rng = random.Random(config.seed)
    total = 4 ** config.n
    if config.sampling == "eq_balanced":
        equal_count = (config.case_limit + 1) // 2
        width = 2 ** config.n
        if width <= 2 ** 63 - 1:
            equal = rng.sample(range(width), equal_count)
        else:
            equal = []
            seen = set()
            while len(equal) < equal_count:
                a = rng.getrandbits(config.n)
                if a not in seen:
                    seen.add(a)
                    equal.append(a)
        indices = [a * (width + 1) for a in equal]
        seen = set(indices)
        while len(indices) < config.case_limit:
            index = rng.randrange(total)
            a, b = divmod(index, width)
            if a != b and index not in seen:
                seen.add(index)
                indices.append(index)
        rng.shuffle(indices)
    elif total <= 2 ** 63 - 1:
        indices = rng.sample(range(total), config.case_limit)
    else:
        # range's length overflows Py_ssize_t for large n; rejection sampling
        # still gives uniformly selected distinct pairs, in a reproducible order.
        seen, indices = set(), []
        while len(indices) < config.case_limit:
            index = rng.getrandbits(2 * config.n)
            if index not in seen:
                seen.add(index)
                indices.append(index)
    width = 2 ** config.n
    return [(f"{a:0{config.n}b}", f"{b:0{config.n}b}")
            for a, b in (divmod(i, width) for i in indices)]


@dataclass(frozen=True)
class RunResult:
    directory: Path
    log_path: Path
    report_path: Path
    summary: dict


def _players(config: RunConfig, log: RunLog, stack: ExitStack, *, persistent: bool):
    players = {}
    reference = reference_player(config)
    for role, settings in ((ALICE, config.alice), (BOB, config.bob)):
        if settings.backend == "python":
            players[role] = reference
        elif settings.backend == "manual":
            def manual(view, cfg=config):
                print(render_prompt(cfg, view), flush=True)
                return input("Ответ: ")
            players[role] = manual
        else:
            player = CodexPlayer(config, settings, role, log, persistent=persistent)
            stack.callback(player.close)
            players[role] = player
    return players


def run_experiment(config: RunConfig) -> RunResult:
    log = RunLog(config.output_dir, fsync=config.fsync)
    print(f"Каталог прогона: {log.directory}", flush=True)
    header = {"schema_version": 2, "mode": config.experiment, "protocol": config.protocol,
              "implementation": config.implementation.name,
              "n": config.n, "function": config.task, "channel": config.channel,
              "explanation": config.explanation, "config": config.record()}
    log.journal.emit("config", **header)
    failure = None
    try:
        pairs = input_pairs(config)
        manifest = {**source_manifest(), **header, "pairs": pairs,
                    "protocol_source": config.implementation.source,
                    "protocol_bound": config.implementation.formula,
                    "max_rounds": config.max_rounds,
                    "tool_policy": {ALICE: tool_policy(config.alice.allow_python),
                                    BOB: tool_policy(config.bob.allow_python)}}
        snapshot = log.directory / "source"
        snapshot.mkdir()
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copyfile(path, snapshot / path.name)
        codex_used = any(p.backend == "codex" for p in (config.alice, config.bob))
        if codex_used:
            executable = shutil.which(config.codex_executable)
            if not executable:
                raise RuntimeError("Codex CLI не найден.")
            version = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=10)
            manifest["codex_version"] = version.stdout.strip()
            if any(p.backend == "codex" and p.allow_python for p in (config.alice, config.bob)):
                manifest["python_preflight"] = PythonSandbox().preflight()
                log.events.emit("python.preflight", **manifest["python_preflight"])
        (log.directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        reference = reference_player(config)
        seen = set()
        for case_index, (x, y) in enumerate(pairs):
            if config.experiment == "episode":
                for repeat in range(1, config.repeats + 1):
                    context = {"case_index": case_index, "repeat": repeat, "x": x, "y": y}
                    log.events.emit("episode.started", **context)
                    with ExitStack() as stack:
                        players = _players(config, log, stack, persistent=True)
                        def emit(kind, **fields):
                            log.events.emit(kind, **context, **fields)
                        episode = play(config.n, x, y, players[ALICE], players[BOB],
                                       channel=config.channel, max_rounds=config.max_rounds, emit=emit)
                        log.journal.emit("episode", **context, **episode.record())
                        log.events.emit("episode.finished", **context, status=episode.status)
            else:
                expected = play(config.n, x, y, reference, reference, channel=config.channel,
                                max_rounds=config.implementation.rounds(config.n))
                with ExitStack() as stack:
                    players = _players(config, log, stack, persistent=False)
                    for decision in expected.decisions():
                        view = decision.view
                        if view in seen:
                            continue
                        seen.add(view)
                        for repeat in range(1, config.repeats + 1):
                            log.events.emit("step.started", case_index=case_index, repeat=repeat, state=asdict(view))
                            raw = players[view.role](view)
                            log.journal.emit("step", case_index=case_index, repeat=repeat,
                                             state=asdict(view), raw_reply=raw)
        log.journal.emit("completed")
    except BaseException as exc:
        failure = exc
        log.journal.emit("interrupted", error=type(exc).__name__, message=str(exc))
        log.events.emit("run.error", error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        log.close()
    report_path = log.directory / "report.jsonl"
    summary = analyze_file(log.directory / "journal.jsonl", report_path)
    (log.directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if failure:
        raise failure
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return RunResult(log.directory, log.directory / "journal.jsonl", report_path, summary)


def audit_tools(player: PlayerConfig | None = None, *, output_dir: str | Path = "runs") -> dict:
    """One live CLI request: require Python use and retain the actual tool events."""
    settings = player or PlayerConfig()
    if settings.backend != "codex" or not settings.allow_python:
        raise ValueError("Аудит требует backend='codex', allow_python=True.")
    log = RunLog(output_dir)
    try:
        preflight = PythonSandbox().preflight()
        config = RunConfig(alice=settings, output_dir=output_dir)
        with ExitStack() as stack:
            model = CodexPlayer(config, settings, ALICE, log, persistent=True)
            stack.callback(model.close)
            reply = model.turn(
                "Проверка инструментов. Обязательно вызови инструмент local_python.python. "
                "Выполни Python: import os, socket; print(6*7); "
                "print(os.path.exists('/home')); s=socket.socket(); s.settimeout(0.3); "
                "print(s.connect_ex(('192.0.2.1',443))); s.close(). "
                "Не вызывай другие инструменты. После результата верни ровно AUDIT_OK.")
        events = [json.loads(line) for line in log.events.path.read_text().splitlines()]
        tools = [e for e in events if e["type"] == "tool.observed"]
        if not any(e["tool"] == "local_python.python" and e.get("status") == "completed" for e in tools):
            raise RuntimeError("Модель не продемонстрировала успешный вызов Python; аудит не пройден.")
        if any(not e["allowed"] for e in tools):
            raise RuntimeError("Обнаружен запрещённый инструмент.")
        python_results = []
        for path in (log.directory / "calls").glob("*/python.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event["type"] == "python.completed":
                    python_results.append(event)
        def valid_computation(event):
            lines = event["stdout"].splitlines()
            return (event["returncode"] == 0 and not event["timed_out"] and len(lines) == 3
                    and lines[:2] == ["42", "False"] and lines[2].isdigit() and int(lines[2]) != 0)
        if not any(valid_computation(e) for e in python_results):
            raise RuntimeError("Не подтверждены вычисление Python, изоляция файлов и сети в фактическом вызове.")
        result = {"passed": True, "reply": reply.strip(), "tools": tools, "python_preflight": preflight,
                  "python_results": python_results, "directory": str(log.directory),
                  "policy": tool_policy(True), "source": source_manifest()}
        (log.directory / "tool_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    except BaseException as exc:
        result = {"passed": False, "error": type(exc).__name__, "message": str(exc),
                  "directory": str(log.directory), "policy": tool_policy(True)}
        (log.directory / "tool_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    finally:
        log.close()
