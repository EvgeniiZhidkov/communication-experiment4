"""Recalculate metrics from saved decisions; no model calls."""
from __future__ import annotations

import itertools
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import PlayerConfig, RunConfig
from .core import ALICE, ANSWERS, BOB, Decision, Episode, Round, Turn, View, channel_step, function_bit, play


def metric(count: int, total: int) -> dict:
    return {"count": count, "total": total, "rate": count / total if total else None}


def reference_player(config: RunConfig):
    protocol = config.implementation
    return lambda view: protocol.choose(view, config.n, config.task)


def restore_episode(record: dict, channel: str) -> Episode:
    views = (View(ALICE, record["x"]), View(BOB, record["y"]))
    episode = Episode(status=record.get("status", "completed"))
    for row in record["rounds"]:
        a, b = Decision(views[0], row["alice_reply"]), Decision(views[1], row["bob_reply"])
        observations = tuple(row["observations"])
        if observations != channel_step(a.token, b.token, channel):
            raise ValueError("Наблюдения в журнале не соответствуют действиям и каналу.")
        episode.rounds.append(Round((a, b), observations))
        views = tuple(v.after(d.token, o) for v, d, o in zip(views, (a, b), observations))
    outputs = record.get("raw_outputs")
    if outputs is not None:
        episode.terminal = (Decision(views[0], outputs[ALICE]), Decision(views[1], outputs[BOB]))
    return episode


def decision_error(actual: Decision, expected: str) -> str | None:
    token = actual.token
    if token == expected:
        return None
    if token not in {"SEND_0", "SEND_1", "LISTEN"} | ANSWERS:
        return "invalid_format"
    if (token in ANSWERS) != (expected in ANSWERS):
        return "wrong_phase"
    return "wrong_answer" if expected in ANSWERS else "wrong_action"


def score_episode(config: RunConfig, actual: Episode, expected: Episode, truth: int) -> dict:
    reference = reference_player(config)
    deviations, local_errors = [], []
    actual_decisions, expected_decisions = list(actual.decisions()), list(expected.decisions())
    for choice, wanted in itertools.zip_longest(actual_decisions, expected_decisions):
        anchor = choice or wanted
        if choice is not None:
            try:
                local_expected = reference(choice.view)
                error = decision_error(choice, local_expected)
            except (ValueError, IndexError, KeyError):
                local_expected, error = None, "invalid_history"
        else:
            local_expected, error = None, "missing_decision"
        differs = choice is None or wanted is None or choice.token != wanted.token
        row = {"role": anchor.view.role, "round": anchor.view.round_number,
               "stage": "output" if (wanted and wanted.token in ANSWERS) else "action",
               "expected": wanted.token if wanted else None,
               "expected_from_actual_history": local_expected,
               "actual": choice.raw_reply if choice else None, "error": error}
        if differs:
            deviations.append(row)
        if error is not None:
            local_errors.append(row)
    answers = None
    if actual.terminal is not None:
        answers = {d.view.role: int(d.token[-1]) if d.token in ANSWERS else None for d in actual.terminal}
    joint = actual.status == "completed" and answers is not None and all(v == truth for v in answers.values())
    actions_match = (len(actual.rounds) == len(expected.rounds)
                     and all(tuple(d.token for d in a.decisions) == tuple(d.token for d in e.decisions)
                             for a, e in zip(actual.rounds, expected.rounds)))
    return {"truth": truth, "joint_correct": joint,
            "exact": actual.status == "completed" and not deviations,
            "actions_match": actions_match, "answers": answers,
            "first_deviation": deviations[0] if deviations else None, "deviations": deviations,
            "first_local_error": local_errors[0] if local_errors else None, "local_errors": local_errors,
            "rounds_used": len(actual.rounds), "reference_rounds": len(expected.rounds),
            **actual.record()}


def _write(stream, kind: str, **values):
    if stream is not None:
        print(json.dumps({"type": kind, **values}, ensure_ascii=False), file=stream, flush=True)


def _read_record(line: str) -> dict:
    # analyze_file uses surrogateescape so malformed bytes remain detectable.
    line.encode("utf-8")
    record = json.loads(line)
    if not isinstance(record, dict):
        raise ValueError("Запись журнала должна быть JSON-объектом.")
    return record


def analyze_run(source, report=None) -> dict:
    first = next(source, "")
    header = _read_record(first)
    if not isinstance(header, dict) or header.get("type") != "config":
        raise ValueError("Первая строка должна содержать настройки прогона.")
    if "schema_version" not in header:
        from .legacy import analyze_run as legacy_analyze
        return legacy_analyze(iter(itertools.chain([first], source)), report)
    if header["schema_version"] != 2:
        raise ValueError("Неподдерживаемая версия журнала.")
    values = dict(header["config"])
    values["alice"] = PlayerConfig(**values["alice"])
    values["bob"] = PlayerConfig(**values["bob"])
    config = RunConfig(**values)
    reference = reference_player(config)
    counts, correct, errors, delta = Counter(), Counter(), Counter(), Counter()
    classes = {"0": {"count": 0, "joint_correct": 0}, "1": {"count": 0, "joint_correct": 0}}
    rounds_total, max_observed = 0, 0
    run_complete = False
    _write(report, "config", **{k: v for k, v in header.items() if k != "type"})
    for line in source:
        try:
            row = _read_record(line)
        except (json.JSONDecodeError, UnicodeError):
            if not run_complete and not line.endswith("\n"):
                break
            raise
        kind = row.get("type")
        if kind == "completed":
            run_complete = True
            continue
        if kind == "interrupted":
            run_complete = False
            continue
        if kind == "summary":
            run_complete = row.get("run_complete", True)
            continue
        if run_complete:
            raise ValueError("После завершения прогона обнаружены новые решения.")
        if kind == "episode" and config.experiment == "episode":
            actual = restore_episode(row, config.channel)
            expected = play(config.n, row["x"], row["y"], reference, reference,
                            channel=config.channel, max_rounds=config.implementation.rounds(config.n))
            truth = function_bit(config.task, row["x"], row["y"])
            result = score_episode(config, actual, expected, truth)
            counts["episode"] += 1
            correct["joint"] += result["joint_correct"]
            correct["exact"] += result["exact"]
            counts["matching_actions"] += result["actions_match"]
            if result["actions_match"]:
                counts["output_failure"] += not result["joint_correct"]
                counts["wrong_output"] += any(v is not None and v != truth for v in (result["answers"] or {}).values())
            rounds_total += result["rounds_used"]
            max_observed = max(max_observed, result["rounds_used"])
            delta[result["rounds_used"] - result["reference_rounds"]] += 1
            if result["first_local_error"]:
                errors[result["first_local_error"]["error"]] += 1
            classes[str(truth)]["count"] += 1
            classes[str(truth)]["joint_correct"] += result["joint_correct"]
            _write(report, "episode", x=row["x"], y=row["y"], repeat=row["repeat"], **result)
        elif kind == "step" and config.experiment == "step":
            state = row["state"]
            view = View(state["role"], state["own_input"], tuple(Turn(**t) for t in state["history"]))
            expected = reference(view)
            decision = Decision(view, row["raw_reply"])
            stage = "output" if expected in ANSWERS else "action"
            error = decision_error(decision, expected)
            counts[stage] += 1
            correct[stage] += error is None
            if error:
                errors[error] += 1
            _write(report, "step", **{k: v for k, v in row.items() if k not in {"type", "expected", "error"}},
                   expected=expected, error=error)
        else:
            raise ValueError(f"Неожиданный тип записи {kind!r}.")
    if config.experiment == "episode":
        total = counts["episode"]
        summary = {"joint_accuracy": metric(correct["joint"], total),
                   "exact_execution": metric(correct["exact"], total),
                   "wrong_output_given_matching_actions": metric(counts["wrong_output"], counts["matching_actions"]),
                   "output_failure_given_matching_actions": metric(counts["output_failure"], counts["matching_actions"]),
                   "mean_rounds": rounds_total / total if total else None, "max_observed_rounds": max_observed if total else None,
                   "reference_max_rounds": config.implementation.rounds(config.n),
                   "round_delta_counts": dict(sorted(delta.items())), "first_errors": dict(errors),
                   "by_class": {label: metric(c["joint_correct"], c["count"]) for label, c in classes.items()}}
    else:
        summary = {"next_action_accuracy": metric(correct["action"], counts["action"]),
                   "output_accuracy": metric(correct["output"], counts["output"]),
                   "decision_accuracy": metric(sum(correct.values()), sum(counts.values())), "errors": dict(errors)}
    summary["run_complete"] = run_complete
    _write(report, "summary", **summary)
    return summary


def analyze_file(log_path: str | Path, report_path: str | Path | None = None) -> dict:
    log_path = Path(log_path)
    if report_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        report_path = log_path.with_name(f"{log_path.stem}_analysis_{stamp}.jsonl")
    with log_path.open(encoding="utf-8", errors="surrogateescape") as source:
        report_path = Path(report_path)
        report = report_path.open("x", encoding="utf-8")
        try:
            with report:
                return analyze_run(source, report)
        except BaseException:
            report_path.unlink(missing_ok=True)
            raise


def verify_reference(config: RunConfig, *, max_pairs: int = 1_048_576) -> dict:
    total = 4 ** config.n
    if total > max_pairs:
        raise ValueError(f"Полный перебор: {total} пар; лимит проверки {max_pairs}.")
    player = reference_player(config)
    implementation = config.implementation
    bound = implementation.rounds(config.n)
    counts = Counter()
    for a in range(2 ** config.n):
        for b in range(2 ** config.n):
            x, y = f"{a:0{config.n}b}", f"{b:0{config.n}b}"
            actual = play(config.n, x, y, player, player, channel=config.channel, max_rounds=bound)
            truth = function_bit(config.task, x, y)
            if actual.status != "completed" or not all(d.token == f"ANSWER_{truth}" for d in actual.terminal):
                raise AssertionError(f"Эталон неверен: {config.protocol}, x={x}, y={y}, {actual}")
            counts[len(actual.rounds)] += 1
    return {"protocol": config.protocol, "task": config.task, "channel": config.channel,
            "implementation": implementation.name, "n": config.n, "pairs": total, "correct": total,
            "exhaustive": True, "c_max": max(counts), "round_counts": dict(sorted(counts.items())),
            "implemented_upper_bound": bound, "bound_formula": implementation.formula}
