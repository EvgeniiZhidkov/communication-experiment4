"""Historical pilot: pure simulation, prompt previews and old journal compatibility.
New model launches use halfduplex.experiment; the old CLI adapter was removed.
"""

"""Пилот эксперимента 4. Python 3.10+, только стандартная библиотека."""

import argparse
import json
import os
import random
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterator, TextIO


ALICE, BOB = "Alice", "Bob"
ACTIONS = {"SEND_0", "SEND_1", "LISTEN"}
ANSWERS = {"ANSWER_0", "ANSWER_1"}

FUNCTIONS = {
    "ip": "IP равна чётности числа позиций, где оба бита равны 1.",
    "disj": "DISJ равна 1, если ни в одной позиции оба бита не равны 1, иначе 0.",
}


@dataclass(frozen=True)
class Config:
    n: int
    function: str
    channel: str
    explanation: str

    @property
    def total_rounds(self) -> int:
        return self.n + 1


@dataclass(frozen=True)
class Turn:
    action: str
    observation: str | None


@dataclass(frozen=True)
class View:
    role: str
    own_input: str
    history: tuple[Turn, ...] = ()

    @property
    def round_number(self) -> int:
        return len(self.history) + 1

    def after(self, action: str, observation: str | None) -> "View":
        return View(self.role, self.own_input, self.history + (Turn(action, observation),))


@dataclass(frozen=True)
class Decision:
    view: View
    raw_reply: str

    @property
    def token(self) -> str:
        return self.raw_reply.strip()


@dataclass(frozen=True)
class Round:
    decisions: tuple[Decision, Decision]
    observations: tuple[str | None, str | None] | None


@dataclass
class Episode:
    rounds: list[Round]
    outputs: tuple[Decision, Decision] | None = None

    def decisions(self) -> Iterator[Decision]:
        for round_ in self.rounds:
            yield from round_.decisions
        if self.outputs is not None:
            yield from self.outputs


Player = Callable[[View], str]

def function_bit(function: str, x: str, y: str) -> int:
    overlap = sum(a == b == "1" for a, b in zip(x, y))
    return overlap % 2 if function == "ip" else int(overlap == 0)


def channel_step(
    alice: str, bob: str, channel: str
) -> tuple[str | None, str | None]:
    """Принимает два уже выбранных действия и выдаёт два наблюдения."""
    if alice == bob == "LISTEN":
        symbol = "SILENCE" if channel == "silence" else "0"
        return symbol, symbol
    if alice == "LISTEN":
        return bob[-1], None
    if bob == "LISTEN":
        return None, alice[-1]
    return None, None


def observed_bit(turn: Turn) -> str:
    # После отклонения SILENCE или отсутствие наблюдения трактуем как 0.
    return "1" if turn.observation == "1" else "0"


def reference_choice(config: Config, view: View) -> str:
    """Правило одной роли. У функции нет доступа к чужому входу."""
    r = view.round_number
    if r <= config.n:
        if view.role == ALICE:
            return f"SEND_{view.own_input[r - 1]}"
        return "LISTEN"

    if view.role == ALICE:
        if r == config.total_rounds:
            return "LISTEN"
        return f"ANSWER_{observed_bit(view.history[-1])}"

    received_x = "".join(observed_bit(turn) for turn in view.history[:config.n])
    result = function_bit(config.function, received_x, view.own_input)
    if r == config.total_rounds:
        return f"SEND_{result}"
    return f"ANSWER_{result}"

def play(config: Config, x: str, y: str, alice: Player, bob: Player) -> Episode:
    """Вызывает игроков с разными локальными представлениями."""
    alice_view, bob_view = View(ALICE, x), View(BOB, y)
    episode = Episode([])

    for _ in range(config.total_rounds):
        a = Decision(alice_view, alice(alice_view))
        b = Decision(bob_view, bob(bob_view))
        if a.token not in ACTIONS or b.token not in ACTIONS:
            episode.rounds.append(Round((a, b), None))
            return episode

        observations = channel_step(a.token, b.token, config.channel)
        episode.rounds.append(Round((a, b), observations))
        alice_view = alice_view.after(a.token, observations[0])
        bob_view = bob_view.after(b.token, observations[1])

    episode.outputs = (
        Decision(alice_view, alice(alice_view)),
        Decision(bob_view, bob(bob_view)),
    )
    return episode

def decision_request(config: Config, view: View) -> tuple[str, str]:
    if view.round_number <= config.total_rounds:
        return f"Выбери действие для раунда {view.round_number}.", "SEND_0, SEND_1 или LISTEN"
    return f"Выдай итоговый бит после {config.total_rounds} раундов.", "ANSWER_0 или ANSWER_1"


def render_prompt(config: Config, view: View, *, persistent: bool = False) -> str:
    if config.explanation == "prose":
        protocol = (
            "С первого по n-й раунд Alice отправляет очередной бит своего входа "
            "слева направо, а Bob слушает. В раунде n+1 Alice слушает, а Bob "
            "восстанавливает x из наблюдений за первыми n раундами, вычисляет f(x,y) "
            "и отправляет этот бит. После раунда n+1 оба дают отдельный итоговый "
            "ответ. Alice отвечает битом, который наблюдала в раунде n+1, "
            "а Bob отвечает вычисленным значением."
        )
    else:
        protocol = (
            "Индексы битов и раундов начинаются с 1. SEND(b) отправляет бит b.\n"
            "Alice: при r <= n → SEND(x[r]); при r = n+1 → LISTEN; "
            "после n+1 → ответить наблюдением раунда n+1.\n"
            "Bob: при r <= n → LISTEN; затем x' = наблюдения раундов 1..n; "
            "при r = n+1 → SEND(f(x', y)); после n+1 → ответить f(x', y)."
        )

    silence = "SILENCE" if config.channel == "silence" else "0"
    history_section = ""
    if not persistent:
        history = "\n".join(
            f"{i}. {turn.action}, наблюдение: "
            f"{turn.observation if turn.observation is not None else 'нет'}"
            for i, turn in enumerate(view.history, start=1)
        ) or "Пока пусто."
        history_section = f"\nТвоя история действий и наблюдений:\n{history}"
    task, allowed = decision_request(config, view)
    context_rule = (
        "Используй правила, свой вход и наблюдения из этой беседы. "
        "В следующих сообщениях ты будешь получать наблюдения после каждого раунда."
        if persistent
        else "Используй только данные этого сообщения."
    )

    return f"""Ты исполняешь одну роль в полудуплексном протоколе.
{context_rule} Строки входа читаются слева направо.

Общие правила
n = {config.n}. Alice получает x, Bob получает y.
Функция f: {FUNCTIONS[config.function]}
Игроки выбирают действия одновременно. Если один отправляет бит, а другой
слушает, слушатель наблюдает этот бит. Отправитель остаётся без наблюдения.
Если оба отправляют, оба остаются без наблюдения. Если оба слушают, каждый
наблюдает {silence}. После n+1 раундов каждый даёт итоговый ответ отдельно.

Заданный протокол
{protocol}
Для восстановления бита в этом протоколе наблюдения SILENCE и «нет»
считай нулём. Обычные наблюдения 0 и 1 используй как есть.

Твои данные
Роль: {view.role}
Твой вход: {view.own_input}{history_section}

{task}
Верни ровно один токен из списка: {allowed}. Без пояснений и Markdown."""


def render_followup(config: Config, view: View) -> str:
    previous = view.history[-1].observation
    observation = previous if previous is not None else "нет"
    task, allowed = decision_request(config, view)
    return (
        f"Раунд {view.round_number - 1} завершён. Твоё наблюдение: {observation}.\n"
        f"{task}\n"
        f"Верни ровно один токен из списка: {allowed}. Без пояснений и Markdown."
    )


def read_codex_reply(prompt: str) -> str:
    print(prompt, flush=True)
    first_line = input("Ответ Codex или :multi для многострочного ответа")
    if first_line != ":multi":
        return first_line

    lines = []
    while (line := input("Следующая строка или :end")) != ":end":
        lines.append(line)
    return "\n".join(lines)


def make_manual_player(config: Config, *, persistent: bool) -> Player:
    def choose(view: View) -> str:
        if persistent:
            action = "Начни новую беседу" if not view.history else "Продолжи беседу"
            print(f"{action} для роли {view.role}.", flush=True)
        if persistent and view.history:
            prompt = render_followup(config, view)
        else:
            prompt = render_prompt(config, view, persistent=persistent)
        return read_codex_reply(prompt)

    return choose






def oracle_bit(function: str, x: str, y: str) -> int:
    """Считает правильный итог по исходным входам независимо от игрока."""
    overlap_bits = int(x, 2) & int(y, 2)
    if function == "ip":
        return overlap_bits.bit_count() % 2
    return int(overlap_bits == 0)


def decision_error(config: Config, actual: Decision, expected_token: str) -> str | None:
    token = actual.token
    if token == expected_token:
        return None
    is_output = actual.view.round_number > config.total_rounds
    allowed = ANSWERS if is_output else ACTIONS
    if token not in allowed:
        if token in ACTIONS or token in ANSWERS:
            return "wrong_phase"
        return "invalid_format"
    if is_output:
        return "wrong_answer"
    if actual.view.role == BOB and actual.view.round_number == config.total_rounds:
        if token.startswith("SEND_"):
            return "wrong_result_bit"
    return "wrong_action"


def round_record(round_: Round) -> dict:
    a, b = round_.decisions
    return {
        "round": a.view.round_number,
        "alice_reply": a.raw_reply,
        "bob_reply": b.raw_reply,
        "observations": round_.observations,
    }


def episode_record(episode: Episode) -> dict:
    raw_outputs = (
        {decision.view.role: decision.raw_reply for decision in episode.outputs}
        if episode.outputs is not None else None
    )
    return {
        "rounds": [round_record(round_) for round_ in episode.rounds],
        "raw_outputs": raw_outputs,
    }


def score_episode(
    config: Config, actual: Episode, expected: Episode, truth: int
) -> dict:
    deviations = []
    local_errors = []
    for choice, reference in zip(actual.decisions(), expected.decisions()):
        local_expected = reference_choice(config, choice.view)
        error = decision_error(config, choice, local_expected)
        trace_differs = choice.token != reference.token
        if not trace_differs and error is None:
            continue

        record = {
            "role": choice.view.role,
            "round": choice.view.round_number,
            "stage": "action" if choice.view.round_number <= config.total_rounds else "output",
            "expected": reference.token,
            "expected_from_actual_history": local_expected,
            "actual": choice.raw_reply,
            "error": error,
        }
        if trace_differs:
            deviations.append(record)
        if error is not None:
            local_errors.append(record)

    answers = None
    if actual.outputs is not None:
        answers = {
            d.view.role: int(d.token[-1]) if d.token in ANSWERS else None
            for d in actual.outputs
        }

    completed_rounds = sum(r.observations is not None for r in actual.rounds)
    actions_match = completed_rounds == config.total_rounds and all(
        deviation["stage"] == "output" for deviation in deviations
    )
    return {
        "joint_correct": answers is not None and all(bit == truth for bit in answers.values()),
        "exact": not deviations and actions_match and answers is not None,
        "actions_match": actions_match,
        "answers": answers,
        "first_deviation": deviations[0] if deviations else None,
        "deviations": deviations,
        "first_local_error": local_errors[0] if local_errors else None,
        "local_errors": local_errors,
        "rounds_used": completed_rounds,
        "reference_rounds": config.total_rounds,
        **episode_record(actual),
        "reference_trace": [round_record(r) for r in expected.rounds],
    }


def log_record(log: TextIO, kind: str, **fields: object) -> None:
    print(json.dumps({"type": kind, **fields}, ensure_ascii=False), file=log, flush=True)


def metric(correct: int, total: int) -> dict:
    return {"count": correct, "total": total, "rate": correct / total if total else None}

def collect_steps(
    config: Config, cases: Iterator[tuple[str, str]], repeats: int, log: TextIO,
    codex_player: Player,
) -> None:
    reference = partial(reference_choice, config)
    seen: set[View] = set()
    for x, y in cases:
        expected = play(config, x, y, reference, reference)
        for decision in expected.decisions():
            view = decision.view
            if view in seen:
                continue
            seen.add(view)
            stage = "action" if view.round_number <= config.total_rounds else "output"

            for repeat in range(1, repeats + 1):
                raw_reply = codex_player(view)
                log_record(
                    log, "step", state=asdict(view), round=view.round_number,
                    stage=stage, repeat=repeat, raw_reply=raw_reply,
                )


def collect_episodes(
    config: Config, cases: Iterator[tuple[str, str]], repeats: int,
    players: str, log: TextIO, new_codex_player: Callable[[], Player] | None = None,
) -> None:
    reference = partial(reference_choice, config)
    a_kind, b_kind = players.split("-")
    if "codex" in (a_kind, b_kind) and new_codex_player is None:
        raise ValueError("Передай способ создать игрока Codex для нового эпизода.")

    for x, y in cases:
        for repeat in range(1, repeats + 1):
            codex_player = new_codex_player() if new_codex_player is not None else None
            alice = codex_player if a_kind == "codex" else reference
            bob = codex_player if b_kind == "codex" else reference
            actual = play(config, x, y, alice, bob)
            log_record(log, "episode", x=x, y=y, repeat=repeat, **episode_record(actual))

def restore_episode(record: dict) -> Episode:
    """Восстанавливает локальные истории по сохранённым наблюдениям."""
    alice_view, bob_view = View(ALICE, record["x"]), View(BOB, record["y"])
    episode = Episode([])
    for row in record["rounds"]:
        alice = Decision(alice_view, row["alice_reply"])
        bob = Decision(bob_view, row["bob_reply"])
        observations = row["observations"]
        if observations is None:
            episode.rounds.append(Round((alice, bob), None))
        else:
            alice_observation, bob_observation = observations
            episode.rounds.append(Round((alice, bob), (alice_observation, bob_observation)))
            alice_view = alice_view.after(alice.token, alice_observation)
            bob_view = bob_view.after(bob.token, bob_observation)

    outputs = record["raw_outputs"]
    if outputs is not None:
        episode.outputs = (
            Decision(alice_view, outputs[ALICE]),
            Decision(bob_view, outputs[BOB]),
        )
    return episode


def summarize_steps(config: Config, records: Iterator[dict], report: TextIO | None) -> dict:
    seen: set[View] = set()
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for record in records:
        state = record["state"]
        view = View(state["role"], state["own_input"], tuple(Turn(**turn) for turn in state["history"]))
        seen.add(view)
        stage = "action" if view.round_number <= config.total_rounds else "output"
        actual = Decision(view, record["raw_reply"])
        expected_token = reference_choice(config, view)
        error = decision_error(config, actual, expected_token)
        totals[stage] += 1
        if error is None:
            correct[stage] += 1
        else:
            errors[error] += 1
        if report is not None:
            log_record(
                report, "step", state=state, round=view.round_number, stage=stage,
                repeat=record["repeat"], expected=expected_token, raw_reply=actual.raw_reply, error=error,
            )
    return {
        "unique_states": len(seen),
        "next_action_accuracy": metric(correct["action"], totals["action"]),
        "output_accuracy": metric(correct["output"], totals["output"]),
        "decision_accuracy": metric(sum(correct.values()), sum(totals.values())),
        "errors": dict(errors),
    }


def summarize_episodes(config: Config, records: Iterator[dict], report: TextIO | None) -> dict:
    reference = partial(reference_choice, config)
    count = jointly_correct = exact = 0
    clean_actions = wrong_bits = output_failures = total_rounds = 0
    first_errors: Counter[str] = Counter()
    round_deviations: Counter[int] = Counter()

    for record in records:
        x, y = record["x"], record["y"]
        truth = oracle_bit(config.function, x, y)
        expected = play(config, x, y, reference, reference)
        actual = restore_episode(record)
        result = score_episode(config, actual, expected, truth)
        count += 1
        jointly_correct += result["joint_correct"]
        exact += result["exact"]
        clean_actions += result["actions_match"]
        if result["actions_match"]:
            output_failures += not result["joint_correct"]
            wrong_bits += any(
                bit is not None and bit != truth for bit in result["answers"].values()
            )
        total_rounds += result["rounds_used"]
        round_deviations[result["rounds_used"] - config.total_rounds] += 1
        if result["first_local_error"] is not None:
            first_errors[result["first_local_error"]["error"]] += 1
        if report is not None:
            log_record(report, "episode", x=x, y=y, truth=truth, repeat=record["repeat"], **result)

    return {
        "joint_accuracy": metric(jointly_correct, count),
        "exact_execution": metric(exact, count),
        "wrong_output_given_matching_actions": metric(wrong_bits, clean_actions),
        "output_failure_given_matching_actions": metric(output_failures, clean_actions),
        "mean_rounds": total_rounds / count if count else None,
        "round_delta_counts": dict(sorted(round_deviations.items())),
        "first_errors": dict(first_errors),
    }


def analyze_run(source: TextIO, report: TextIO | None = None) -> dict:
    """Пересчитывает ошибки и сводку из журнала без запуска Codex."""
    try:
        header_line = next(source, "{}")
        header = json.loads(header_line.encode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Настройки в первой строке журнала повреждены.") from exc
    if (
        not isinstance(header, dict) or header.get("type") != "config"
        or header.get("mode") not in ("step", "episode", "baseline")
    ):
        raise ValueError("Первая строка журнала должна содержать настройки прогона.")
    if header.get("protocol", "send_x_then_result_v1") != "send_x_then_result_v1":
        raise ValueError("Версия протокола в журнале не поддерживается.")
    config = Config(header["n"], header["function"], header["channel"], header["explanation"])
    mode = "step" if header["mode"] == "step" else "episode"
    run_complete = False

    def saved_records() -> Iterator[dict]:
        nonlocal run_complete
        for line_number, line in enumerate(source, start=2):
            try:
                record = json.loads(line.encode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                if not run_complete and not line.endswith("\n"):
                    # Последняя строка могла остаться недописанной при остановке процесса.
                    break
                raise ValueError(f"Строка {line_number} журнала содержит повреждённый JSON.") from exc

            kind = record.get("type") if isinstance(record, dict) else None
            if kind == mode:
                yield record
            elif kind == "completed":
                run_complete = True
            elif kind == "summary":
                # В старых журналах summary отмечала завершение.
                run_complete = record.get("run_complete", True)
            elif kind == "interrupted":
                run_complete = False
            else:
                raise ValueError(f"В строке {line_number} неожиданный тип записи для режима {mode}.")

    if report is not None:
        log_record(report, "config", **{key: value for key, value in header.items() if key != "type"})
    summarize = summarize_steps if mode == "step" else summarize_episodes
    summary = summarize(config, saved_records(), report)
    summary["run_complete"] = run_complete
    if report is not None:
        log_record(report, "summary", **summary)
    return summary


def analyze_file(log_path: Path) -> dict:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    report_path = log_path.with_name(f"{log_path.stem}_analysis_{timestamp}.jsonl")
    # Сохраняем ошибочные байты до проверки отдельной строки.
    with log_path.open(encoding="utf-8", errors="surrogateescape") as source:
        report = report_path.open("x", encoding="utf-8")
        try:
            with report:
                summary = analyze_run(source, report)
        except BaseException:
            report_path.unlink(missing_ok=True)
            raise
    print(f"Отчёт: {report_path.resolve()}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary

def input_pairs(args: argparse.Namespace) -> Iterator[tuple[str, str]]:
    if args.case:
        yield tuple(args.case)
        return
    width = 2 ** args.n
    indices = range(width ** 2)
    if args.limit is not None:
        # Одинаковые n, limit и seed дают один и тот же набор пар.
        indices = random.Random(args.seed).sample(indices, args.limit)
    for i in indices:
        x, y = divmod(i, width)
        yield f"{x:0{args.n}b}", f"{y:0{args.n}b}"