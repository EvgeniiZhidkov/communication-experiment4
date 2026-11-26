"""Local views, channel semantics and a simulator without a model dependency."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator

ALICE, BOB = "Alice", "Bob"
ACTIONS = frozenset({"SEND_0", "SEND_1", "LISTEN"})
ANSWERS = frozenset({"ANSWER_0", "ANSWER_1"})


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

    def after(self, action: str, observation: str | None) -> View:
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
    observations: tuple[str | None, str | None]


@dataclass
class Episode:
    rounds: list[Round] = field(default_factory=list)
    terminal: tuple[Decision, Decision] | None = None
    status: str = "running"

    def decisions(self) -> Iterator[Decision]:
        for event in self.rounds:
            yield from event.decisions
        if self.terminal is not None:
            yield from self.terminal

    def record(self) -> dict:
        outputs = None
        if self.terminal is not None:
            outputs = {d.view.role: d.raw_reply for d in self.terminal}
        return {
            "status": self.status,
            "rounds": [
                {"round": i, "alice_reply": r.decisions[0].raw_reply,
                 "bob_reply": r.decisions[1].raw_reply, "observations": list(r.observations)}
                for i, r in enumerate(self.rounds, 1)
            ],
            "raw_outputs": outputs,
        }


Player = Callable[[View], str]


def validate_inputs(n: int, x: str, y: str) -> None:
    if any(len(bits) != n or set(bits) - {"0", "1"} for bits in (x, y)):
        raise ValueError(f"Оба входа должны быть битовыми строками длины {n}.")


def function_bit(task: str, x: str, y: str) -> int:
    if len(x) != len(y) or (set(x) | set(y)) - {"0", "1"}:
        raise ValueError("Некорректные битовые строки.")
    if task == "ip":
        return (int(x, 2) & int(y, 2)).bit_count() % 2
    if task == "disj":
        return int((int(x, 2) & int(y, 2)) == 0)
    if task == "eq":
        return int(x == y)
    raise ValueError(f"Неизвестная задача: {task}")


def channel_step(alice: str, bob: str, channel: str) -> tuple[str | None, str | None]:
    if alice not in ACTIONS or bob not in ACTIONS:
        raise ValueError("Канал принимает только SEND_0, SEND_1 и LISTEN.")
    if channel not in {"silence", "zero"}:
        raise ValueError(f"Неизвестный канал: {channel}")
    if alice == bob == "LISTEN":
        symbol = "SILENCE" if channel == "silence" else "0"
        return symbol, symbol
    if alice == "LISTEN":
        return bob[-1], None
    if bob == "LISTEN":
        return None, alice[-1]
    return None, None


def play(n: int, x: str, y: str, alice: Player, bob: Player, *, channel: str,
         max_rounds: int, emit: Callable[..., None] | None = None) -> Episode:
    """Obtain both choices before delivering either local observation.

    ANSWER is a local stop decision and costs no channel round. An asymmetric
    stop fails; its answer is never transmitted to the other player.
    """
    validate_inputs(n, x, y)
    if max_rounds < 0:
        raise ValueError("Лимит раундов должен быть неотрицательным.")
    views = (View(ALICE, x), View(BOB, y))
    episode = Episode()
    while True:
        decisions = []
        for player, view in zip((alice, bob), views):
            decision = Decision(view, player(view))
            decisions.append(decision)
            if emit:
                emit("decision", state=asdict(view), raw_reply=decision.raw_reply)
        a, b = decisions
        tokens = (a.token, b.token)
        if any(t not in ACTIONS | ANSWERS for t in tokens):
            episode.status = "invalid_token"
        elif all(t in ANSWERS for t in tokens):
            episode.status = "completed"
        elif any(t in ANSWERS for t in tokens):
            episode.status = "asymmetric_stop"
        elif len(episode.rounds) >= max_rounds:
            episode.status = "round_limit"
        else:
            observations = channel_step(*tokens, channel)
            episode.rounds.append(Round((a, b), observations))
            if emit:
                emit("round", round=len(episode.rounds), alice_reply=a.raw_reply,
                     bob_reply=b.raw_reply, observations=observations)
            views = tuple(v.after(t, o) for v, t, o in zip(views, tokens, observations))
            continue
        episode.terminal = (a, b)
        return episode
