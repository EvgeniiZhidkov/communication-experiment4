"""Experiment settings are data; notebooks contain no simulation logic."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path

from .core import validate_inputs
from .protocols import Protocol, get_protocol, resolve_selection


@dataclass(frozen=True)
class PlayerConfig:
    backend: str = "codex"  # codex, python (reference), manual
    model: str | None = None  # None = the CLI's built-in default, not user config
    reasoning_effort: str | None = None
    timeout_seconds: float = 120
    allow_python: bool = True

    def __post_init__(self):
        if self.backend not in {"codex", "python", "manual"}:
            raise ValueError(f"Неизвестный backend: {self.backend}")
        if (not isinstance(self.timeout_seconds, (int, float)) or isinstance(self.timeout_seconds, bool)
                or not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("Таймаут должен быть положительным конечным числом.")
        if not isinstance(self.allow_python, bool):
            raise ValueError("allow_python должно быть логическим значением True или False.")
        if self.reasoning_effort not in {None, "minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError("Неизвестный уровень reasoning_effort.")


@dataclass(frozen=True)
class RunConfig:
    protocol: str = "specialized"  # baseline or specialized
    n: int = 5
    experiment: str = "episode"  # episode or step; both are E4
    alice: PlayerConfig = field(default_factory=PlayerConfig)
    bob: PlayerConfig = field(default_factory=PlayerConfig)
    task: str | None = None  # defaults to ip; old protocol aliases can supply it
    channel: str | None = None  # defaults to silence; old aliases can supply it
    explanation: str = "prose"
    input_format: str = "bits"  # bits, bit_list, positions
    sampling: str = "uniform"  # uniform, eq_balanced (explicitly a different distribution)
    case_limit: int | None = 15
    case: tuple[str, str] | None = None  # used when case_limit is None
    seed: int = 42
    repeats: int = 1
    round_limit: int | None = None
    output_dir: str | Path = "runs"
    codex_executable: str = "codex"
    fsync: bool = True

    def __post_init__(self):
        protocol, task, channel = resolve_selection(self.protocol, self.task, self.channel)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "channel", channel)
        if not isinstance(self.n, int) or isinstance(self.n, bool) or not 1 <= self.n <= 1024:
            raise ValueError("n должно быть целым числом от 1 до 1024.")
        if self.experiment not in {"episode", "step"}:
            raise ValueError("experiment: episode или step (воспроизведение E4).")
        if self.explanation not in {"prose", "pseudocode"}:
            raise ValueError("explanation: prose или pseudocode.")
        if self.input_format not in {"bits", "bit_list", "positions"}:
            raise ValueError("input_format: bits, bit_list или positions.")
        if self.sampling not in {"uniform", "eq_balanced"}:
            raise ValueError("sampling: uniform или eq_balanced.")
        if self.sampling == "eq_balanced" and self.task != "eq":
            raise ValueError("eq_balanced применим только к EQ.")
        if self.case_limit is not None and (
            not isinstance(self.case_limit, int) or isinstance(self.case_limit, bool)
            or not 1 <= self.case_limit <= 4 ** self.n
        ):
            raise ValueError("case_limit должно быть от 1 до 4**n, либо None.")
        if self.sampling == "eq_balanced" and self.case_limit is not None:
            if (self.case_limit + 1) // 2 > 2 ** self.n:
                raise ValueError("Для такого размера сбалансированной выборки недостаточно равных пар.")
        if self.case_limit is None:
            if self.case is None:
                raise ValueError("При case_limit=None задайте case=(x, y).")
            validate_inputs(self.n, *self.case)
        if not isinstance(self.repeats, int) or isinstance(self.repeats, bool) or self.repeats < 1:
            raise ValueError("repeats должно быть положительным целым числом.")
        if self.round_limit is not None and (not isinstance(self.round_limit, int)
                                             or isinstance(self.round_limit, bool) or self.round_limit < 0):
            raise ValueError("round_limit должно быть неотрицательным целым числом.")
        if not isinstance(self.fsync, bool):
            raise ValueError("fsync должно быть логическим значением True или False.")

    @property
    def implementation(self) -> Protocol:
        return get_protocol(self.protocol, self.task, self.channel)

    @property
    def max_rounds(self) -> int:
        return self.round_limit if self.round_limit is not None else self.implementation.rounds(self.n)

    def record(self) -> dict:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        return result
