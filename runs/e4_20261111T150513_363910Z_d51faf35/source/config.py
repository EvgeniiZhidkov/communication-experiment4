"""Experiment settings are data; notebooks contain no simulation logic."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .core import validate_inputs
from .protocols import get_protocol


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
        if self.timeout_seconds <= 0:
            raise ValueError("Таймаут должен быть положительным.")
        if self.reasoning_effort not in {None, "minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError("Неизвестный уровень reasoning_effort.")


@dataclass(frozen=True)
class RunConfig:
    protocol: str = "ip_silence"
    n: int = 5
    experiment: str = "episode"  # episode or step; both are E4
    alice: PlayerConfig = field(default_factory=PlayerConfig)
    bob: PlayerConfig = field(default_factory=PlayerConfig)
    task: str | None = None  # inferred from the protocol; set for the baseline
    channel: str | None = None
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
        p = get_protocol(self.protocol)
        object.__setattr__(self, "task", self.task or p.task or "ip")
        object.__setattr__(self, "channel", self.channel or p.channel or "silence")
        if not isinstance(self.n, int) or isinstance(self.n, bool) or not 1 <= self.n <= 1024:
            raise ValueError("n должно быть целым числом от 1 до 1024.")
        if self.task not in {"ip", "eq", "disj"} or self.channel not in {"silence", "zero"}:
            raise ValueError("Неизвестная задача или канал.")
        if (p.task and p.task != self.task) or (p.channel and p.channel != self.channel):
            raise ValueError("Протокол несовместим с выбранной задачей или каналом.")
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
        if self.round_limit is not None and (not isinstance(self.round_limit, int) or self.round_limit < 0):
            raise ValueError("round_limit должно быть неотрицательным целым числом.")

    @property
    def max_rounds(self) -> int:
        return self.round_limit if self.round_limit is not None else get_protocol(self.protocol).rounds(self.n)

    def record(self) -> dict:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        return result
