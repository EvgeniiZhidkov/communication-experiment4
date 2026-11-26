"""Deterministic local protocols. Every choice uses one private View only.

IP: Dementiev et al., ECCC TR26-112, Theorem 3.
EQ: Hoover et al., ECCC TR18-089 revision 2, Theorems 6 and 10.
The EQ counter widths below include zero and avoid the n=1 edge case.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .core import ALICE, View, function_bit

IP_SOURCE = "https://eccc.weizmann.ac.il/report/2026/112/download/"
EQ_SOURCE = "https://eccc.weizmann.ac.il/report/2018/089/revision/2/download/"


def digits(value: int, base: int, width: int) -> tuple[int, ...]:
    result = [0] * width
    for i in range(width - 1, -1, -1):
        value, result[i] = divmod(value, base)
    if value:
        raise ValueError("Число не помещается в заданную запись.")
    return tuple(result)


def capacity_width(size: int, base: int) -> int:
    """Smallest k with base**k >= size; exact integer arithmetic."""
    width, capacity = 0, 1
    while capacity < size:
        width, capacity = width + 1, capacity * base
    return width


def received_bit(observation: str | None) -> int:
    if observation not in {"0", "1"}:
        raise ValueError(f"Ожидался полученный бит, получено {observation!r}.")
    return int(observation)


def decode(observations, base: int) -> int:
    value = 0
    for o in observations:
        digit = 2 if base == 3 and o == "SILENCE" else received_bit(o)
        value = value * base + digit
    return value


def baseline(view: View, n: int, task: str) -> str:
    r = len(view.history)
    if r < n:
        return f"SEND_{view.own_input[r]}" if view.role == ALICE else "LISTEN"
    if view.role == ALICE:
        return "LISTEN" if r == n else f"ANSWER_{received_bit(view.history[n].observation)}"
    x = "".join(str(received_bit(t.observation)) for t in view.history[:n])
    result = function_bit(task, x, view.own_input)
    return f"SEND_{result}" if r == n else f"ANSWER_{result}"


def ip_silence(view: View, n: int, task: str) -> str:
    k = (n + 1) // 2
    bits = view.own_input + ("0" if n % 2 else "")
    blocks = [bits[i:i + 2] for i in range(0, len(bits), 2)]
    mapping = ({"00": "SEND_0", "01": "SEND_1", "10": "LISTEN", "11": "LISTEN"}
               if view.role == ALICE else
               {"00": "SEND_0", "01": "LISTEN", "10": "SEND_1", "11": "LISTEN"})
    r = len(view.history)
    if r < k:
        return mapping[blocks[r]]
    accumulator = 0
    for block, turn in zip(blocks, view.history[:k]):
        flip = ((block in {"01", "10"} and turn.observation == "1")
                or (block == "11" and turn.observation != "0"))
        accumulator ^= int(flip)
    if r == k:
        return f"SEND_{accumulator}" if view.role == ALICE else "LISTEN"
    if view.role == ALICE:
        return "LISTEN" if r == k + 1 else f"ANSWER_{received_bit(view.history[k + 1].observation)}"
    answer = accumulator ^ received_bit(view.history[k].observation)
    return f"SEND_{answer}" if r == k + 1 else f"ANSWER_{answer}"


def eq_silence(view: View, n: int, task: str) -> str:
    k = capacity_width(2 ** n, 5)
    width = capacity_width(k + 1, 3)
    symbols = digits(int(view.own_input, 2), 5, k)
    actions = (("SEND_0", "SEND_1", "LISTEN", "LISTEN", "LISTEN")
               if view.role == ALICE else
               ("LISTEN", "LISTEN", "SEND_0", "SEND_1", "LISTEN"))
    expected = ({2: "0", 3: "1", 4: "SILENCE"} if view.role == ALICE
                else {0: "0", 1: "1", 4: "SILENCE"})
    r = len(view.history)
    if r < k:
        return actions[symbols[r]]
    mismatch = any(t.action == "LISTEN" and t.observation != expected[s]
                   for s, t in zip(symbols, view.history[:k]))
    if r == k:
        return "SEND_0" if mismatch else "LISTEN"
    flag = view.history[k]
    if flag.action.startswith("SEND_") or flag.observation != "SILENCE":
        return "ANSWER_0"
    if r < k + 1 + width:
        if view.role != ALICE:
            return "LISTEN"
        count = sum(t.action == "LISTEN" and t.observation in {"0", "1"}
                    for t in view.history[:k])
        digit = digits(count, 3, width)[r - k - 1]
        return "LISTEN" if digit == 2 else f"SEND_{digit}"
    if view.role == ALICE:
        return "LISTEN" if r == k + 1 + width else f"ANSWER_{received_bit(view.history[-1].observation)}"
    count = decode((t.observation for t in view.history[k + 1:k + 1 + width]), 3)
    sent = sum(t.action.startswith("SEND_") for t in view.history[:k])
    answer = int(count == sent)
    return f"SEND_{answer}" if r == k + 1 + width else f"ANSWER_{answer}"


def eq_zero(view: View, n: int, task: str) -> str:
    k = capacity_width(2 ** n, 3)
    width = capacity_width(k + 1, 2)
    symbols = digits(int(view.own_input, 2), 3, k)
    actions = (("LISTEN", "SEND_1", "LISTEN") if view.role == ALICE
               else ("LISTEN", "LISTEN", "SEND_1"))
    expected = {0: "0", 2: "1"} if view.role == ALICE else {0: "0", 1: "1"}
    r = len(view.history)
    if r < k:
        return actions[symbols[r]]
    sent = sum(t.action == "SEND_1" for t in view.history[:k])
    heard = sum(t.action == "LISTEN" and t.observation == "1" for t in view.history[:k])
    if r < k + width:
        return f"SEND_{digits(sent, 2, width)[r - k]}" if view.role == ALICE else "LISTEN"
    if r < k + 2 * width:
        return f"SEND_{digits(sent, 2, width)[r - k - width]}" if view.role != ALICE else "LISTEN"
    count_history = (view.history[k + width:k + 2 * width] if view.role == ALICE
                     else view.history[k:k + width])
    other_sent = decode((t.observation for t in count_history), 2)
    mismatch = other_sent != heard or any(
        t.action == "LISTEN" and t.observation != expected[s]
        for s, t in zip(symbols, view.history[:k])
    )
    if r == k + 2 * width:
        return "SEND_1" if mismatch else "LISTEN"
    flag = view.history[-1]
    return f"ANSWER_{int(not (mismatch or flag.action == 'SEND_1' or flag.observation == '1'))}"


@dataclass(frozen=True)
class Protocol:
    name: str
    task: str | None
    channel: str | None
    choose: Callable[[View, int, str], str]
    rounds: Callable[[int], int]
    source: str
    formula: str


# Public exchange strategies; task and channel are separate settings.
PROTOCOL_MODES = ("baseline", "specialized")

# Concrete implementations, also kept as aliases for saved configurations.
PROTOCOLS = {
    "send_x_then_result": Protocol("send_x_then_result", None, None, baseline,
                                    lambda n: n + 1, "", "n + 1"),
    "ip_silence": Protocol("ip_silence", "ip", "silence", ip_silence,
                            lambda n: (n + 1) // 2 + 2, IP_SOURCE, "ceil(n/2) + 2"),
    "eq_silence": Protocol("eq_silence", "eq", "silence", eq_silence,
                            lambda n: (k := capacity_width(2 ** n, 5)) + capacity_width(k + 1, 3) + 2,
                            EQ_SOURCE, "k + ceil(log_3(k+1)) + 2; k = ceil(n/log_2(5))"),
    "eq_zero": Protocol("eq_zero", "eq", "zero", eq_zero,
                         lambda n: (k := capacity_width(2 ** n, 3)) + 2 * capacity_width(k + 1, 2) + 1,
                         EQ_SOURCE, "k + 2*ceil(log_2(k+1)) + 1; k = ceil(n/log_2(3))"),
}


def resolve_selection(name: str, task: str | None = None,
                      channel: str | None = None) -> tuple[str, str, str]:
    """Normalize old names, then validate the independent experiment settings."""
    legacy = PROTOCOLS.get(name)
    if legacy is not None:
        if ((legacy.task is not None and task is not None and legacy.task != task)
                or (legacy.channel is not None and channel is not None and legacy.channel != channel)):
            raise ValueError(f"Протокол {name!r} несовместим с выбранной задачей или каналом.")
        task = legacy.task if task is None else task
        channel = legacy.channel if channel is None else channel
        name = "baseline" if legacy.choose is baseline else "specialized"
    elif name not in PROTOCOL_MODES:
        raise ValueError(f"Неизвестный протокол {name!r}. Доступны: {', '.join(PROTOCOL_MODES)}")
    task = "ip" if task is None else task
    channel = "silence" if channel is None else channel
    if task not in {"ip", "eq", "disj"}:
        raise ValueError(f"Неизвестная задача {task!r}. Доступны: ip, eq, disj.")
    if channel not in {"silence", "zero"}:
        raise ValueError(f"Неизвестный канал {channel!r}. Доступны: silence, zero.")
    if name == "specialized" and f"{task}_{channel}" not in PROTOCOLS:
        raise ValueError(
            f"Для protocol='specialized', task={task!r}, channel={channel!r} нет реализации. "
            "Поддерживаются IP/silence, EQ/silence и EQ/zero; "
            "protocol='baseline' поддерживает все задачи и оба канала.")
    return name, task, channel


def get_protocol(name: str, task: str | None = None, channel: str | None = None) -> Protocol:
    """Select a concrete implementation; legacy names remain readable."""
    mode, task, channel = resolve_selection(name, task, channel)
    key = "send_x_then_result" if mode == "baseline" else f"{task}_{channel}"
    return PROTOCOLS[key]
