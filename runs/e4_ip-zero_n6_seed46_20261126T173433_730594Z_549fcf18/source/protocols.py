"""Deterministic local protocols. Every choice uses one private View only.

IP and DISJ: Dementiev et al., ECCC TR26-112, Theorems 3, 5 and 6.
EQ: Hoover et al., ECCC TR18-089 revision 2, Theorems 6 and 10.
The EQ counter widths below include zero and avoid the n=1 edge case.
"""
from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Callable

from .core import ALICE, ANSWERS, BOB, View, function_bit

DEMENTIEV_SOURCE = "https://eccc.weizmann.ac.il/report/2026/112/download/"
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
class LocalMove:
    token: str
    phase: str
    unit: str = ""


LocalSteps = Generator[LocalMove, str | None, None]


def replay_local_steps(view: View, steps: LocalSteps) -> LocalMove:
    """Replay a role's own history to recover its next action and phase."""
    move = next(steps)
    for turn in view.history:
        if turn.action != move.token:
            raise ValueError("История действий отклонилась от локального протокола.")
        move = steps.send(turn.observation)
    return move


def transmit_number(role: str, sender: str, value: int, width: int, phase: str) -> Generator[LocalMove, str | None, int]:
    """Send an unsigned integer most-significant bit first, or receive it."""
    if role == sender:
        for bit in digits(value, 2, width):
            yield LocalMove(f"SEND_{bit}", phase)
        return value
    received = 0
    for _ in range(width):
        observation = yield LocalMove("LISTEN", phase)
        received = 2 * received + received_bit(observation)
    return received


# For each most frequent Alice pair, list (Alice, Bob) triggers for both passes.
IP_ZERO_TRIGGERS = {
    "00": (("00", "00"), ("01", "10")),
    "01": (("01", "10"), ("00", "00")),
    "10": (("10", "01"), ("11", "11")),
    "11": (("11", "11"), ("10", "01")),
}
IP_ZERO_PARTNER = {"00": "11", "01": "01", "10": "10", "11": "00"}


def ip_zero_steps(role: str, bits: str) -> LocalSteps:
    padded = bits + ("0" if len(bits) % 2 else "")
    pairs = [padded[i:i + 2] for i in range(0, len(padded), 2)]
    most_frequent = max(IP_ZERO_TRIGGERS, key=pairs.count) if role == ALICE else "00"
    announced = yield from transmit_number(role, ALICE, int(most_frequent, 2), 2, "Alice's most frequent pair")
    parity = pairs.count("11") % 2
    active = pairs
    for pass_number, (alice_trigger, bob_trigger) in enumerate(IP_ZERO_TRIGGERS[f"{announced:02b}"], 1):
        own_trigger, other_trigger = ((alice_trigger, bob_trigger) if role == ALICE
                                      else (bob_trigger, alice_trigger))
        silent = []
        for pair in active:
            action = "SEND_1" if pair == own_trigger else "LISTEN"
            observation = yield LocalMove(action, f"IP pass {pass_number}", pair)
            if observation == "0":
                silent.append(pair)
            elif observation == "1" and IP_ZERO_PARTNER[pair] == other_trigger:
                parity ^= 1
        active = silent
    alice_parity = yield from transmit_number(role, ALICE, parity, 1, "Alice's accumulator")
    answer = yield from transmit_number(role, BOB, parity ^ alice_parity, 1, "Return XOR result")
    yield LocalMove(f"ANSWER_{answer}", "Answer")


def ip_zero(view: View, n: int, task: str) -> str:
    return replay_local_steps(view, ip_zero_steps(view.role, view.own_input)).token


def ip_zero_rounds(n: int) -> int:
    k = (n + 1) // 2
    return 4 + k + (3 * k) // 4


DISJ_ALPHA = 2559
DISJ_DENOMINATOR = 5000


def run_subprotocol(role: str, bits: str, task: str, choose: Callable[[View, int, str], str],
                    phase: str) -> LocalSteps:
    view = View(role, bits)
    while True:
        token = choose(view, len(bits), task)
        observation = yield LocalMove(token, phase)
        if token in ANSWERS:
            return
        view = view.after(token, observation)


def disj_zero_steps(role: str, bits: str) -> LocalSteps:
    n = len(bits)
    if n == 0:
        yield LocalMove("ANSWER_1", "Empty instance")
        return
    if n == 1:
        yield from run_subprotocol(role, bits, "disj", baseline, "Single-bit DISJ")
        return

    if n % 2:
        trailing = int(bits[-1])
        alice_bit = yield from transmit_number(role, ALICE, trailing, 1, "Alice's last bit")
        bob_bit = yield from transmit_number(role, BOB, trailing, 1, "Bob's last bit")
        if alice_bit & bob_bit:
            yield LocalMove("ANSWER_0", "Intersection at the last bit")
            return
        bits = bits[:-1]

    pairs = [bits[i:i + 2] for i in range(0, len(bits), 2)]
    k = len(pairs)
    counts = {ALICE: {}, BOB: {}}
    for sender in (BOB, ALICE):
        for pair in ("00", "01", "10"):
            label = f"{sender}'s count of {pair}"
            counts[sender][pair] = yield from transmit_number(role, sender, pairs.count(pair), k.bit_length(), label)
        remainder = k - sum(counts[sender].values())
        if remainder < 0:
            raise ValueError("Счётчики пар превышают длину ввода.")
        counts[sender]["11"] = remainder

    a, b = counts[ALICE], counts[BOB]
    # Allow equality in the threshold tests so the four choices cover exact ties.
    choices = (("01", a["01"] + b["01"], DISJ_ALPHA),
               ("00", a["00"] + a["11"], DISJ_DENOMINATOR - DISJ_ALPHA),
               ("10", a["10"] + b["10"], DISJ_ALPHA),
               ("00", b["00"] + b["11"], DISJ_DENOMINATOR - DISJ_ALPHA))
    trigger = next(pair for pair, count, threshold in choices if count * DISJ_DENOMINATOR >= threshold * k)

    kept, received, reject = [], 0, False
    for pair in pairs:
        action = "SEND_1" if pair == trigger else "LISTEN"
        observation = yield LocalMove(action, f"DISJ pass for {trigger}", pair)
        if observation == "0":
            kept.append(pair)
        received += observation == "1"
        if pair == "11" and observation == ("0" if trigger == "00" else "1"):
            reject = True
    if trigger != "00":
        other = BOB if role == ALICE else ALICE
        reject |= received < counts[other][trigger]

    alice_rejects = yield from transmit_number(role, ALICE, int(reject), 1, "Alice's rejection flag")
    bob_rejects = yield from transmit_number(role, BOB, int(reject), 1, "Bob's rejection flag")
    if alice_rejects or bob_rejects:
        yield LocalMove("ANSWER_0", "Intersection detected")
    elif not kept:
        yield LocalMove("ANSWER_1", "Empty instance")
    elif trigger == "00":
        reduced = "".join(pair[0 if role == ALICE else 1] for pair in kept)
        yield from run_subprotocol(role, reduced, "eq", eq_zero, "Reduced equality test")
    else:
        reduced = "".join(pair[0 if trigger == "01" else 1] for pair in kept)
        yield from disj_zero_steps(role, reduced)


def disj_zero(view: View, n: int, task: str) -> str:
    return replay_local_steps(view, disj_zero_steps(view.role, view.own_input)).token


def eq_zero_rounds(n: int) -> int:
    k = capacity_width(2 ** n, 3)
    return k + 2 * capacity_width(k + 1, 2) + 1


def disj_zero_rounds(n: int) -> int:
    """Bound all sizes up to n, including the two rounds for an odd tail."""
    if n <= 1:
        return 2 * n
    k = n // 2
    disj_size = (DISJ_DENOMINATOR - DISJ_ALPHA) * k // DISJ_DENOMINATOR
    eq_size = DISJ_ALPHA * k // DISJ_DENOMINATOR
    eq_rounds = eq_zero_rounds(eq_size) if eq_size else 0
    return 4 + 6 * k.bit_length() + k + max(disj_zero_rounds(disj_size), eq_rounds)


@dataclass(frozen=True)
class Protocol:
    name: str
    task: str | None
    channel: str | None
    choose: Callable[[View, int, str], str]
    rounds: Callable[[int], int]
    source: str
    formula: str
    local_steps: Callable[[str, str], LocalSteps] | None = None


# Public exchange strategies; task and channel are separate settings.
PROTOCOL_MODES = ("baseline", "specialized")

# Concrete implementations, also kept as aliases for saved configurations.
PROTOCOLS = {
    "send_x_then_result": Protocol("send_x_then_result", None, None, baseline,
                                    lambda n: n + 1, "", "n + 1"),
    "ip_silence": Protocol("ip_silence", "ip", "silence", ip_silence,
                            lambda n: (n + 1) // 2 + 2, DEMENTIEV_SOURCE, "ceil(n/2) + 2"),
    "ip_zero": Protocol("ip_zero", "ip", "zero", ip_zero, ip_zero_rounds, DEMENTIEV_SOURCE,
                        "4 + k + floor(3*k/4); k = ceil(n/2)", local_steps=ip_zero_steps),
    "eq_silence": Protocol("eq_silence", "eq", "silence", eq_silence,
                            lambda n: (k := capacity_width(2 ** n, 5)) + capacity_width(k + 1, 3) + 2,
                            EQ_SOURCE, "k + ceil(log_3(k+1)) + 2; k = ceil(n/log_2(5))"),
    "eq_zero": Protocol("eq_zero", "eq", "zero", eq_zero, eq_zero_rounds,
                         EQ_SOURCE, "k + 2*ceil(log_2(k+1)) + 1; k = ceil(n/log_2(3))"),
    "disj_zero": Protocol("disj_zero", "disj", "zero", disj_zero, disj_zero_rounds, DEMENTIEV_SOURCE,
                          "4 + 6*bit_length(k) + k + max(B(floor(0.4882*k)), E(floor(0.5118*k)))"
                          ", k = floor(n/2), B = DISJ/zero bound, E = EQ/zero bound for nonempty inputs"
                          ", B(0) = E(0) = 0, B(1) = 2", local_steps=disj_zero_steps),
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
        available = ", ".join(f"{p.task.upper()}/{p.channel}" for p in PROTOCOLS.values() if p.task)
        raise ValueError(
            f"Для protocol='specialized', task={task!r}, channel={channel!r} нет реализации. "
            f"Поддерживаются {available}; "
            "protocol='baseline' поддерживает все задачи и оба канала.")
    return name, task, channel


def get_protocol(name: str, task: str | None = None, channel: str | None = None) -> Protocol:
    """Select a concrete implementation; legacy names remain readable."""
    mode, task, channel = resolve_selection(name, task, channel)
    key = "send_x_then_result" if mode == "baseline" else f"{task}_{channel}"
    return PROTOCOLS[key]
