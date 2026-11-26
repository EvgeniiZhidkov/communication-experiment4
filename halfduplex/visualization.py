"""Sequence diagrams of saved episodes: an offline HTML viewer and paper figures.

This module never calls a model. Matplotlib is imported only when drawing.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from io import StringIO
import json
from pathlib import Path
import textwrap

from .analysis import analyze_run, reference_player
from .config import RunConfig
from .core import ACTIONS, ANSWERS, ALICE, BOB, View, channel_step, function_bit, play
from .protocols import capacity_width, digits, replay_local_steps


@dataclass(frozen=True)
class ProtocolExport:
    html_path: Path
    files: tuple[Path, ...]
    episodes: int


def exchange_kind(alice: str, bob: str) -> str:
    """Classify channel use, not whether it follows the protocol."""
    if alice not in ACTIONS or bob not in ACTIONS:
        raise ValueError("Ожидались действия SEND_0, SEND_1 или LISTEN.")
    if alice == bob == "LISTEN":
        return "silence"
    if alice != "LISTEN" and bob != "LISTEN":
        return "collision"
    return "alice_to_bob" if bob == "LISTEN" else "bob_to_alice"


def _phase(config: RunConfig, index: int) -> tuple[str, int | None]:
    """The nominal phase; a deviation may have taken the agents off this path."""
    name, n = config.implementation.name, config.n
    if name == "send_x_then_result":
        return (f"Input bit {index}/{n}", index - 1) if index <= n else ("Return result", None)
    if name == "ip_silence":
        k = (n + 1) // 2
        if index <= k:
            return f"Bit pair {index}/{k}", index - 1
        return ("Alice accumulator" if index == k + 1 else "Return XOR result", None)
    base = 5 if name == "eq_silence" else 3
    k = capacity_width(2 ** n, base)
    if index <= k:
        return f"Base-{base} digit {index}/{k}", index - 1
    if name == "eq_silence":
        w = capacity_width(k + 1, 3)
        phase = ("Mismatch flag" if index == k + 1 else
                 "Alice receive count (base 3)" if index <= k + 1 + w else "Return result")
    else:
        w = capacity_width(k + 1, 2)
        phase = ("Alice send count (base 2)" if index <= k + w else
                 "Bob send count (base 2)" if index <= k + 2 * w else "Mismatch flag")
    return phase, None


def _input_units(config: RunConfig, bits: str) -> list[str]:
    name = config.implementation.name
    if name in {"ip_silence", "ip_zero", "disj_zero"}:
        if name != "disj_zero" and config.n % 2:
            bits += "0"
        return [bits[i:i + 2] for i in range(0, len(bits), 2)]
    if name.startswith("eq_"):
        base = 5 if name == "eq_silence" else 3
        return [str(d) for d in digits(int(bits, 2), base, capacity_width(2 ** config.n, base))]
    return list(bits)


def _trace(config: RunConfig, record: dict, *, reference: bool = False) -> dict:
    views = {ALICE: View(ALICE, record["x"]), BOB: View(BOB, record["y"])}
    choose = reference_player(config)
    local_steps = config.implementation.local_steps
    units = {role: _input_units(config, v.own_input) for role, v in views.items()}
    rows, final = [], record.get("raw_outputs")
    for row in record["rounds"]:
        actions = {ALICE: row["alice_reply"].strip(), BOB: row["bob_reply"].strip()}
        observations = row.get("observations")
        # Old logs may include an unexecuted stop/invalid decision in rounds.
        if observations is None:
            final = final or actions
            continue
        expected_observations = channel_step(actions[ALICE], actions[BOB], config.channel)
        if tuple(observations) != expected_observations:
            raise ValueError("Наблюдения в журнале не соответствуют действиям и каналу.")
        index = len(rows) + 1
        expected, round_units, phases = {}, {}, set()
        for role, view in views.items():
            try:
                if local_steps is None:
                    expected[role] = choose(view)
                else:
                    move = replay_local_steps(view, local_steps(role, view.own_input))
                    expected[role], round_units[role] = move.token, move.unit
                    phases.add(move.phase)
            except (ValueError, IndexError, KeyError):
                expected[role] = None
            round_units.setdefault(role, "")
        if local_steps is None:
            phase, unit_index = _phase(config, index)
            if unit_index is not None:
                round_units = {role: values[unit_index] if unit_index < len(values) else ""
                               for role, values in units.items()}
        elif len(phases) == 1:
            phase = phases.pop()
        else:
            phase = "Different local phases" if phases else "Phase unavailable"
        rows.append({
            "index": index, "actions": actions, "observations": dict(zip((ALICE, BOB), observations)),
            "kind": exchange_kind(actions[ALICE], actions[BOB]), "phase": phase,
            "units": round_units, "expected_local": expected,
        })
        views = {role: view.after(actions[role], obs)
                 for (role, view), obs in zip(views.items(), observations)}
    final = {role: str(final.get(role, "NO ANSWER")).strip() for role in (ALICE, BOB)} if final else None
    status = record.get("status")
    if status is None:
        status = "completed" if final and all(v in ANSWERS for v in final.values()) else "stopped"
    return {"rows": rows, "final": final, "status": status,
            "inputs": {ALICE: record["x"], BOB: record["y"]}, "units": units,
            "deviations": [] if reference else record.get("deviations", []),
            "local_errors": [] if reference else record.get("local_errors", [])}


def load_protocol_log(log_path: str | Path) -> dict:
    """Validate and re-score a journal/report, including historical JSONL logs."""
    log_path = Path(log_path)
    with log_path.open(encoding="utf-8", errors="surrogateescape") as source:
        first = source.readline()
        if not first:
            raise ValueError("Журнал пуст: нет настроек и завершённых эпизодов.")
        header = json.loads(first)
        if not isinstance(header, dict) or header.get("type") != "config":
            raise ValueError("Первая строка должна содержать настройки прогона.")
        if header.get("mode") == "step":
            raise ValueError("Режим step содержит отдельные локальные решения. Для диаграммы обмена нужен episode.")
        source.seek(0)
        with StringIO() as report:
            summary = analyze_run(source, report)
            records = [json.loads(line) for line in report.getvalue().splitlines()]
    selection = header.get("config", header)
    name = selection.get("protocol", "send_x_then_result_v1")
    config = RunConfig(protocol="baseline" if name == "send_x_then_result_v1" else name,
                       task=selection.get("task", header.get("function")),
                       channel=selection.get("channel", header.get("channel")),
                       n=header["n"], case_limit=1)
    episodes = []
    choose = reference_player(config)
    for record in records:
        if record["type"] != "episode":
            continue
        x, y = record["x"], record["y"]
        expected = play(config.n, x, y, choose, choose, channel=config.channel,
                        max_rounds=config.implementation.rounds(config.n))
        ideal = {**expected.record(), "x": x, "y": y}
        episodes.append({
            "number": len(episodes) + 1, "x": x, "y": y, "repeat": record.get("repeat", 1),
            "truth": function_bit(config.task, x, y), "joint_correct": record["joint_correct"],
            "exact": record["exact"], "actual": _trace(config, record),
            "reference": _trace(config, ideal, reference=True), "record": record,
        })
    if not episodes:
        raise ValueError("В журнале нет сохранённых эпизодов для визуализации.")
    return {"source": log_path.name, "task": config.task, "n": config.n, "channel": config.channel,
            "implementation": config.implementation.name, "summary": summary, "episodes": episodes}


def _short(value: str, limit: int = 42) -> str:
    return value if len(value) <= limit else value[:limit - 1] + "…"


def draw_protocol(data: dict, episode: dict, *, reference: bool = False, page: int = 0,
                  rounds_per_page: int = 20):
    """Return a Matplotlib Figure. Labels are English for use in papers.

    A page includes at most rounds_per_page delivered rounds. Final decisions are
    shown only on the last page, without an arrow or another communication round.
    """
    try:
        from matplotlib.figure import Figure
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except ImportError as exc:
        raise RuntimeError("Для диаграмм установите зависимости: python3 -m pip install -r requirements-plots.txt") from exc
    if not isinstance(rounds_per_page, int) or isinstance(rounds_per_page, bool) or rounds_per_page < 1:
        raise ValueError("rounds_per_page должно быть положительным целым числом.")
    trace = episode["reference" if reference else "actual"]
    pages = max(1, (len(trace["rows"]) + rounds_per_page - 1) // rounds_per_page)
    if not 0 <= page < pages:
        raise ValueError(f"Страница должна быть от 0 до {pages - 1}.")
    rows = trace["rows"][page * rounds_per_page:(page + 1) * rounds_per_page]
    is_last = page == pages - 1
    slots = len(rows) + 1  # Reserve the last slot for STOP or the continuation note.
    height = 1.8 + slots * .86
    fig = Figure(figsize=(10.6, height), facecolor="white", layout=None)
    ax = fig.add_axes((.02, .025, .96, .95))
    top = slots + 1.8
    ax.set(xlim=(0, 10), ylim=(-.4, top))
    ax.axis("off")
    ink, muted, teal, purple, red, amber = "#172b3a", "#647580", "#147d83", "#775aa0", "#b63838", "#ae6b16"
    title = f"{data['task'].upper()} · n = {data['n']} · {data['implementation']} · {data['channel']}"
    ax.text(.2, top - .08, title, fontsize=15, fontweight="bold", color=ink, va="top")
    subtitle = f"Episode {episode['number']} · {'Reference' if reference else 'Observed'} · repeat {episode['repeat']}"
    if pages > 1:
        subtitle += f" · page {page + 1}/{pages}"
    ax.text(.2, top - .48, subtitle, fontsize=12, color=muted, va="top")
    ax.text(9.8, top - .48, f"f(x, y) = {episode['truth']}", fontsize=13, color=ink, ha="right", va="top")
    columns = {ALICE: 2.05, BOB: 7.95}
    first_y = slots - .34
    for role, x in columns.items():
        ax.text(x, top - 1.0, role, ha="center", fontsize=16, fontweight="bold", color=ink)
        bits = _short(trace["inputs"][role], 38)
        ax.text(x, top - 1.35, f"{'x' if role == ALICE else 'y'} = {bits}", ha="center",
                fontsize=11, fontfamily="monospace", color=ink)
        if data["implementation"] != "send_x_then_result":
            encoding = {"eq_silence": "base 5", "eq_zero": "base 3"}.get(data["implementation"], "pairs")
            encoded = _short(" ".join(trace["units"][role]), 43)
            ax.text(x, top - 1.67, f"{encoding}: {encoded}", ha="center", fontsize=10.5, color=muted)
        ax.plot([x, x], [.62, first_y + .2], color="#ccd5da", linewidth=1, zorder=0)

    def border(role, index, token, *, terminal=False):
        local = any(e["role"] == role and e["round"] == index for e in trace["local_errors"])
        off_trace = any(e["role"] == role and e["round"] == index for e in trace["deviations"])
        wrong_answer = terminal and (token not in ANSWERS or token != f"ANSWER_{episode['truth']}")
        return red if local or wrong_answer else amber if off_trace else "#bdccd2"

    def node(role, y, token, observation, index, unit="", terminal=False):
        x = columns[role]
        edge = border(role, index, token, terminal=terminal)
        fill = "#edf6f3" if terminal and token == f"ANSWER_{episode['truth']}" else "#f7fafb"
        ax.add_patch(FancyBboxPatch((x - 1.22, y - .35), 2.44, .70,
                                   boxstyle="round,pad=0.035,rounding_size=0.06",
                                   facecolor=fill, edgecolor=edge, linewidth=1.5, zorder=3))
        ax.text(x, y + .08, _short(token, 22), ha="center", va="center", fontsize=13,
                fontfamily="monospace", color=ink, zorder=4, parse_math=False)
        if terminal:
            detail = "local decision"
        else:
            detail = f"obs: {'—' if observation is None else observation}"
            if unit:
                detail += f" · unit: {unit}"
        ax.text(x, y - .17, detail, ha="center", va="center", fontsize=11, color=muted, zorder=4)

    def arrow(x1, x2, y, label, color=teal):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=11,
                                    linewidth=1.7, color=color, zorder=2))
        ax.text((x1 + x2) / 2, y + .12, label, ha="center", fontsize=12, color=color)

    for offset, row in enumerate(rows):
        y, index = first_y - offset, row["index"]
        ax.text(.20, y + .04, str(index), fontsize=11, color=muted, ha="center")
        for role in columns:
            node(role, y, row["actions"][role], row["observations"][role], index, row["units"][role])
        kind = row["kind"]
        if kind == "alice_to_bob":
            arrow(3.35, 6.65, y + .04, row["actions"][ALICE][-1])
        elif kind == "bob_to_alice":
            arrow(6.65, 3.35, y + .04, row["actions"][BOB][-1])
        elif kind == "collision":
            arrow(3.35, 4.84, y + .04, row["actions"][ALICE][-1], purple)
            arrow(6.65, 5.16, y + .04, row["actions"][BOB][-1], purple)
            ax.plot([4.93, 5.07], [y - .04, y + .12], color=purple, linewidth=1.8)
            ax.plot([4.93, 5.07], [y + .12, y - .04], color=purple, linewidth=1.8)
        else:
            ax.plot([3.38, 6.62], [y + .04, y + .04], color="#aab7bf", linestyle=(0, (3, 4)), linewidth=1)
            label = "SILENCE" if data["channel"] == "silence" else "0 / 0"
            ax.text(5, y + .10, label, ha="center", fontsize=11, color=muted,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
        exchange = "collision · bits lost" if kind == "collision" else "both listen" if kind == "silence" else "delivered"
        ax.text(5, y - .24, exchange, ha="center", fontsize=10.5, color=purple if kind == "collision" else muted)
        phase_lines = textwrap.fill(row["phase"], 40)
        ax.text(5, y - .48, phase_lines, ha="center", va="top", fontsize=10.5, color=muted)
    if is_last:
        y = first_y - len(rows)
        index = len(trace["rows"]) + 1
        for role in columns:
            node(role, y, (trace["final"] or {}).get(role, "NO ANSWER"), None, index, terminal=True)
        ax.text(5, y + .06, "STOP", ha="center", fontsize=12, fontweight="bold", color=ink)
        ax.text(5, y - .20, _short(trace["status"], 30), ha="center", fontsize=10.5, color=muted)
    else:
        ax.text(5, .42, "Continued on the next page", ha="center", fontsize=10.5, color=muted)
    ax.text(.22, -.04, "→ delivered bit     × collision     - - both listen     obs: — = no observation",
            fontsize=10.5, color=muted)
    ax.text(.22, -.29, "Red border: local error / wrong answer. Amber: deviation from reference. A collision can be valid.",
            fontsize=10, color=muted)
    return fig


def export_protocols(log_path: str | Path, *, output_dir: str | Path | None = None,
                     formats: tuple[str, ...] = ("svg", "pdf", "png"), episode: int | None = None,
                     rounds_per_page: int = 20, dpi: int = 300,
                     results_url: str | None = None) -> ProtocolExport:
    """Write a self-contained viewer plus actual/reference figures for each episode.

    episode is a one-based ordinal in the journal, not case_index. Long traces are
    paginated; all recorded rounds and the final local decisions remain visible.
    results_url optionally adds navigation back to the hosted results.
    """
    formats = tuple(dict.fromkeys(formats))
    if not formats or set(formats) - {"svg", "pdf", "png"}:
        raise ValueError("formats: один или несколько форматов svg, pdf, png.")
    if not isinstance(rounds_per_page, int) or isinstance(rounds_per_page, bool) or rounds_per_page < 1:
        raise ValueError("rounds_per_page должно быть положительным целым числом.")
    if not isinstance(dpi, int) or isinstance(dpi, bool) or not 72 <= dpi <= 1200:
        raise ValueError("dpi должно быть целым числом от 72 до 1200.")
    data = load_protocol_log(log_path)
    if episode is not None:
        if not isinstance(episode, int) or isinstance(episode, bool) or not 1 <= episode <= len(data["episodes"]):
            raise ValueError(f"episode должно быть от 1 до {len(data['episodes'])}.")
        data["episodes"] = [data["episodes"][episode - 1]]
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("Для диаграмм установите зависимости: python3 -m pip install -r requirements-plots.txt") from exc
    directory = Path(output_dir) if output_dir is not None else Path(log_path).parent / "protocols"
    directory.mkdir(parents=True, exist_ok=True)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    files = []
    settings = {"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42, "text.parse_math": False,
                "ps.fonttype": 42, "font.size": 9}
    with matplotlib.rc_context(settings):
        for item in data["episodes"]:
            for kind in ("actual", "reference"):
                trace = item[kind]
                trace["pages"] = []
                pages = max(1, (len(trace["rows"]) + rounds_per_page - 1) // rounds_per_page)
                for page in range(pages):
                    fig = draw_protocol(data, item, reference=kind == "reference", page=page,
                                        rounds_per_page=rounds_per_page)
                    stem = f"episode_{item['number']:04d}_{kind}"
                    if pages > 1:
                        stem += f"_p{page + 1:02d}"
                    with StringIO() as stream:
                        fig.savefig(stream, format="svg", metadata={"Date": None, "Creator": "halfduplex"})
                        svg = stream.getvalue()
                    svg = svg[svg.index("<svg"):]
                    exports = {}
                    for fmt in formats:
                        path = figures / f"{stem}.{fmt}"
                        if fmt == "svg":
                            path.write_text(svg, encoding="utf-8")
                        else:
                            metadata = {"CreationDate": None, "Creator": "halfduplex"} if fmt == "pdf" else None
                            fig.savefig(path, format=fmt, dpi=dpi, metadata=metadata, facecolor="white")
                        exports[fmt] = path.relative_to(directory).as_posix()
                        files.append(path)
                    trace["pages"].append({"svg": svg, "exports": exports})
                    fig.clear()
    # JSON is inside a script element. Escape HTML delimiters even in raw replies.
    data["rounds_per_page"] = rounds_per_page
    payload = json.dumps(data, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(__file__).with_name("templates") / "protocol_viewer.html"
    results_link = ""
    if results_url:
        results_link = (f'<nav class="results-nav" aria-label="Навигация">'
                        f'<a class="results-link" href="{escape(results_url)}">← К прогонам</a></nav>')
    html = template.read_text(encoding="utf-8").replace("__RESULTS_LINK__", results_link)
    html_path = directory / "index.html"
    html_path.write_text(html.replace("__PROTOCOL_DATA__", payload), encoding="utf-8")
    return ProtocolExport(html_path, tuple(files), len(data["episodes"]))
