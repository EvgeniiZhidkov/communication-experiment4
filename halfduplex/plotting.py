"""Optional plotting; importing/running experiments needs no matplotlib."""
from __future__ import annotations

import json
from collections import Counter
from io import StringIO
from pathlib import Path

from .analysis import analyze_run


def plot_results(log_path: str | Path, *, save_path: str | Path | None = None):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:
        raise RuntimeError("Для графиков установите matplotlib в окружении ядра ноутбука.") from exc
    with Path(log_path).open(encoding="utf-8", errors="surrogateescape") as source, StringIO() as report:
        summary = analyze_run(source, report)
        report.seek(0)
        header = json.loads(next(report))
        errors = {"Alice": Counter(), "Bob": Counter()}
        for line in report:
            row = json.loads(line)
            if row["type"] == "episode":
                for error in row.get("local_errors", []):
                    errors[error["role"]][error["error"]] += 1
            elif row["type"] == "step" and row.get("error"):
                errors[row["state"]["role"]][row["error"]] += 1
    keys = (("Ходы", "next_action_accuracy"), ("Ответы", "output_accuracy"), ("Все решения", "decision_accuracy"))
    if header["mode"] != "step":
        keys = (("Оба ответа верны", "joint_accuracy"), ("Точное исполнение", "exact_execution"))
    fig, (accuracy_ax, errors_ax) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    title = f"{header['function'].upper()} · n={header['n']} · {header['mode']} · {header['channel']}"
    if header.get("protocol"):
        title += f"\n{header['protocol']}"
    if not summary["run_complete"]:
        title += " · неполный прогон"
    fig.suptitle(title)
    accuracy_ax.set_ylim(0, 118)
    accuracy_ax.set_yticks(range(0, 101, 25))
    accuracy_ax.set_ylabel("%")
    accuracy_ax.set_xticks(range(len(keys)), [label for label, _ in keys])
    accuracy_ax.set_title("Точность")
    for i, (_, key) in enumerate(keys):
        value = summary[key]
        if value["rate"] is None:
            accuracy_ax.text(i, 3, "нет данных", ha="center")
        else:
            percent = 100 * value["rate"]
            accuracy_ax.bar(i, percent, color="#247989", width=0.6)
            accuracy_ax.text(i, percent + 2, f"{percent:.1f}%\n{value['count']}/{value['total']}", ha="center")
    totals = errors["Alice"] + errors["Bob"]
    errors_ax.set_title("Локальные ошибки")
    if totals:
        kinds = sorted(totals)
        a = [errors["Alice"][k] for k in kinds]
        b = [errors["Bob"][k] for k in kinds]
        errors_ax.barh(range(len(kinds)), a, label="Алиса")
        errors_ax.barh(range(len(kinds)), b, left=a, label="Боб")
        errors_ax.set_yticks(range(len(kinds)), kinds)
        errors_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        errors_ax.legend()
    else:
        errors_ax.set_axis_off()
        errors_ax.text(0.5, 0.5, "Локальных ошибок нет", ha="center", va="center", transform=errors_ax.transAxes)
    if save_path is not None:
        fig.savefig(save_path, dpi=180)
    return fig
