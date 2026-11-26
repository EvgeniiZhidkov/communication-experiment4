"""Build the E4 dashboard and protocol viewers from the experiment archive."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import quote


SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
RUNS = ROOT / "runs"
PUBLIC_FILES = ("manifest.json", "report.jsonl", "calculations.jsonl", "deviation_explanations.jsonl")
OLD_LINKS = {
    "e4_ip-silence_n4_seed42_20261117T152217_079029Z_7144d9f1": "ip-n4",
    "e4_eq-silence_n4_seed42_20261117T163132_834171Z_ab15fbd6": "eq-n4",
}
sys.path.insert(0, str(ROOT))

from halfduplex import analyze_run, export_protocols


@dataclass
class Run:
    journal: Path
    folder: Path
    header: dict
    summary: dict
    started: datetime
    plot: Path | None

    @property
    def slug(self):
        return "--".join(self.folder.parts)

    @property
    def title(self):
        return f"{self.header['function'].upper()} · n={self.header['n']}"

    @property
    def has_viewer(self):
        return self.summary.get("joint_accuracy", {}).get("total", 0) > 0


def run_start(header, journal):
    if header.get("time_utc"):
        return datetime.fromisoformat(header["time_utc"].replace("Z", "+00:00"))
    stamp = re.search(r"\d{8}T\d{6}", journal.as_posix())
    if stamp:
        return datetime.strptime(stamp[0] + "Z", "%Y%m%dT%H%M%S%z")
    raise ValueError(f"No timestamp in {journal}")


def collect_runs(root):
    journals = []
    for directory, children, files in os.walk(root):
        children[:] = [child for child in children if not child.startswith(".")
                       and child not in {"calls", "sessions", "protocols", "source"}]
        folder = Path(directory)
        if "journal.jsonl" in files:
            journals.append(folder / "journal.jsonl")
        else:
            journals.extend(folder / name for name in files if name.startswith("experiment4_episode_")
                            and name.endswith(".jsonl") and "_analysis_" not in name)

    runs, seen = [], set()
    for journal in sorted(journals, key=lambda path: (len(path.relative_to(root).parts), path)):
        raw = journal.read_bytes()
        if not raw.strip():
            continue
        digest = sha256(raw).digest()
        if digest in seen:
            continue
        seen.add(digest)
        with journal.open(encoding="utf-8", errors="surrogateescape") as source:
            header = json.loads(source.readline())
            summary_file = journal.with_name("summary.json")
            if summary_file.is_file():
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
            else:
                source.seek(0)
                summary = analyze_run(source)
        folder = journal.parent.relative_to(root)
        plot = journal.with_name("plots.png")
        if not plot.is_file() and journal.name != "journal.jsonl":
            plot = journal.with_name(f"{journal.stem}_plots.png")
        runs.append(Run(journal, folder, header, summary, run_start(header, journal),
                        plot if plot.is_file() else None))
    return sorted(runs, key=lambda run: (run.started, run.slug), reverse=True)


def write_viewer(journal, target, back):
    export_protocols(journal, output_dir=target, formats=("svg",), results_url=back)


def publish_run(output, run):
    relative = Path("runs") / run.folder
    target = output / relative
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run.journal, target / "journal.jsonl")
    (target / "summary.json").write_text(json.dumps(run.summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    available = []
    for name in PUBLIC_FILES:
        source = run.journal.with_name(name)
        if source.is_file() and source.stat().st_size:
            shutil.copyfile(source, target / name)
            available.append(name)
    if run.plot:
        shutil.copyfile(run.plot, target / "plots.png")
    return quote(relative.as_posix(), safe="/") + "/", available


def describe_run(run):
    settings = run.header.get("config", run.header)
    details = [run.header["channel"], run.started.strftime("%d.%m.%Y · %H:%M UTC")]
    if "seed" in settings:
        details.insert(1, f"seed {settings['seed']}")
    players = settings.get("alice"), settings.get("bob")
    models = [player.get("model") or player.get("backend") for player in players if isinstance(player, dict)]
    if models:
        details.append(models[0] if len(set(models)) == 1 else " / ".join(models))
    if run.header.get("protocol") == "baseline":
        details.append("baseline")
    if run.header.get("mode") == "step":
        details.append("отдельные ходы")
    return " · ".join(details)


def ratio(metric):
    return f"{metric['count']}/{metric['total']}" if metric["total"] else "—"


def render_metrics(summary):
    if "joint_accuracy" in summary:
        rounds = summary.get("mean_rounds")
        mean = f"{rounds:.2f}".rstrip("0").rstrip(".").replace(".", ",") if rounds is not None else "—"
        metrics = (("Оба ответа верны", ratio(summary["joint_accuracy"])),
                   ("Точное исполнение", ratio(summary["exact_execution"])),
                   ("Раундов в среднем", mean))
    else:
        metrics = tuple((label, ratio(summary[key])) for label, key in (
            ("Правильные ходы", "next_action_accuracy"), ("Правильные ответы", "output_accuracy"),
            ("Все решения", "decision_accuracy")))
    return "".join(f"<div><dt>{label}</dt><dd>{value}</dd></div>" for label, value in metrics)


def render_run(run, url, available, *, active):
    alias = OLD_LINKS.get(run.slug)
    alias_attribute = f' data-alias="{alias}"' if alias else ""
    visibility = "" if active else " hidden"
    status = []
    if not run.summary.get("run_complete", True):
        status.append("Прерван")
    if count := run.summary.get("timed_out_episodes"):
        status.append(f"Таймауты — {count}")
    if "joint_accuracy" in run.summary and not run.has_viewer:
        status.append("Нет сохранённых игр")
    status_line = f'<p class="run-status">{" · ".join(status)}</p>' if status else ""
    analyze = f'<a class="button" href="{url}rounds/index.html">Разобрать раунды</a>' if run.has_viewer else ""
    downloads = [("journal.jsonl", "Журнал JSONL"), ("summary.json", "Сводка JSON")]
    if "calculations.jsonl" in available:
        downloads.append(("calculations.jsonl", "Расчёты"))
    if "deviation_explanations.jsonl" in available:
        downloads.append(("deviation_explanations.jsonl", "Разбор ошибок"))
    links = "".join(f'<a href="{url}{name}" download>{label}</a>' for name, label in downloads)
    graph = f'''<figure class="run-plot">
      <a href="{url}plots.png"><img src="{url}plots.png" alt="Графики точности и ошибок для {escape(run.title)}" loading="lazy"></a>
      <figcaption><a href="{url}plots.png">Открыть график</a></figcaption>
    </figure>''' if run.plot else ""
    return f'''<article class="run" id="{escape(run.slug)}" aria-labelledby="{escape(run.slug)}-title"{alias_attribute}{visibility}>
      <div class="run-info">
        <header><h3 id="{escape(run.slug)}-title">{escape(run.title)}</h3>
        <p class="run-meta">{escape(describe_run(run))}</p>{status_line}</header>
        <dl class="metrics">{render_metrics(run.summary)}</dl>
        <div class="run-links">{analyze}{links}</div>
      </div>
      {graph}
    </article>'''


def old_redirect(output, slug, url):
    target = output / "protocols" / slug
    target.mkdir(parents=True, exist_ok=True)
    address = f"../../{url}rounds/index.html"
    (target / "index.html").write_text(
        f'<!doctype html><html lang="ru"><meta charset="utf-8"><title>Протокол обмена</title>'
        f'<meta http-equiv="refresh" content="0; url={address}">'
        f'<a href="{address}">Открыть разбор раундов</a></html>', encoding="utf-8")


def write_project(output):
    source = (ROOT / "communication_complexity_final (1).html").read_text(encoding="utf-8")
    style = """<style>
      html { scroll-padding-top: 3.5rem; }
      .project-nav { position: sticky; top: 0; z-index: 5; margin: 0; padding: 0 1.5rem;
        background: #142d3b; min-height: 3rem; align-items: center; }
      .project-nav a { color: #d5f1eb; font-weight: 600; }
      @media print { .project-nav { display: none; } }
    </style>"""
    nav = '<nav class="project-nav" aria-label="Навигация сайта"><a href="index.html#results">← К прогонам</a></nav>'
    html = source.replace("</head>", f"{style}\n</head>", 1).replace("<body>", f"<body>\n{nav}", 1)
    (output / "project.html").write_text(html, encoding="utf-8")


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    cards, options, viewers = [], [], []
    for run in collect_runs(RUNS):
        url, available = publish_run(output, run)
        cards.append(render_run(run, url, available, active=not cards))
        label = f"{run.title} · {describe_run(run)}"
        options.append(f'<option value="{escape(run.slug)}">{escape(label)}</option>')
        if run.has_viewer:
            target = output / "runs" / run.folder / "rounds"
            back = Path(os.path.relpath(output / "index.html", target)).as_posix() + f"#{quote(run.slug)}"
            viewers.append((run.journal, target, back))
            if alias := OLD_LINKS.get(run.slug):
                old_redirect(output, alias, url)

    template = (SITE / "index.html").read_text(encoding="utf-8")
    html = template.replace("<!-- RUN_OPTIONS -->", "\n".join(options) or "<option>Нет прогонов</option>")
    html = html.replace("<!-- RUN_CARDS -->", "\n".join(cards) or "<p>В архиве пока нет результатов.</p>")
    (output / "index.html").write_text(html, encoding="utf-8")
    for name in ("style.css", "app.js"):
        shutil.copyfile(SITE / name, output / name)
    write_project(output)
    (output / ".nojekyll").touch()
    print(f"Обновляю разбор раундов для {len(viewers)} прогонов", flush=True)
    if viewers:
        with ProcessPoolExecutor(max_workers=min(4, len(viewers))) as pool:
            futures = [pool.submit(write_viewer, *viewer) for viewer in viewers]
            for result in futures:
                result.result()
    print(f"Готово: {len(cards)} прогонов в {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SITE / "dist")
    output = parser.parse_args().output.resolve()
    if SITE.is_relative_to(output) or RUNS.is_relative_to(output) or output.is_relative_to(RUNS):
        parser.error("--output must be outside the site source and run archive")
    build(output)


if __name__ == "__main__":
    main()
