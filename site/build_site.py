"""Build the E4 dashboard and protocol viewers for GitHub Pages."""
import argparse
from html import escape
from pathlib import Path
import shutil
import sys


SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
sys.path.insert(0, str(ROOT))

from halfduplex import export_protocols, load_protocol_log


RUNS = (
    ("ip-n4", "e4_20261117T152217_079029Z_7144d9f1"),
    ("eq-n4", "e4_20261117T163132_834171Z_ab15fbd6"),
)


def render_run(output, slug, run_id):
    result = ROOT / "results" / run_id
    source = result / "journal.jsonl"
    view = output / "protocols" / slug
    run = load_protocol_log(source)
    export_protocols(source, output_dir=view, results_url=f"../../index.html#{slug}")
    shutil.copyfile(source, view / "journal.jsonl")
    shutil.copyfile(result / "plots.png", view / "results.png")

    title = f"{run['task'].upper()} · n={run['n']}"
    summary = run["summary"]
    joint = summary["joint_accuracy"]
    exact = summary["exact_execution"]
    rounds = f"{summary['mean_rounds']:.2f}".rstrip("0").rstrip(".").replace(".", ",")
    url = f"protocols/{slug}/"
    return f"""
      <article class="run" aria-labelledby="{slug}">
        <header class="run-header">
          <h3 id="{slug}">{escape(title)}</h3>
          <p>Codex–Codex · <code>{escape(run['implementation'])}</code></p>
        </header>
        <dl class="metrics">
          <div><dt>Оба ответа верны</dt><dd>{joint['count']}/{joint['total']}</dd></div>
          <div><dt>Точное исполнение</dt><dd>{exact['count']}/{exact['total']}</dd></div>
          <div><dt>Раундов в среднем</dt><dd>{rounds}</dd></div>
        </dl>
        <a class="plot" href="{url}results.png">
          <img src="{url}results.png" alt="Графики точности и локальных ошибок для {escape(title)}" width="1980" height="810" loading="lazy">
        </a>
        <div class="run-links">
          <a class="button" href="{url}">Разобрать раунды</a>
          <a href="{url}journal.jsonl" download>Скачать JSONL</a>
        </div>
      </article>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SITE / "dist")
    output = parser.parse_args().output.resolve()
    if SITE.is_relative_to(output):
        parser.error("--output must not overwrite the site source")
    output.mkdir(parents=True, exist_ok=True)

    run_cards = "\n".join(render_run(output, *run) for run in RUNS)
    template = (SITE / "index.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(template.replace("<!-- RUN_CARDS -->", run_cards), encoding="utf-8")
    shutil.copyfile(SITE / "style.css", output / "style.css")
    shutil.copyfile(ROOT / "communication_complexity_final (1).html", output / "project.html")
    (output / ".nojekyll").touch()
    print(f"GitHub Pages dashboard built at {output}")


if __name__ == "__main__":
    main()
