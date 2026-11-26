"""python -m halfduplex reference/run/analyze/audit-tools/visualize"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from . import PlayerConfig, RunConfig, analyze_file, audit_tools, export_protocols, run_experiment, verify_reference
from .protocols import PROTOCOL_MODES


def main():
    parser = argparse.ArgumentParser(description="Воспроизведение полудуплексных протоколов (E4)")
    sub = parser.add_subparsers(dest="command", required=True)
    reference = sub.add_parser("reference", help="Полный перебор Python-эталона, без моделей")
    reference.add_argument("--protocol", default="specialized",
                           metavar="{" + ",".join(PROTOCOL_MODES) + "}",
                           help="Алгоритм обмена; прежние имена реализаций также поддерживаются")
    reference.add_argument("--task", choices=("ip", "eq", "disj"), help="Вычисляемая функция (по умолчанию ip)")
    reference.add_argument("--channel", choices=("silence", "zero"), help="Канал (по умолчанию silence)")
    reference.add_argument("--n", type=int, default=5)
    run = sub.add_parser("run", help="Запуск по JSON-конфигурации")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--python-players", action="store_true", help="Обе роли — Python-эталон")
    run.add_argument("--alice-model")
    run.add_argument("--bob-model")
    run.add_argument("--limit", type=int)
    run.add_argument("--jobs", type=int, help="Сколько игр одновременно запускать в режиме episode")
    run.add_argument("--self-check", action=argparse.BooleanOptionalAction,
                     help="Просить модель самостоятельно проверять вычисления перед ходом")
    run.add_argument("--explain-deviations", action=argparse.BooleanOptionalAction,
                     help="При первом отклонении спросить Codex о прошлом ходе в отдельной ветке")
    run.add_argument("--record-calculations", action=argparse.BooleanOptionalAction,
                     help="В ip_zero сохранять промежуточный расчёт перед выбором действия")
    run.add_argument("--track-bob-accumulator", action=argparse.BooleanOptionalAction,
                     help="В ip_zero записывать обновления аккумулятора Боба по ходу игры")
    run.add_argument("--allow-bob-accumulator-correction", action=argparse.BooleanOptionalAction,
                     help="Разрешить Бобу самому исправлять аккумулятор до отправки итогового бита")
    run.add_argument("--track-alice-second-pass", action=argparse.BooleanOptionalAction,
                     help="В ip_zero Алиса ведёт список оставшихся позиций второго прохода")
    analyze = sub.add_parser("analyze", help="Пересчёт нового или старого JSONL-журнала")
    analyze.add_argument("log_path", type=Path)
    audit = sub.add_parser("audit-tools", help="Один реальный запрос с проверкой вызова Python")
    audit.add_argument("--model")
    audit.add_argument("--output-dir", default="runs")
    visualize = sub.add_parser("visualize", help="Диаграммы обмена: HTML и SVG/PDF/PNG без запросов к моделям")
    visualize.add_argument("log_path", type=Path, help="journal.jsonl или report.jsonl (episode)")
    visualize.add_argument("--output-dir", type=Path)
    visualize.add_argument("--formats", nargs="+", choices=("svg", "pdf", "png"), default=("svg", "pdf", "png"))
    visualize.add_argument("--episode", type=int, help="Одна игра: порядковый номер в журнале, начиная с 1")
    visualize.add_argument("--rounds-per-page", type=int, default=20)
    visualize.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.command == "reference":
        try:
            config = RunConfig(protocol=args.protocol, task=args.task, channel=args.channel,
                               n=args.n, case_limit=min(15, 4 ** args.n))
        except ValueError as exc:
            parser.error(str(exc))
        result = verify_reference(config)
    elif args.command == "analyze":
        result = analyze_file(args.log_path)
    elif args.command == "audit-tools":
        result = audit_tools(PlayerConfig(model=args.model), output_dir=args.output_dir)
    elif args.command == "visualize":
        try:
            exported = export_protocols(args.log_path, output_dir=args.output_dir,
                                        formats=tuple(args.formats), episode=args.episode,
                                        rounds_per_page=args.rounds_per_page, dpi=args.dpi)
        except (ValueError, RuntimeError, OSError) as exc:
            parser.error(str(exc))
        result = {"html": str(exported.html_path), "episodes": exported.episodes, "figures": len(exported.files)}
    else:
        values = json.loads(args.config.read_text(encoding="utf-8"))
        for role in ("alice", "bob"):
            values[role] = PlayerConfig(**values.get(role, {}))
        config = RunConfig(**values)
        if args.python_players:
            config = replace(config, alice=PlayerConfig(backend="python"), bob=PlayerConfig(backend="python"))
        for role in ("alice", "bob"):
            model = getattr(args, role + "_model")
            if model:
                config = replace(config, **{role: replace(getattr(config, role), model=model)})
        if args.limit is not None:
            config = replace(config, case_limit=args.limit)
        option_keys = ("jobs", "self_check", "explain_deviations", "record_calculations", "track_bob_accumulator",
                       "allow_bob_accumulator_correction", "track_alice_second_pass")
        overrides = {key: getattr(args, key) for key in option_keys if getattr(args, key) is not None}
        if overrides:
            try:
                config = replace(config, **overrides)
            except ValueError as exc:
                parser.error(str(exc))
        result = run_experiment(config).summary
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
