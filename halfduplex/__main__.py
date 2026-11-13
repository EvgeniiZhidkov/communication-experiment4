"""python -m halfduplex reference/run/analyze/audit-tools"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from . import PlayerConfig, RunConfig, analyze_file, audit_tools, run_experiment, verify_reference
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
    analyze = sub.add_parser("analyze", help="Пересчёт нового или старого JSONL-журнала")
    analyze.add_argument("log_path", type=Path)
    audit = sub.add_parser("audit-tools", help="Один реальный запрос с проверкой вызова Python")
    audit.add_argument("--model")
    audit.add_argument("--output-dir", default="runs")
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
        result = run_experiment(config).summary
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
