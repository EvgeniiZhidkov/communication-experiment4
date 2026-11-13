"""Validate actual OS isolation, optionally using one live model request."""
import argparse
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-audit", action="store_true", help="Also make one real Codex request")
    parser.add_argument("--model")
    parser.add_argument("--smoke", action="store_true", help="One IP n=2 episode with two real Codex players")
    args = parser.parse_args()
    os.environ["HALFDUPLEX_SANDBOX_TESTS"] = "1"
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_python_tool")
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        return 1
    if args.live_audit:
        from halfduplex import PlayerConfig, audit_tools
        print(json.dumps(audit_tools(PlayerConfig(model=args.model)), ensure_ascii=False, indent=2))
    if args.smoke:
        from halfduplex import PlayerConfig, RunConfig, run_experiment
        config = RunConfig(task="ip", channel="silence", protocol="specialized",
                           n=2, case_limit=None, case=("01", "11"),
                           alice=PlayerConfig(model=args.model), bob=PlayerConfig(model=args.model))
        result = run_experiment(config)
        print("SMOKE_RESULT", result.directory)
        if result.summary["exact_execution"]["rate"] != 1:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
