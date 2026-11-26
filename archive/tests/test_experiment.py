import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from subprocess import TimeoutExpired
from threading import Event
from unittest.mock import patch

from halfduplex import PlayerConfig, RunConfig, analyze_file, analyze_run, input_pairs, run_experiment
from halfduplex.core import View
from halfduplex.experiment import _players, _run_episode
from halfduplex.prompts import render_prompt


class ExperimentTests(unittest.TestCase):
    def config(self, directory, **values):
        return RunConfig(alice=PlayerConfig(backend="python"), bob=PlayerConfig(backend="python"),
                         output_dir=directory, fsync=False, **values)

    def run_quietly(self, config):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_experiment(config)

    def test_all_protocols_both_modes_and_report_reanalysis(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = (("specialized", "ip", "silence", "ip_silence"),
                     ("specialized", "ip", "zero", "ip_zero"),
                     ("specialized", "disj", "zero", "disj_zero"),
                     ("specialized", "eq", "silence", "eq_silence"),
                     ("specialized", "eq", "zero", "eq_zero"),
                     ("baseline", "eq", "zero", "send_x_then_result"))
            for protocol, task, channel, implementation in cases:
                for mode in ("episode", "step"):
                    with self.subTest(protocol=protocol, task=task, channel=channel, mode=mode):
                        cfg = self.config(directory, protocol=protocol, task=task, channel=channel,
                                          n=3, experiment=mode, repeats=2,
                                          sampling="eq_balanced" if task == "eq" else "uniform")
                        result = self.run_quietly(cfg)
                        self.assertTrue(result.directory.name.startswith(f"e4_{task}-{channel}_n3_seed42_"))
                        self.assertTrue(result.summary["run_complete"])
                        key = "joint_accuracy" if mode == "episode" else "decision_accuracy"
                        self.assertEqual(result.summary[key]["rate"], 1)
                        if mode == "episode":
                            self.assertEqual(result.summary["exact_execution"]["rate"], 1)
                            self.assertEqual(result.summary[key]["total"], 30)
                        for path in (result.log_path, result.report_path):
                            with path.open() as stream:
                                summary = analyze_run(stream)
                            self.assertEqual(json.loads(json.dumps(summary)), json.loads(json.dumps(result.summary)))
                        manifest = json.loads((result.directory / "manifest.json").read_text())
                        self.assertEqual(manifest["pairs"], [list(p) for p in input_pairs(cfg)])
                        self.assertEqual(manifest["protocol"], protocol)
                        self.assertEqual(manifest["implementation"], implementation)
                        self.assertTrue((result.directory / "source" / "protocols.py").is_file())
                        # Version 2 journals previously recorded concrete names as protocol.
                        rows = result.log_path.read_text().splitlines()
                        header = json.loads(rows[0])
                        header["protocol"] = header["config"]["protocol"] = implementation
                        header.pop("implementation")
                        rows[0] = json.dumps(header)
                        old_summary = analyze_run(io.StringIO("\n".join(rows) + "\n"))
                        self.assertEqual(old_summary, result.summary)

    def test_sampling_reproducible_distinct_and_independent_of_models_and_format(self):
        cfg = RunConfig(n=5)
        original = input_pairs(cfg)
        changed = replace(cfg, input_format="positions", explanation="pseudocode", bob=PlayerConfig(model="example-model"))
        self.assertEqual(original, input_pairs(changed))
        self.assertEqual(len(set(original)), 15)
        self.assertEqual(original[0], ("00111", "00100"))  # Historical seed=42 pilot.
        balanced = RunConfig(task="eq", channel="zero", protocol="specialized", sampling="eq_balanced", n=5)
        pairs = input_pairs(balanced)
        self.assertEqual(sum(x == y for x, y in pairs), 8)
        self.assertEqual(len(set(pairs)), 15)
        self.assertEqual(pairs, input_pairs(replace(balanced, channel="silence")))
        self.assertEqual(pairs, input_pairs(replace(balanced, protocol="baseline")))
        self.assertEqual(len(input_pairs(RunConfig(n=100, case_limit=20))), 20)

    def test_legacy_journal_metrics_still_match(self):
        root = Path(__file__).resolve().parents[2]
        run = root / "runs" / "e4_ip-silence_n5_seed42_20261109T221631_854494Z"
        path = run / "experiment4_episode_20261109T221631_854494Z.jsonl"
        report = run / "experiment4_episode_20261109T221631_854494Z_analysis_20261109T223023_735588Z.jsonl"
        if not path.is_file():
            self.skipTest("Historical local result not present")
        with path.open() as stream:
            actual = analyze_run(stream)
        saved = json.loads(report.read_text().splitlines()[-1])
        saved.pop("type")
        self.assertEqual(json.loads(json.dumps(actual)), saved)

    def test_parallel_and_serial_runs_produce_the_same_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, task="ip", channel="zero", n=32, case_limit=8, repeats=2)
            serial = self.run_quietly(config)
            parallel = self.run_quietly(replace(config, jobs=4))
            self.assertEqual(parallel.summary, serial.summary)

            def read_episodes(path):
                episodes = []
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if row["type"] == "episode":
                        row.pop("time_utc")
                        episodes.append(row)
                return sorted(episodes, key=lambda row: (row["case_index"], row["repeat"]))

            self.assertEqual(len(read_episodes(parallel.log_path)), 16)
            self.assertEqual(read_episodes(parallel.log_path), read_episodes(serial.log_path))

    def test_parallel_failure_saves_completed_games_before_interrupting(self):
        saved = Event()

        def run_game(config, log, case_index, x, y, repeat, stop):
            if case_index == 0:
                recorded = _run_episode(config, log, case_index, x, y, repeat, stop)
                saved.set()
                self.assertTrue(stop.wait(timeout=5))
                return recorded
            else:
                self.assertEqual(case_index, 1)
                self.assertTrue(saved.wait(timeout=5))
                raise RuntimeError("synthetic worker failure")

        with tempfile.TemporaryDirectory() as directory, patch("halfduplex.experiment._run_episode", run_game):
            with self.assertRaisesRegex(RuntimeError, "synthetic worker failure"):
                self.run_quietly(self.config(directory, case_limit=5, jobs=2))
            run = next(Path(directory).glob("e4_*"))
            rows = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
            self.assertEqual([row["type"] for row in rows], ["config", "episode", "interrupted"])
            self.assertEqual(rows[1]["case_index"], 0)
            summary = json.loads((run / "summary.json").read_text())
            self.assertFalse(summary["run_complete"])
            self.assertEqual(summary["exact_execution"]["total"], 1)

    def test_episode_timeout_preserves_the_other_results(self):
        def timeout(view):
            raise TimeoutExpired("test-codex", 120)

        def players(config, log, stack, *, persistent, episode_context):
            result = _players(config, log, stack, persistent=persistent, episode_context=episode_context)
            if episode_context["case_index"] == 0:
                result["Bob"] = timeout
            return result

        for jobs in (1, 4):
            with self.subTest(jobs=jobs), tempfile.TemporaryDirectory() as directory, \
                    patch("halfduplex.experiment._players", players):
                result = self.run_quietly(self.config(directory, case_limit=5, jobs=jobs))
                self.assertFalse(result.summary["run_complete"])
                self.assertEqual(result.summary["timed_out_episodes"], 1)
                self.assertEqual(result.summary["joint_accuracy"], {"count": 4, "total": 4, "rate": 1.0})
                rows = [json.loads(line) for line in result.log_path.read_text().splitlines()]
                self.assertEqual([r["case_index"] for r in rows if r["type"] == "episode.timeout"], [0])
                self.assertEqual(sorted(r["case_index"] for r in rows if r["type"] == "episode"), [1, 2, 3, 4])
                with result.report_path.open() as stream:
                    self.assertEqual(result.summary, analyze_run(stream))

    def test_interrupted_and_truncated_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_quietly(self.config(directory, case_limit=2))
            rows = result.log_path.read_text().splitlines(keepends=True)
            partial = "".join(rows[:-1]) + '{"type":"episode"'
            summary = analyze_run(io.StringIO(partial))
            self.assertFalse(summary["run_complete"])
            self.assertEqual(summary["joint_accuracy"]["total"], 2)
            with self.assertRaises(json.JSONDecodeError):
                analyze_run(io.StringIO(rows[0] + '{broken}\n'))

    def test_corrupted_observations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_quietly(self.config(directory, case_limit=1))
            rows = [json.loads(line) for line in result.log_path.read_text().splitlines()]
            rows[1]["rounds"][0]["observations"] = ["1", "1"]
            with self.assertRaises(ValueError):
                analyze_run(io.StringIO("\n".join(json.dumps(r) for r in rows)))

    def test_non_object_records_and_invalid_utf8_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_quietly(self.config(directory, case_limit=1))
            header = result.log_path.read_text().splitlines()[0]
            for bad in ("[]", "null", "42", '"text"', '{"type":"completed","note":"\udcff"}'):
                with self.subTest(record=repr(bad)), self.assertRaises(ValueError):
                    analyze_run(io.StringIO(header + "\n" + bad + "\n"))

    def test_failed_analysis_removes_its_partial_report(self):
        with tempfile.TemporaryDirectory() as directory:
            journal, report = Path(directory) / "journal.jsonl", Path(directory) / "report.jsonl"
            journal.write_text('[]\n')
            with self.assertRaises(ValueError):
                analyze_file(journal, report)
            self.assertFalse(report.exists())
            report.write_text("existing report")
            with self.assertRaises(FileExistsError):
                analyze_file(journal, report)
            self.assertEqual(report.read_text(), "existing report")

    def test_input_formats_are_local(self):
        for kind, encoded in (("bits", "1010"), ("bit_list", "[1, 0, 1, 0]"), ("positions", "[1, 3]")):
            config = RunConfig(n=4, input_format=kind)
            prompt = render_prompt(config, View("Alice", "1010"))
            self.assertIn("Твой приватный вход: " + encoded, prompt)
            self.assertNotIn("Твой приватный вход: 0101", prompt)

    def test_incompatible_settings_fail_before_run(self):
        for values in ({"protocol": "ip_silence", "task": "eq"},
                       {"protocol": "eq_zero", "channel": "silence"},
                       {"protocol": "specialized", "task": "disj"},
                       {"protocol": "disj_zero", "channel": "silence"},
                       {"n": 0}, {"case_limit": 0}, {"case_limit": None},
                       {"sampling": "eq_balanced"},
                       {"task": "eq", "channel": "zero", "n": 2, "sampling": "eq_balanced", "case_limit": 15}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RunConfig(**values)


if __name__ == "__main__":
    unittest.main()
