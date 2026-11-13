import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from halfduplex import PlayerConfig, RunConfig, analyze_file, analyze_run, input_pairs, run_experiment
from halfduplex.core import View
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
        root = Path(__file__).resolve().parents[1]
        path = root / "results" / "experiment4_episode_20261109T221631_854494Z.jsonl"
        report = root / "results" / "experiment4_episode_20261109T221631_854494Z_analysis_20261109T223023_735588Z.jsonl"
        if not path.is_file():
            self.skipTest("Historical local result not present")
        with path.open() as stream:
            actual = analyze_run(stream)
        saved = json.loads(report.read_text().splitlines()[-1])
        saved.pop("type")
        self.assertEqual(json.loads(json.dumps(actual)), saved)

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
                       {"protocol": "specialized", "task": "ip", "channel": "zero"},
                       {"n": 0}, {"case_limit": 0}, {"case_limit": None},
                       {"sampling": "eq_balanced"},
                       {"task": "eq", "channel": "zero", "n": 2, "sampling": "eq_balanced", "case_limit": 15}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RunConfig(**values)


if __name__ == "__main__":
    unittest.main()
