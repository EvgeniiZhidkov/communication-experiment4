import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from halfduplex import RunConfig, export_protocols, load_protocol_log
from halfduplex.__main__ import main
from halfduplex.analysis import reference_player
from halfduplex.core import play
from halfduplex.visualization import exchange_kind


HAS_PLOTS = importlib.util.find_spec("matplotlib") is not None


class VisualizationTests(unittest.TestCase):
    def write_log(self, directory, config, pairs, *, alice=None, bob=None, limit=None, complete=True):
        reference = reference_player(config)
        rows = [{"type": "config", "schema_version": 2, "mode": config.experiment,
                 "protocol": config.protocol, "implementation": config.implementation.name,
                 "function": config.task, "channel": config.channel, "n": config.n, "config": config.record()}]
        for x, y in pairs:
            episode = play(config.n, x, y, alice or reference, bob or reference, channel=config.channel,
                           max_rounds=config.max_rounds if limit is None else limit)
            rows.append({"type": "episode", "x": x, "y": y, "repeat": 1, **episode.record()})
        if complete:
            rows.append({"type": "completed"})
        path = Path(directory) / "journal.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_valid_collisions_do_not_count_as_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_log(directory, RunConfig(n=4), [("0000", "0000")])
            data = load_protocol_log(path)
            episode = data["episodes"][0]
            self.assertTrue(episode["exact"])
            self.assertEqual([r["kind"] for r in episode["actual"]["rows"]],
                             ["collision", "collision", "alice_to_bob", "bob_to_alice"])
            self.assertEqual(episode["actual"]["rows"][0]["observations"], {"Alice": None, "Bob": None})
            self.assertFalse(episode["actual"]["local_errors"])

    def test_distinguishes_silence_default_zero_and_delivered_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            for channel, symbol in (("silence", "SILENCE"), ("zero", "0")):
                with self.subTest(channel=channel):
                    cfg = RunConfig(task="eq", channel=channel, n=4)
                    path = self.write_log(directory, cfg, [("0000", "0000")])
                    episode = load_protocol_log(path)["episodes"][0]
                    row = next(r for r in episode["actual"]["rows"] if r["kind"] == "silence")
                    self.assertEqual(row["observations"], {"Alice": symbol, "Bob": symbol})
            cfg = RunConfig(protocol="baseline", n=1, case_limit=1)
            path = self.write_log(directory, cfg, [("0", "1")])
            first = load_protocol_log(path)["episodes"][0]["actual"]["rows"][0]
            self.assertEqual(first["kind"], "alice_to_bob")
            self.assertEqual(first["observations"], {"Alice": None, "Bob": "0"})

    def test_eq_early_stop_and_full_reference_use_same_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_log(directory, RunConfig(task="eq", n=4),
                                  [("0000", "0001"), ("0000", "0000")])
            episodes = load_protocol_log(path)["episodes"]
            self.assertEqual([len(e["actual"]["rows"]) for e in episodes], [3, 5])
            self.assertEqual([e["truth"] for e in episodes], [0, 1])
            for episode in episodes:
                self.assertEqual(episode["actual"]["inputs"], episode["reference"]["inputs"])
                self.assertEqual(episode["actual"]["final"], episode["reference"]["final"])

    def test_zero_protocols_label_variable_passes_and_recursion(self):
        cases = (
            ("ip", "0001", "1111", ("IP pass 1", "IP pass 2", "Return XOR result")),
            ("disj", "01" * 3 + "00" * 2, "0" * 10, ("DISJ pass for 01", "DISJ pass for 00")),
            ("disj", "0001", "0010", ("DISJ pass for 00", "Reduced equality test")),
        )
        with tempfile.TemporaryDirectory() as directory:
            for task, x, y, expected_phases in cases:
                with self.subTest(task=task, x=x, y=y):
                    config = RunConfig(task=task, channel="zero", n=len(x))
                    log = self.write_log(directory, config, [(x, y)])
                    episode = load_protocol_log(log)["episodes"][0]
                    self.assertTrue(episode["joint_correct"])
                    self.assertTrue(episode["exact"])
                    phases = {row["phase"] for row in episode["reference"]["rows"]}
                    self.assertTrue(set(expected_phases) <= phases, phases)

    def test_wrong_answer_and_unexecuted_terminal_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = RunConfig(n=2, case_limit=1)
            reference = reference_player(cfg)
            def wrong_answer(view):
                token = reference(view)
                return f"ANSWER_{1 - int(token[-1])}" if token.startswith("ANSWER_") else token
            path = self.write_log(directory, cfg, [("01", "11")], alice=wrong_answer)
            episode = load_protocol_log(path)["episodes"][0]
            self.assertFalse(episode["joint_correct"])
            self.assertEqual(episode["actual"]["final"]["Alice"], "ANSWER_0")
            self.assertEqual(episode["reference"]["final"]["Alice"], "ANSWER_1")
            for actions, limit, status in ((("ANSWER_0", "LISTEN"), None, "asymmetric_stop"),
                                           (("LISTEN", "LISTEN"), 0, "round_limit"),
                                           (("bad", "LISTEN"), None, "invalid_token")):
                path = self.write_log(directory, cfg, [("01", "11")],
                                      alice=lambda v: actions[0], bob=lambda v: actions[1], limit=limit)
                trace = load_protocol_log(path)["episodes"][0]["actual"]
                self.assertEqual(trace["status"], status)
                self.assertEqual(trace["rows"], [])
                self.assertEqual(tuple(trace["final"].values()), actions)
            with self.assertRaises(ValueError):
                exchange_kind("ANSWER_0", "LISTEN")

    def test_old_journals_and_reports_are_supported(self):
        root = Path(__file__).resolve().parents[2]
        run = root / "runs" / "e4_ip-silence_n5_seed42_20261109T221631_854494Z"
        journal = run / "experiment4_episode_20261109T221631_854494Z.jsonl"
        report = run / "experiment4_episode_20261109T221631_854494Z_analysis_20261109T223023_735588Z.jsonl"
        if not journal.exists():
            self.skipTest("Historical result not present")
        original, rescored = load_protocol_log(journal), load_protocol_log(report)
        self.assertEqual(original["implementation"], "send_x_then_result")
        self.assertEqual(len(original["episodes"]), 15)
        self.assertEqual(original["episodes"][0]["actual"], rescored["episodes"][0]["actual"])

    def test_corrupt_observations_are_rejected_and_partial_run_is_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_log(directory, RunConfig(n=4), [("0000", "0000")], complete=False)
            with path.open("a") as stream:
                stream.write('{"type":"episode"')
            self.assertFalse(load_protocol_log(path)["summary"]["run_complete"])
            rows = path.read_text().splitlines()[:-1]
            episode = json.loads(rows[1])
            episode["rounds"][0]["observations"] = ["0", "0"]
            rows[1] = json.dumps(episode)
            path.write_text("\n".join(rows) + "\n")
            with self.assertRaisesRegex(ValueError, "Наблюдения"):
                load_protocol_log(path)

    def test_step_empty_and_out_of_range_inputs_fail_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "пуст"):
                load_protocol_log(path)
            path.write_text(json.dumps({"type": "config", "mode": "step"}) + "\n")
            with self.assertRaisesRegex(ValueError, "step"):
                load_protocol_log(path)
            path = self.write_log(directory, RunConfig(n=4), [("0000", "0000")])
            for episode in (0, -1, 2, True):
                with self.subTest(episode=episode), self.assertRaises(ValueError):
                    export_protocols(path, episode=episode)
            for value in (0, -1, True):
                with self.subTest(rounds_per_page=value), self.assertRaises(ValueError):
                    export_protocols(path, rounds_per_page=value)
            with self.assertRaises(ValueError):
                export_protocols(path, formats=("html",))

    @unittest.skipUnless(HAS_PLOTS, "Install requirements-plots.txt to test rendering")
    def test_export_paginates_and_writes_real_vector_and_raster_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_log(directory, RunConfig(protocol="baseline", n=4), [("0000", "0000")])
            output = export_protocols(path, rounds_per_page=2, dpi=72)
            self.assertEqual(output.episodes, 1)
            self.assertEqual(len(output.files), 18)  # 3 pages x actual/reference x 3 formats.
            for p in output.files:
                if p.suffix == ".svg":
                    root = ET.fromstring(p.read_text())
                    self.assertTrue(root.tag.endswith("svg"))
                    text = " ".join(root.itertext())
                    self.assertEqual("STOP" in text, "_p03" in p.name)
                elif p.suffix == ".pdf":
                    self.assertTrue(p.read_bytes().startswith(b"%PDF"))
                else:
                    self.assertTrue(p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            html = output.html_path.read_text()
            self.assertNotIn("__PROTOCOL_DATA__", html)
            self.assertNotIn('src="http', html)
            self.assertIn('"rounds_per_page": 2', html)

    @unittest.skipUnless(HAS_PLOTS, "Install requirements-plots.txt to test rendering")
    def test_model_reply_is_escaped_inside_html_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            malicious = '</script><script>alert("raw-reply")</script>'
            path = self.write_log(directory, RunConfig(n=4), [("0000", "0000")], alice=lambda v: malicious)
            output = export_protocols(path, formats=("svg",), episode=1)
            html = output.html_path.read_text()
            self.assertNotIn(malicious, html)
            self.assertIn("\\u003c/script\\u003e", html)

    @unittest.skipUnless(HAS_PLOTS, "Install requirements-plots.txt to test rendering")
    def test_visualize_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_log(directory, RunConfig(n=4), [("0000", "0000")])
            with patch("sys.argv", ["halfduplex", "visualize", str(path), "--formats", "svg"]), contextlib.redirect_stdout(io.StringIO()) as stream:
                main()
            result = json.loads(stream.getvalue())
            self.assertEqual(result["episodes"], 1)
            self.assertEqual(result["figures"], 2)
            self.assertTrue(Path(result["html"]).exists())


if __name__ == "__main__":
    unittest.main()
