import contextlib
import io
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from halfduplex import PlayerConfig, RunConfig, get_protocol
from halfduplex.__main__ import main
from halfduplex.core import View
from halfduplex.prompts import render_followup, render_prompt


class ConfigurationTests(unittest.TestCase):
    def test_invalid_timeouts_and_flags_fail_before_start(self):
        for timeout in (0, -1, float("nan"), float("inf"), float("-inf"), True, "120"):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                PlayerConfig(timeout_seconds=timeout)
        for allow_python in ("false", 0, None):
            with self.subTest(allow_python=allow_python), self.assertRaises(ValueError):
                PlayerConfig(allow_python=allow_python)
        with self.assertRaises(ValueError):
            RunConfig(round_limit=True)
        with self.assertRaises(ValueError):
            RunConfig(fsync="false")

    def test_jobs_are_validated_before_the_run(self):
        self.assertEqual(RunConfig().jobs, 1)
        self.assertEqual(RunConfig(jobs=4).record()["jobs"], 4)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "jobs"):
                RunConfig(jobs=value)
        with self.assertRaisesRegex(ValueError, "episode"):
            RunConfig(experiment="step", jobs=2)
        with self.assertRaisesRegex(ValueError, "ручного ввода"):
            RunConfig(alice=PlayerConfig(backend="manual"), jobs=2)

    def test_task_channel_and_strategy_can_change_independently(self):
        cfg = RunConfig(task="ip", channel="silence", protocol="specialized", n=5)
        equality = replace(cfg, task="eq")
        zero = replace(equality, channel="zero")
        baseline = replace(zero, protocol="baseline")
        self.assertEqual([(c.implementation.name, c.max_rounds) for c in (cfg, equality, zero, baseline)],
                         [("ip_silence", 5), ("eq_silence", 7), ("eq_zero", 11), ("send_x_then_result", 6)])
        self.assertEqual((baseline.task, baseline.channel), ("eq", "zero"))
        # An experiment budget must not change the reference protocol's bound.
        limited = replace(zero, round_limit=2)
        self.assertEqual(limited.max_rounds, 2)
        self.assertEqual(limited.implementation.rounds(limited.n), 11)

    def test_old_configurations_normalize_to_separate_fields(self):
        for old, mode, task, channel in (("ip_silence", "specialized", "ip", "silence"),
                                         ("eq_silence", "specialized", "eq", "silence"),
                                         ("eq_zero", "specialized", "eq", "zero"),
                                         ("send_x_then_result", "baseline", "ip", "silence")):
            with self.subTest(old=old):
                cfg = RunConfig(protocol=old)
                self.assertEqual(cfg, RunConfig(protocol=mode, task=task, channel=channel))
                self.assertEqual(get_protocol(old), cfg.implementation)
                self.assertEqual(cfg.record()["protocol"], mode)
        explicit = RunConfig(protocol="send_x_then_result", task="disj", channel="zero")
        self.assertEqual(explicit, RunConfig(protocol="baseline", task="disj", channel="zero"))

    def test_prompts_follow_selected_function_and_implementation(self):
        cases = (("specialized", "ip", "silence", "Разбей свой вход на пары."),
                 ("specialized", "ip", "zero", "проход 2 Alice / Bob"),
                 ("specialized", "disj", "zero", "Подпротокол EQ для ветви 00"),
                 ("specialized", "eq", "silence", "основании 5"),
                 ("specialized", "eq", "zero", "основании 3"),
                 ("baseline", "disj", "zero", "Alice отправляет биты x слева направо"))
        for mode, task, channel, instruction in cases:
            for explanation in ("prose", "pseudocode"):
                with self.subTest(mode=mode, task=task, channel=channel, explanation=explanation):
                    cfg = RunConfig(protocol=mode, task=task, channel=channel, explanation=explanation)
                    prompt = render_prompt(cfg, View("Alice", "10101"))
                    self.assertIn(instruction, prompt)
                    self.assertIn(f"{task.upper()}:", prompt)
                    if explanation == "pseudocode":
                        self.assertIn(f"def {cfg.implementation.choose.__name__}(", prompt)
                        if cfg.implementation.local_steps is not None:
                            self.assertIn(f"def {cfg.implementation.local_steps.__name__}(", prompt)

    def test_bob_can_repair_only_his_own_accumulator_before_sending_result(self):
        with self.assertRaisesRegex(ValueError, "track_bob_accumulator"):
            RunConfig(task="ip", channel="zero", allow_bob_accumulator_correction=True)
        control = RunConfig(n=12, task="ip", channel="zero", record_calculations=True,
                            track_bob_accumulator=True)
        repair = replace(control, allow_bob_accumulator_correction=True)
        bob = View("Bob", "010111011100").after("LISTEN", "0").after("LISTEN", "1")
        self.assertEqual(render_followup(repair, bob), render_followup(control, bob))
        bob = bob.after("LISTEN", "0")
        prompt = render_followup(repair, bob)
        self.assertNotIn("если заметил ошибку", render_followup(control, bob))
        self.assertIn("До отправки итогового бита, если заметил ошибку", prompt)
        self.assertIn("исключая этот раунд", prompt)
        self.assertIn("с тем битом, который уже отправил", prompt)
        self.assertIn("Твоё наблюдение: 0", prompt)
        alice = View("Alice", "000101000111").after("SEND_0", None)
        self.assertEqual(render_followup(repair, alice), render_followup(control, alice))

    def test_reference_cli_accepts_separate_fields_and_legacy_names(self):
        cases = ((["--task", "eq", "--channel", "zero", "--protocol", "specialized"], "eq_zero"),
                 (["--task", "ip", "--channel", "zero", "--protocol", "specialized"], "ip_zero"),
                 (["--task", "disj", "--channel", "zero", "--protocol", "specialized"], "disj_zero"),
                 (["--task", "disj", "--channel", "zero", "--protocol", "baseline"], "send_x_then_result"),
                 (["--protocol", "eq_zero"], "eq_zero"))
        for args, implementation in cases:
            with self.subTest(args=args), patch("sys.argv", ["halfduplex", "reference", "--n", "2", *args]):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    main()
                result = json.loads(output.getvalue())
                self.assertEqual(result["implementation"], implementation)
                self.assertEqual(result["correct"], 16)

    def test_unsupported_cli_combination_explains_available_options(self):
        error = io.StringIO()
        with patch("sys.argv", ["halfduplex", "reference", "--task", "disj"]):
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as caught:
                main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("нет реализации", error.getvalue())
        self.assertIn("baseline", error.getvalue())


if __name__ == "__main__":
    unittest.main()
