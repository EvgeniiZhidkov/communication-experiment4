import unittest

from halfduplex import RunConfig, verify_reference
from halfduplex.analysis import reference_player, score_episode
from halfduplex.core import ALICE, ACTIONS, channel_step, play
from halfduplex.protocols import PROTOCOLS, capacity_width, digits


class ProtocolTests(unittest.TestCase):
    def test_exhaustive_reproduction(self):
        for task, channel in (("ip", "silence"), ("eq", "silence"), ("eq", "zero")):
            name = f"{task}_{channel}"
            for n in range(1, 7):
                with self.subTest(protocol=name, n=n):
                    result = verify_reference(RunConfig(task=task, channel=channel, protocol="specialized",
                                                        n=n, case_limit=min(15, 4 ** n)))
                    self.assertEqual(result["correct"], 4 ** n)
                    self.assertEqual(result["c_max"], PROTOCOLS[name].rounds(n))

    def test_baseline_all_tasks_and_channels(self):
        for task in ("ip", "eq", "disj"):
            for channel in ("silence", "zero"):
                config = RunConfig(protocol="baseline", task=task, channel=channel, n=3)
                result = verify_reference(config)
                self.assertEqual(result["correct"], 64)
                self.assertEqual(result["c_max"], 4)

    def test_channel_all_action_pairs(self):
        for channel in ("silence", "zero"):
            for a in ACTIONS:
                for b in ACTIONS:
                    x, y = channel_step(a, b, channel)
                    if a.startswith("SEND_"):
                        self.assertIsNone(x)  # Sending reveals no collision information.
                    if b.startswith("SEND_"):
                        self.assertIsNone(y)
                    if a == b == "LISTEN":
                        self.assertEqual((x, y), ("SILENCE", "SILENCE") if channel == "silence" else ("0", "0"))
                    elif a == "LISTEN":
                        self.assertEqual(x, b[-1])
                    elif b == "LISTEN":
                        self.assertEqual(y, a[-1])

    def test_early_stop_uses_no_hidden_round(self):
        cfg = RunConfig(task="eq", channel="silence", protocol="specialized", n=2)
        player = reference_player(cfg)
        episode = play(2, "00", "01", player, player, channel=cfg.channel, max_rounds=cfg.max_rounds)
        self.assertEqual(episode.status, "completed")
        self.assertEqual(len(episode.rounds), 2)
        self.assertTrue(all(d.token == "ANSWER_0" for d in episode.terminal))

    def test_invalid_asymmetric_and_round_limit(self):
        cases = ((lambda v: "oops", lambda v: "LISTEN", "invalid_token", 0),
                 (lambda v: "ANSWER_0", lambda v: "LISTEN", "asymmetric_stop", 0),
                 (lambda v: "LISTEN", lambda v: "LISTEN", "round_limit", 2))
        for alice, bob, status, rounds in cases:
            result = play(1, "0", "1", alice, bob, channel="silence", max_rounds=2)
            self.assertEqual(result.status, status)
            self.assertEqual(len(result.rounds), rounds)

    def test_correct_answer_after_wrong_trace_is_not_exact(self):
        cfg = RunConfig(protocol="baseline", task="eq", n=1, case_limit=4)
        reference = reference_player(cfg)
        wrong_trace = lambda v: "LISTEN" if len(v.history) < 2 else "ANSWER_1"
        actual = play(1, "1", "1", wrong_trace, wrong_trace, channel="silence", max_rounds=2)
        expected = play(1, "1", "1", reference, reference, channel="silence", max_rounds=2)
        score = score_episode(cfg, actual, expected, 1)
        self.assertTrue(score["joint_correct"])
        self.assertFalse(score["exact"])
        self.assertFalse(score["actions_match"])

    def test_integer_encodings_at_boundaries(self):
        for base in (2, 3, 5):
            for k in range(1, 10):
                self.assertEqual(capacity_width(base ** k, base), k)
                self.assertEqual(capacity_width(base ** k + 1, base), k + 1)
                value = base ** k - 1
                encoded = digits(value, base, k)
                self.assertEqual(encoded, (base - 1,) * k)


if __name__ == "__main__":
    unittest.main()
