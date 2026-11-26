import contextlib
import io
import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from halfduplex import PlayerConfig, RunConfig, run_experiment
from halfduplex.backends import CodexPlayer, audit_event, command_for
from halfduplex.logging import RunLog

FAKE_CLI = '''#!/usr/bin/python3
import json, os, pathlib, sys, time
if '--version' in sys.argv:
    print('fake-codex test-only')
    sys.exit(0)
if '--ephemeral' in sys.argv and 'resume' in sys.argv:
    sys.exit(4)  # Ephemeral threads cannot be resumed.
prompt = sys.stdin.read()
parallel = pathlib.Path(sys.argv[0]).name == 'parallel-codex'
diagnostics = pathlib.Path(sys.argv[0]).name.endswith('diagnostic-codex')
calculations = pathlib.Path(sys.argv[0]).name.endswith('calculation-codex')
tracking = pathlib.Path(sys.argv[0]).name == 'tracked-calculation-codex'
positions = pathlib.Path(sys.argv[0]).name.endswith('positions-calculation-codex')
combined = pathlib.Path(sys.argv[0]).name == 'full-positions-calculation-codex'
thread_id = pathlib.Path.cwd().parent.name if parallel or diagnostics or calculations else 'test-thread'
resume = 'resume' in sys.argv
fork = 'fork' in sys.argv
if ((parallel or diagnostics) and resume or fork) and thread_id not in sys.argv:
    sys.exit(6)
if fork:
    thread_id += '-fork'
print(json.dumps({'type':'thread.started','thread_id':thread_id}),flush=True)
print('diagnostic stderr',file=sys.stderr,flush=True)
if parallel and not resume and thread_id.startswith('halfduplex-alice-'):
    ready = pathlib.Path(sys.argv[0]).parent / 'ready'
    ready.mkdir(exist_ok=True)
    (ready / thread_id).touch()
    deadline = time.monotonic() + 5
    while len(list(ready.iterdir())) < 4:
        if time.monotonic() > deadline:
            sys.exit(5)
        time.sleep(0.01)
if 'timeout' in pathlib.Path(sys.argv[0]).name:
    time.sleep(10)
if 'fail' in pathlib.Path(sys.argv[0]).name and (not diagnostics or fork):
    sys.exit(2)
if 'forbidden' in pathlib.Path(sys.argv[0]).name:
    print(json.dumps({'type':'item.started','item':{'type':'web_search','status':'in_progress'}}),flush=True)
    time.sleep(10)
if 'pipe-holder' in pathlib.Path(sys.argv[0]).name:
    import subprocess
    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
reply = pathlib.Path(sys.argv[sys.argv.index('--output-last-message')+1])
token = 'LISTEN' if parallel and not resume else 'ANSWER_0'
if diagnostics or calculations:
    if fork:
        token = 'Я использовал неправильное промежуточное значение.'
    else:
        position_file = pathlib.Path(os.environ['CODEX_HOME']) / 'game-position'
        position = int(position_file.read_text()) if position_file.exists() else 0
        position_file.write_text(str(position + 1))
        token = 'SEND_1' if position < 2 else 'ANSWER_0'
        if calculations:
            alice = thread_id.startswith('halfduplex-alice-')
            moves = ('SEND_0 SEND_0 SEND_1 LISTEN LISTEN LISTEN LISTEN SEND_0 LISTEN ANSWER_1' if alice
                     else 'LISTEN LISTEN LISTEN LISTEN LISTEN LISTEN SEND_1 LISTEN SEND_1 ANSWER_1')
            if tracking:
                moves = ('SEND_0 SEND_0 LISTEN SEND_1 SEND_1 LISTEN SEND_1 LISTEN ANSWER_1' if alice
                         else 'LISTEN LISTEN LISTEN LISTEN SEND_1 SEND_1 LISTEN SEND_1 ANSWER_1')
            if positions:
                moves = ('SEND_0 SEND_0 SEND_1 LISTEN SEND_1 LISTEN SEND_1 LISTEN LISTEN SEND_0 LISTEN ANSWER_0'
                         if alice else 'LISTEN LISTEN SEND_1 SEND_1 SEND_1 LISTEN LISTEN SEND_1 LISTEN LISTEN SEND_0 ANSWER_0')
            token = moves.split()[position]
            tracks_positions = positions and alice and position >= 8
            tracks_alice_bit = combined and alice and position >= 2
            expects_schema = ((position < 2 or tracks_positions or tracks_alice_bit) if alice
                              else position >= (2 if tracking or positions else 5))
            assert ('--output-schema' in sys.argv) == expects_schema
            if expects_schema:
                schema = pathlib.Path(sys.argv[sys.argv.index('--output-schema')+1])
                state = {}
                if tracking and not alice:
                    state = {'accumulator_before': int(position <= 4), 'accumulator_after': int(position < 4)}
                    assert 'accumulator_after' in prompt
                if (positions and not alice) or tracks_alice_bit:
                    state = {'accumulator_before': 0, 'accumulator_after': 0}
                    assert 'accumulator_after' in prompt
                if tracks_positions:
                    state['remaining_positions'] = [4] if position == 8 else []
                    assert 'remaining_positions' in prompt
                    assert '[4]' not in prompt
                assert json.loads(schema.read_text())['required'] == [*state, 'calculation', 'action']
                assert 'Верни ровно один токен' not in prompt
                if 'invalid-' not in pathlib.Path(sys.argv[0]).name:
                    role = 'Alice' if alice else 'Bob'
                    token = json.dumps({**state, 'calculation': f'локальный расчёт {role}', 'action': token})
            else:
                assert 'Верни ровно один токен' in prompt
reply.write_text(token + '\\n')
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':token}}),flush=True)
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':2}}),flush=True)
'''


class BackendTests(unittest.TestCase):
    def fake(self, root, name="fake-codex"):
        executable = Path(root) / name
        executable.write_text(FAKE_CLI)
        executable.chmod(0o700)
        return str(executable)

    def test_policy_pins_start_and_resume_and_distinct_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            player = PlayerConfig(model="alice-model", allow_python=True)
            for session in (None, "thread-id"):
                command = command_for("codex", player, workspace=root, reply_path=root / "reply",
                                      python_log=root / "python.jsonl", session_id=session, persistent=True)
                text = " ".join(command)
                for required in ('web_search="disabled"', 'features.skip_host_skill_discovery=true',
                                 '--ignore-user-config', '--disable shell_tool', '--disable apps',
                                 '--disable plugins', '--disable browser_use', '--sandbox read-only',
                                 'mcp_servers.local_python.required=true', '--json', '--model alice-model'):
                    self.assertIn(required, text)
                self.assertEqual("resume" in command, bool(session))
                self.assertIn('mcp_servers.local_python.tools.python.approval_mode="approve"', text)
            bob = command_for("codex", PlayerConfig(model="bob-model", allow_python=False), workspace=root,
                              reply_path=root / "reply", python_log=root / "py", session_id=None, persistent=False)
            self.assertIn("bob-model", bob)
            self.assertNotIn("mcp_servers.local_python.required=true", " ".join(bob))
            self.assertIn("--ephemeral", bob)

    def test_streams_and_session_metadata_are_retained(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CODEX_HOME": directory}):
            config = RunConfig(codex_executable=self.fake(directory), output_dir=directory)
            log = RunLog(directory, fsync=False)
            model = CodexPlayer(config, PlayerConfig(allow_python=False), "Alice", log, persistent=True)
            try:
                self.assertEqual(model.turn("first").strip(), "ANSWER_0")
                self.assertEqual(model.session_id, "test-thread")
                self.assertEqual(model.turn("second").strip(), "ANSWER_0")
                calls = list((log.directory / "calls").iterdir())
                self.assertEqual(len(calls), 2)
                commands = [json.loads((p / "metadata.json").read_text())["command"] for p in calls]
                self.assertTrue(any("resume" in c for c in commands))
                for path in calls:
                    self.assertIn("diagnostic stderr", (path / "stderr.log").read_text())
                    self.assertIn('"usage"', (path / "stdout.jsonl").read_text())
                    self.assertTrue((path / "prompt.txt").is_file())
                    self.assertEqual(json.loads((path / "metadata.json").read_text())["outcome"], "completed")
                self.assertNotIn("OPENAI_API_KEY", model.env)
            finally:
                model.close()
                log.close()

    def test_timeout_and_forbidden_tools_fail_with_raw_logs(self):
        for name, timeout in (("timeout-codex", 0.1), ("forbidden-codex", 2)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CODEX_HOME": directory}):
                config = RunConfig(codex_executable=self.fake(directory, name), output_dir=directory)
                log = RunLog(directory, fsync=False)
                model = CodexPlayer(config, PlayerConfig(allow_python=False, timeout_seconds=timeout), "Alice", log, persistent=False)
                try:
                    with self.assertRaises(Exception):
                        model.turn("test")
                    call = next((log.directory / "calls").iterdir())
                    self.assertTrue((call / "stdout.jsonl").exists())
                    self.assertEqual(json.loads((call / "metadata.json").read_text())["outcome"], "error")
                finally:
                    model.close()
                    log.close()

    def test_failed_run_keeps_partial_diagnostics_and_summary(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CODEX_HOME": directory}):
            config = RunConfig(codex_executable=self.fake(directory, "fail-codex"), output_dir=directory,
                               alice=PlayerConfig(allow_python=False), bob=PlayerConfig(backend="python"), fsync=False)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                run_experiment(config)
            run = next(p for p in Path(directory).glob("e4_*"))
            summary = json.loads((run / "summary.json").read_text())
            self.assertFalse(summary["run_complete"])
            self.assertEqual(summary["joint_accuracy"]["total"], 0)
            self.assertIn("model.error", (run / "events.jsonl").read_text())
            self.assertTrue(next((run / "calls").glob("*/stderr.log")).is_file())

    def test_step_requests_start_independent_ephemeral_threads(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CODEX_HOME": directory}):
            config = RunConfig(codex_executable=self.fake(directory), output_dir=directory,
                               alice=PlayerConfig(allow_python=False), bob=PlayerConfig(allow_python=False),
                               n=1, case_limit=1, experiment="step", fsync=False)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_experiment(config)
            self.assertTrue(result.summary["run_complete"])
            self.assertEqual(result.summary["decision_accuracy"]["total"], 8)
            for path in (result.directory / "calls").glob("*/metadata.json"):
                command = json.loads(path.read_text())["command"]
                self.assertIn("--ephemeral", command)
                self.assertNotIn("resume", command)

    def test_first_deviation_is_explained_in_a_fork_and_the_game_continues(self):
        for fails in (False, True):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as directory, \
                    patch.dict(os.environ, {"CODEX_HOME": directory}):
                executable = self.fake(directory, "fail-diagnostic-codex" if fails else "diagnostic-codex")
                config = RunConfig(protocol="baseline", n=1, case_limit=None, case=("0", "0"),
                                   codex_executable=executable, output_dir=directory,
                                   alice=PlayerConfig(allow_python=False), bob=PlayerConfig(backend="python"),
                                   self_check=True, explain_deviations=True, fsync=False)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_experiment(config)
                self.assertTrue(result.summary["run_complete"])
                self.assertEqual(result.summary["joint_accuracy"]["count"], 1)
                self.assertEqual(result.summary["exact_execution"]["count"], 0)
                rows = (result.directory / "deviation_explanations.jsonl").read_text().splitlines()
                self.assertEqual(len(rows), 1)
                explanation = json.loads(rows[0])
                self.assertEqual((explanation["role"], explanation["round"]), ("Alice", 1))
                self.assertEqual((explanation["expected"], explanation["actual"].strip()), ("SEND_0", "SEND_1"))
                if fails:
                    self.assertEqual(explanation["type"], "deviation.explanation_failed")
                    self.assertEqual(explanation["error"], "RuntimeError")
                else:
                    self.assertEqual(explanation["type"], "deviation.explanation")
                    self.assertIn("промежуточное значение", explanation["explanation"])
                events = [json.loads(line) for line in (result.directory / "events.jsonl").read_text().splitlines()]
                game = [e for e in events if e["type"] == "model.finished" and not e.get("purpose")]
                branches = [e for e in events if e["type"] == "model.finished" and e.get("purpose")]
                self.assertEqual([e["position"] for e in game], [1, 2, 3])
                self.assertEqual(len(branches), 1)
                branch = branches[0]
                self.assertIn("fork", branch["command"])
                session = game[0]["session_id_after"]
                self.assertEqual(branch["session_id_before"], session)
                self.assertNotEqual(branch["session_id_after"], session)
                self.assertEqual([e["session_id_before"] for e in game[1:]], [session, session])
                for turn in game:
                    prompt = result.directory / "calls" / turn["call_id"] / "prompt.txt"
                    self.assertIn("самостоятельно восстанови", prompt.read_text())
                prompt = result.directory / "calls" / branch["call_id"] / "prompt.txt"
                self.assertIn("Если не помнишь или не уверен", prompt.read_text())
                self.assertNotIn("SEND_0", prompt.read_text())

    def test_parallel_episodes_overlap_and_keep_sessions_separate(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CODEX_HOME": directory}):
            settings = PlayerConfig(allow_python=False, timeout_seconds=10)
            config = RunConfig(codex_executable=self.fake(directory, "parallel-codex"), output_dir=directory,
                               alice=settings, bob=settings, n=2, case_limit=3, repeats=2, jobs=4, fsync=False)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_experiment(config)
            self.assertTrue(result.summary["run_complete"])
            self.assertEqual(result.summary["joint_accuracy"]["total"], 6)

            active, peak, calls = set(), 0, defaultdict(list)
            for line in (result.directory / "events.jsonl").read_text().splitlines():
                event = json.loads(line)
                if event["type"] == "model.started":
                    active.add(event["call_id"])
                    peak = max(peak, len(active))
                elif event["type"] == "model.finished":
                    active.remove(event["call_id"])
                    calls[event["case_index"], event["repeat"], event["role"]].append(event)
            self.assertFalse(active)
            self.assertEqual(peak, 4)
            self.assertEqual(len(calls), 12)
            sessions = set()
            for first, followup in calls.values():
                self.assertIsNone(first["session_id_before"])
                self.assertEqual(followup["session_id_before"], first["session_id_after"])
                self.assertEqual(followup["session_id_after"], first["session_id_after"])
                self.assertEqual([first["position"], followup["position"]], [1, 2])
                self.assertNotIn(first["session_id_after"], sessions)
                sessions.add(first["session_id_after"])

    def test_ip_zero_calculations_are_saved_and_only_actions_enter_the_game(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory, \
                    patch.dict(os.environ, {"CODEX_HOME": directory}):
                executable = self.fake(directory, "invalid-calculation-codex" if invalid else "calculation-codex")
                player = PlayerConfig(allow_python=False)
                config = RunConfig(task="ip", channel="zero", n=5, case_limit=None,
                                   case=("00111", "11111"), codex_executable=executable,
                                   output_dir=directory, alice=player, bob=player,
                                   record_calculations=True, fsync=False)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_experiment(config)
                self.assertTrue(result.summary["run_complete"])
                calculations = [json.loads(line) for line in
                                (result.directory / "calculations.jsonl").read_text().splitlines()]
                if invalid:
                    self.assertEqual(result.summary["first_errors"], {"invalid_format": 1})
                    self.assertEqual(len(calculations), 1)
                    self.assertEqual(calculations[0]["type"], "decision.calculation_invalid")
                    self.assertEqual(calculations[0]["raw_reply"].strip(), "SEND_0")
                    continue
                self.assertEqual(result.summary["exact_execution"]["count"], 1)
                self.assertEqual([(row["role"], row["round"]) for row in calculations],
                                 [("Alice", 1), ("Alice", 2), *[("Bob", r) for r in range(6, 11)]])
                self.assertNotIn("локальный расчёт", result.log_path.read_text())
                for row in calculations:
                    call = result.directory / "calls" / row["call_id"]
                    self.assertEqual(json.loads((call / "reply.txt").read_text())["action"], row["action"])
                    self.assertEqual(row["calculation"], f"локальный расчёт {row['role']}")
                for path in (result.directory / "calls").glob("*/prompt.txt"):
                    self.assertNotIn("локальный расчёт", path.read_text())

    def test_ip_zero_bob_records_accumulator_from_first_pass(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CODEX_HOME": directory}):
            player = PlayerConfig(allow_python=False)
            config = RunConfig(task="ip", channel="zero", n=5, case_limit=None,
                               case=("11000", "10110"), codex_executable=self.fake(directory, "tracked-calculation-codex"),
                               output_dir=directory, alice=player, bob=player, fsync=False,
                               record_calculations=True, track_bob_accumulator=True)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_experiment(config)
            self.assertEqual(result.summary["exact_execution"]["count"], 1)
            rows = [json.loads(line) for line in (result.directory / "calculations.jsonl").read_text().splitlines()]
            bob = [row for row in rows if row["role"] == "Bob"]
            self.assertEqual([r["round"] for r in bob], list(range(3, 10)))
            self.assertEqual([(r["accumulator_before"], r["accumulator_after"]) for r in bob],
                             [(1, 1), (1, 1), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0)])
            self.assertEqual(bob[-2]["action"], "SEND_1")
            self.assertNotIn("accumulator_before", "\n".join(result.log_path.read_text().splitlines()[1:]))
            for row in rows:
                prompt = (result.directory / "calls" / row["call_id"] / "prompt.txt").read_text()
                if row["role"] == "Alice":
                    self.assertNotIn("accumulator_after", prompt)
                    self.assertNotIn("accumulator_after", row)
                elif row["round"] == 5:
                    self.assertIn("Раунд 4 завершён", prompt)
                    self.assertIn("Твоё наблюдение: 1", prompt)
                    self.assertIn("из предыдущего ответа", prompt)

    def test_ip_zero_alice_records_remaining_positions_at_the_phase_transition(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CODEX_HOME": directory}):
            player = PlayerConfig(allow_python=False)
            executable = self.fake(directory, "positions-calculation-codex")
            config = RunConfig(task="ip", channel="zero", n=12, case_limit=None,
                               case=("000100100001", "000000010100"), codex_executable=executable,
                               output_dir=directory, alice=player, bob=player, fsync=False,
                               record_calculations=True, track_bob_accumulator=True, track_alice_second_pass=True)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_experiment(config)
            self.assertEqual(result.summary["exact_execution"]["count"], 1)
            rows = [json.loads(line) for line in (result.directory / "calculations.jsonl").read_text().splitlines()]
            alice = [row for row in rows if row["role"] == "Alice"]
            self.assertEqual([r["round"] for r in alice], [1, 2, 9, 10, 11, 12])
            self.assertEqual([(r["remaining_positions"], r["action"]) for r in alice[2:]],
                             [([4], "LISTEN"), ([], "SEND_0"), ([], "LISTEN"), ([], "ANSWER_0")])
            self.assertTrue(all("remaining_positions" not in row for row in rows if row["role"] == "Bob"))
            self.assertNotIn("remaining_positions", result.log_path.read_text())

    def test_ip_zero_alice_records_accumulator_alongside_remaining_positions(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CODEX_HOME": directory}):
            player = PlayerConfig(allow_python=False)
            executable = self.fake(directory, "full-positions-calculation-codex")
            config = RunConfig(task="ip", channel="zero", n=12, case_limit=None,
                               case=("000100100001", "000000010100"), codex_executable=executable,
                               output_dir=directory, alice=player, bob=player, fsync=False,
                               record_calculations=True, track_bob_accumulator=True,
                               track_alice_second_pass=True, track_alice_accumulator=True)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_experiment(config)
            self.assertEqual(result.summary["exact_execution"]["count"], 1)
            rows = [json.loads(line) for line in (result.directory / "calculations.jsonl").read_text().splitlines()]
            alice = {row["round"]: row for row in rows if row["role"] == "Alice"}
            self.assertEqual(set(alice), set(range(1, 13)))
            self.assertNotIn("accumulator_after", alice[2])
            self.assertEqual({r for r in alice if "accumulator_after" in alice[r]}, set(range(3, 13)))
            self.assertEqual({r for r in alice if "remaining_positions" in alice[r]}, set(range(9, 13)))
            self.assertEqual((alice[9]["remaining_positions"], alice[10]["remaining_positions"]), ([4], []))
            self.assertEqual((alice[10]["accumulator_before"], alice[10]["accumulator_after"], alice[10]["action"]),
                             (0, 0, "SEND_0"))
            self.assertTrue(all("remaining_positions" not in row for row in rows if row["role"] == "Bob"))

    def test_timeout_applies_to_descendants_holding_output_pipes(self):
        import subprocess
        import time

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CODEX_HOME": directory}):
            config = RunConfig(codex_executable=self.fake(directory, "pipe-holder-codex"), output_dir=directory)
            log = RunLog(directory, fsync=False)
            model = CodexPlayer(config, PlayerConfig(allow_python=False, timeout_seconds=0.2),
                                "Alice", log, persistent=False)
            try:
                started = time.monotonic()
                with self.assertRaises(subprocess.TimeoutExpired):
                    model.turn("test")
                self.assertLess(time.monotonic() - started, 3)
            finally:
                model.close()
                log.close()

    def test_unknown_tool_items_are_denied(self):
        for kind in ("web_search", "command_execution", "file_change", "new_future_tool"):
            self.assertFalse(audit_event({"item": {"type": kind}}, True)["allowed"])
        event = {"item": {"type": "mcp_tool_call", "server": "local_python", "tool": "python"}}
        self.assertTrue(audit_event(event, True)["allowed"])
        self.assertFalse(audit_event(event, False)["allowed"])
        self.assertIsNone(audit_event({"item": {"type": "error", "message": "CLI startup warning"}}, True))


if __name__ == "__main__":
    unittest.main()
