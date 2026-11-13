import contextlib
import io
import json
import os
import tempfile
import unittest
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
print(json.dumps({'type':'thread.started','thread_id':'test-thread'}),flush=True)
print('diagnostic stderr',file=sys.stderr,flush=True)
if 'timeout' in pathlib.Path(sys.argv[0]).name:
    time.sleep(10)
if 'fail' in pathlib.Path(sys.argv[0]).name:
    sys.exit(2)
if 'forbidden' in pathlib.Path(sys.argv[0]).name:
    print(json.dumps({'type':'item.started','item':{'type':'web_search','status':'in_progress'}}),flush=True)
    time.sleep(10)
if 'pipe-holder' in pathlib.Path(sys.argv[0]).name:
    import subprocess
    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
reply = pathlib.Path(sys.argv[sys.argv.index('--output-last-message')+1])
reply.write_text('ANSWER_0\\n')
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'ANSWER_0'}}),flush=True)
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
