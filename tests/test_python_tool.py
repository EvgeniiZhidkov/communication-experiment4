import json
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from halfduplex.python_tool import PythonSandbox, SandboxUnavailable, serve


@unittest.skipUnless(os.environ.get("HALFDUPLEX_SANDBOX_TESTS") == "1",
                     "Run scripts/validate_runtime.py where user/network namespaces are permitted")
class PythonIsolationTests(unittest.TestCase):
    def test_math_and_isolation(self):
        sandbox = PythonSandbox()
        preflight = sandbox.preflight()
        self.assertTrue(preflight["python_ok"])
        self.assertFalse(preflight["host_home_visible"])
        self.assertNotEqual(preflight["network_errno"], 0)
        result = sandbox.run("import math, itertools; print(math.factorial(6)); print(list(itertools.product([0,1],repeat=2)))")
        self.assertEqual(result["returncode"], 0)
        self.assertIn("720", result["stdout"])

    def test_host_file_and_environment_are_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "private-input.txt"
            secret.write_text("opponent-input")
            with patch.dict(os.environ, {"PRIVATE_TEST_VALUE": "not-for-the-model"}):
                code = f"import pathlib,os; print(pathlib.Path({str(secret)!r}).exists()); print(os.getenv('PRIVATE_TEST_VALUE'))"
                result = PythonSandbox().run(code)
            self.assertEqual(result["stdout"].splitlines(), ["False", "None"])

    def test_timeout_and_output_limit(self):
        result = PythonSandbox(timeout=0.1).run("while True: pass")
        self.assertTrue(result["timed_out"])
        result = PythonSandbox(max_output=1024).run("print('x'*5000)")
        self.assertTrue(result["output_truncated"])
        self.assertEqual(len(result["stdout"]), 1024)

    def test_mcp_transport_and_tool_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "python.jsonl"
            server = Path(__file__).resolve().parents[1] / "halfduplex" / "python_tool.py"
            requests = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "python", "arguments": {"code": "print(6*7)"}}},
            ]
            result = subprocess.run([sys.executable, str(server), "--log", str(log)],
                                    input="\n".join(json.dumps(r) for r in requests) + "\n",
                                    text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            responses = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual([r["id"] for r in responses], [1, 2, 3])
            self.assertEqual(responses[1]["result"]["tools"][0]["name"], "python")
            outcome = json.loads(responses[2]["result"]["content"][0]["text"])
            self.assertEqual(outcome["stdout"], "42\n")
            self.assertIn("python.completed", log.read_text())


class SandboxFailureTests(unittest.TestCase):
    def test_no_unrestricted_fallback(self):
        with patch("halfduplex.python_tool.shutil.which", return_value=None), self.assertRaises(SandboxUnavailable):
            PythonSandbox()

    def test_malformed_mcp_requests_do_not_stop_the_server(self):
        with tempfile.TemporaryDirectory() as directory, patch("halfduplex.python_tool.PythonSandbox"):
            requests = '[]\nnull\n{broken}\n{"jsonrpc":"2.0","id":7,"method":"ping"}\n'
            output = io.StringIO()
            with patch("sys.stdin", io.StringIO(requests)), contextlib.redirect_stdout(output):
                serve(Path(directory) / "python.jsonl")
            responses = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(len(responses), 4)
            self.assertTrue(all("error" in r for r in responses[:3]))
            self.assertEqual(responses[-1], {"jsonrpc": "2.0", "id": 7, "result": {}})


if __name__ == "__main__":
    unittest.main()
