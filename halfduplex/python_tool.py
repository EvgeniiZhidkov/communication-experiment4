"""A single MCP tool: Python in a fresh Linux filesystem/network namespace.

No evaluator/AST sandbox is used: bubblewrap hides the host filesystem and
network. A missing or unusable sandbox is an error, never a fallback to exec().
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


class SandboxUnavailable(RuntimeError):
    pass


class PythonSandbox:
    def __init__(self, *, timeout: float = 5, max_output: int = 65536):
        self.bwrap = shutil.which("bwrap")
        self.python = "/usr/bin/python3"
        self.timeout, self.max_output = timeout, max_output
        if not self.bwrap or not Path(self.python).is_file():
            raise SandboxUnavailable("Нужны Linux/WSL, bubblewrap (bwrap) и /usr/bin/python3.")

    def command(self, code: str) -> list[str]:
        command = [self.bwrap, "--unshare-all", "--die-with-parent", "--new-session"]
        for directory in ("/usr", "/lib", "/lib64", "/bin"):
            if Path(directory).exists():
                command += ["--ro-bind", directory, directory]
        command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                    "--dir", "/work", "--chdir", "/work", "--clearenv",
                    "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/work",
                    self.python, "-I", "-S", "-c", code]
        return command

    @staticmethod
    def _limits():
        resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 ** 2, 256 * 1024 ** 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024, 128 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    def run(self, code: str) -> dict:
        if not isinstance(code, str) or len(code.encode()) > 65536:
            raise ValueError("Код должен быть строкой не более 64 KiB.")
        # Output goes to bounded files, rather than unbounded PIPE buffers.
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(self.command(code), stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, env={"PATH": "/usr/bin:/bin"},
                                       start_new_session=True, preexec_fn=self._limits)
            timed_out = False
            try:
                process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            stdout.seek(0)
            stderr.seek(0)
            out, err = stdout.read(self.max_output + 1), stderr.read(self.max_output + 1)
        return {"returncode": process.returncode, "timed_out": timed_out,
                "stdout": out[:self.max_output].decode("utf-8", errors="replace"),
                "stderr": err[:self.max_output].decode("utf-8", errors="replace"),
                "output_truncated": len(out) > self.max_output or len(err) > self.max_output}

    def preflight(self) -> dict:
        code = """import json, os, socket
s = socket.socket()
s.settimeout(0.3)
network_errno = s.connect_ex(('192.0.2.1', 443))
s.close()
assert network_errno != 0, 'Network unexpectedly reachable'
assert not os.path.exists('/home'), 'Host home is visible'
assert not os.environ.get('OPENAI_API_KEY')
assert not os.environ.get('CODEX_API_KEY')
print(json.dumps({'python_ok': 6*7 == 42, 'network_errno': network_errno,
                  'host_home_visible': os.path.exists('/home'), 'cwd': os.getcwd()}))
"""
        result = self.run(code)
        if result["returncode"] != 0 or result["timed_out"]:
            raise SandboxUnavailable("Изоляция Python не прошла проверку: " + result["stderr"])
        return json.loads(result["stdout"])


def serve(log_path: Path) -> None:
    sandbox = PythonSandbox()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(kind: str, **fields):
        from datetime import datetime, timezone
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"type": kind, "time_utc": datetime.now(timezone.utc).isoformat(),
                                     **fields}, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    for line in sys.stdin:
        request = {}
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("MCP-запрос должен быть JSON-объектом.")
            request_id = request.get("id")
            method = request.get("method")
            if "id" not in request:
                continue
            if method == "initialize":
                result = {"protocolVersion": request.get("params", {}).get("protocolVersion", "2024-11-05"),
                          "capabilities": {"tools": {}},
                          "serverInfo": {"name": "halfduplex-local-python", "version": "1.0"}}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": [{"name": "python", "description":
                          "Выполни Python для локального вычисления. Нет сети, файлов проекта, "
                          "истории другого игрока и переменных среды хоста. Каждый вызов независим. "
                          "Результат — stdout/stderr; печатай нужные значения через print().",
                          "inputSchema": {"type": "object", "properties": {"code": {"type": "string"}},
                                          "required": ["code"], "additionalProperties": False}}]}
            elif method == "tools/call":
                params = request["params"]
                if params.get("name") != "python":
                    raise ValueError("Разрешён только инструмент python.")
                code = params.get("arguments", {}).get("code")
                emit("python.started", request_id=request["id"], code=code)
                outcome = sandbox.run(code)
                emit("python.completed", request_id=request["id"], **outcome)
                result = {"content": [{"type": "text", "text": json.dumps(outcome, ensure_ascii=False)}],
                          "isError": outcome["returncode"] != 0 or outcome["timed_out"]}
            else:
                response = {"jsonrpc": "2.0", "id": request["id"],
                            "error": {"code": -32601, "message": f"Unknown method: {method}"}}
                print(json.dumps(response), flush=True)
                continue
            response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        except Exception as exc:
            emit("python.error", error=type(exc).__name__, message=str(exc))
            code = (-32700 if isinstance(exc, json.JSONDecodeError) else
                    -32600 if not isinstance(request, dict) else -32602)
            response = {"jsonrpc": "2.0", "id": request_id,
                        "error": {"code": code, "message": str(exc)}}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    serve(parser.parse_args().log)
