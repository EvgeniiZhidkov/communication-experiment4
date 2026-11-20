"""Append diagnostics immediately, outside either player's computation sandbox."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlWriter:
    def __init__(self, path: Path, *, fsync: bool = True):
        self.path = path
        self.fsync = fsync
        self.lock = threading.Lock()
        self.stream = path.open("x", encoding="utf-8")

    def emit(self, kind: str, **fields) -> None:
        with self.lock:
            self.stream.write(json.dumps({"type": kind, "time_utc": utc_now(), **fields}, ensure_ascii=False) + "\n")
            self.stream.flush()
            if self.fsync:
                os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


class RunLog:
    def __init__(self, output_dir: str | Path, *, fsync: bool = True):
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.directory = root / f"e4_{stamp}_{uuid4().hex[:8]}"
        self.directory.mkdir(mode=0o700)
        (self.directory / "calls").mkdir()
        self.events = JsonlWriter(self.directory / "events.jsonl", fsync=fsync)
        self.journal = JsonlWriter(self.directory / "journal.jsonl", fsync=fsync)

    def close(self):
        self.events.close()
        self.journal.close()


def source_manifest() -> dict:
    package = Path(__file__).parent
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob("*.py"))}
    commit = None
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package, capture_output=True,
                                text=True, timeout=3)
        if result.returncode == 0:
            commit = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"created_utc": utc_now(), "python": sys.version, "platform": platform.platform(),
            "git_commit": commit, "source_sha256": hashes}
