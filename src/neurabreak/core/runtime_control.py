"""Runtime control files for coordinating a running NeuraBreak instance."""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurabreak.core.config import CONFIG_DIR

RUNTIME_SESSION_FILE = CONFIG_DIR / "runtime-session.json"
QUIT_REQUEST_FILE = CONFIG_DIR / "quit-request.json"
QUIT_WAIT_TIMEOUT_SEC = 5.0


@dataclass(frozen=True)
class RuntimeSession:
    """Identity for the app process currently accepting runtime commands."""

    pid: int
    token: str


def start_runtime_session(runtime_file: Path | None = None) -> RuntimeSession:
    """Register this process as the active NeuraBreak instance."""
    session = RuntimeSession(pid=os.getpid(), token=uuid.uuid4().hex)
    _write_json_atomic(
        runtime_file or RUNTIME_SESSION_FILE,
        {
            "pid": session.pid,
            "token": session.token,
            "started_at": time.time(),
        },
    )
    return session


def clear_runtime_session(
    session: RuntimeSession,
    runtime_file: Path | None = None,
) -> None:
    """Remove this process' runtime session without disturbing a newer one."""
    path = runtime_file or RUNTIME_SESSION_FILE
    current = _read_json(path)
    if current.get("token") != session.token:
        return

    with suppress(FileNotFoundError):
        path.unlink()


def request_running_instance_quit(
    *,
    runtime_file: Path | None = None,
    request_file: Path | None = None,
    wait: bool = True,
    timeout_sec: float = QUIT_WAIT_TIMEOUT_SEC,
) -> bool:
    """Ask the currently registered NeuraBreak instance to quit.

    Returns True when a quit request was written for a known runtime session.
    The running app consumes the request from its Qt event loop and performs
    the normal graceful shutdown path.
    """
    runtime_path = runtime_file or RUNTIME_SESSION_FILE
    request_path = request_file or QUIT_REQUEST_FILE
    session = _read_json(runtime_path)
    token = session.get("token")
    pid = session.get("pid")
    if not isinstance(token, str) or not isinstance(pid, int):
        return False

    _write_json_atomic(
        request_path,
        {
            "pid": pid,
            "token": token,
            "requester_pid": os.getpid(),
            "requested_at": time.time(),
        },
    )

    if not wait:
        return True

    deadline = time.monotonic() + max(timeout_sec, 0.0)
    while time.monotonic() < deadline:
        current = _read_json(runtime_path)
        if current.get("token") != token:
            return True
        time.sleep(0.1)

    return True


def consume_quit_request(
    session: RuntimeSession,
    request_file: Path | None = None,
) -> bool:
    """Return True once for a quit request addressed to this session."""
    path = request_file or QUIT_REQUEST_FILE
    request = _read_json(path)
    if request.get("token") != session.token:
        return False

    with suppress(FileNotFoundError):
        path.unlink()
    return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}
    return data


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
