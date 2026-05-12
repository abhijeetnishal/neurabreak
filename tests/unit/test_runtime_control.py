"""Unit tests for runtime control-file coordination."""

from __future__ import annotations

from neurabreak.core.runtime_control import (
    RuntimeSession,
    clear_runtime_session,
    consume_quit_request,
    request_running_instance_quit,
    start_runtime_session,
)


def test_request_quit_without_runtime_session_returns_false(tmp_path):
    runtime_file = tmp_path / "runtime-session.json"
    request_file = tmp_path / "quit-request.json"

    requested = request_running_instance_quit(
        runtime_file=runtime_file,
        request_file=request_file,
        wait=False,
    )

    assert requested is False
    assert not request_file.exists()


def test_quit_request_is_consumed_by_matching_session(tmp_path):
    runtime_file = tmp_path / "runtime-session.json"
    request_file = tmp_path / "quit-request.json"
    session = start_runtime_session(runtime_file=runtime_file)

    requested = request_running_instance_quit(
        runtime_file=runtime_file,
        request_file=request_file,
        wait=False,
    )

    assert requested is True
    assert consume_quit_request(session, request_file=request_file) is True
    assert consume_quit_request(session, request_file=request_file) is False


def test_quit_request_ignores_different_session_token(tmp_path):
    runtime_file = tmp_path / "runtime-session.json"
    request_file = tmp_path / "quit-request.json"
    start_runtime_session(runtime_file=runtime_file)
    other_session = RuntimeSession(pid=123, token="different-token")

    request_running_instance_quit(
        runtime_file=runtime_file,
        request_file=request_file,
        wait=False,
    )

    assert consume_quit_request(other_session, request_file=request_file) is False
    assert request_file.exists()


def test_clear_runtime_session_only_removes_matching_session(tmp_path):
    runtime_file = tmp_path / "runtime-session.json"
    first_session = start_runtime_session(runtime_file=runtime_file)
    second_session = start_runtime_session(runtime_file=runtime_file)

    clear_runtime_session(first_session, runtime_file=runtime_file)
    assert runtime_file.exists()

    clear_runtime_session(second_session, runtime_file=runtime_file)
    assert not runtime_file.exists()
