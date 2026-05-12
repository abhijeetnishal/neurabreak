"""Unit tests for the NeuraBreak CLI entry point."""

from __future__ import annotations

import sys

from neurabreak.__main__ import main


def test_quit_flag_requests_running_instance(monkeypatch):
    calls = []

    def _request_quit():
        calls.append(True)
        return True

    monkeypatch.setattr(sys, "argv", ["python", "-m", "neurabreak", "--quit"])
    monkeypatch.setattr(
        "neurabreak.core.runtime_control.request_running_instance_quit",
        _request_quit,
    )

    assert main() == 0
    assert calls == [True]
