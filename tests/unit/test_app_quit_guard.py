"""Unit tests for app quit-event guarding logic."""

from __future__ import annotations

from unittest.mock import MagicMock

from PySide6.QtCore import QEvent

from neurabreak.core.config import ConfigManager
from neurabreak.ui.app import NeuraBreakApp, _QuitEventGuard


def test_quit_guard_blocks_unexpected_quit_event(qapp):
    guard = _QuitEventGuard(qapp)

    blocked = guard.eventFilter(qapp, QEvent(QEvent.Type.Quit))
    assert blocked is True


def test_quit_guard_allows_only_one_explicit_quit_event(qapp):
    guard = _QuitEventGuard(qapp)

    guard.allow_next_quit()
    first = guard.eventFilter(qapp, QEvent(QEvent.Type.Quit))
    second = guard.eventFilter(qapp, QEvent(QEvent.Type.Quit))

    assert first is False
    assert second is True


def test_request_app_quit_marks_next_quit_as_explicit(tmp_path, qapp):
    cfg = ConfigManager.load(path=tmp_path / "config.toml")
    app = NeuraBreakApp(cfg)

    guard = _QuitEventGuard(qapp)
    app._quit_guard = guard
    app._qt_app = MagicMock()

    app._request_app_quit()

    app._qt_app.quit.assert_called_once()
    assert app._explicit_quit_requested is True
    assert guard.eventFilter(qapp, QEvent(QEvent.Type.Quit)) is False
    assert guard.eventFilter(qapp, QEvent(QEvent.Type.Quit)) is True


def test_run_main_loop_restarts_on_unexpected_exit_then_quits_explicitly(tmp_path):
    cfg = ConfigManager.load(path=tmp_path / "config.toml")
    app = NeuraBreakApp(cfg)

    app._qt_app = MagicMock()
    app._quit_guard = MagicMock()
    app._on_quit = MagicMock()

    call_count = {"n": 0}

    def _exec_side_effect():
        call_count["n"] += 1
        if call_count["n"] == 1:
            return 0
        app._explicit_quit_requested = True
        return 0

    app._qt_app.exec.side_effect = _exec_side_effect

    exit_code = app._run_main_loop()

    assert exit_code == 0
    assert app._qt_app.exec.call_count == 2
    app._quit_guard.reset.assert_called_once()
    app._on_quit.assert_called_once()


def test_run_main_loop_shuts_down_after_repeated_unexpected_exits(tmp_path):
    cfg = ConfigManager.load(path=tmp_path / "config.toml")
    app = NeuraBreakApp(cfg)

    app._qt_app = MagicMock()
    app._qt_app.exec.side_effect = [0, 0, 0]
    app._quit_guard = MagicMock()
    app._on_quit = MagicMock()

    exit_code = app._run_main_loop()

    assert exit_code == 0
    assert app._qt_app.exec.call_count == 3
    assert app._quit_guard.reset.call_count == 2
    app._on_quit.assert_called_once()
