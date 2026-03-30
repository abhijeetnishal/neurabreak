"""Regression tests for tray menu activation edge-cases on Windows."""

from __future__ import annotations

from PySide6.QtWidgets import QSystemTrayIcon

from neurabreak.core.config import ConfigManager
from neurabreak.ui.tray import NeuraBreakTray, _CONTEXT_MENU_POPUP_DELAY_MS


def _make_tray(qapp, tmp_path, on_quit_requested=None) -> NeuraBreakTray:
    cfg = ConfigManager.load(path=tmp_path / "config.toml")
    return NeuraBreakTray(
        config_manager=cfg,
        app=qapp,
        on_quit_requested=on_quit_requested,
    )


def test_context_clicks_queue_only_one_popup_while_pending(qapp, tmp_path, monkeypatch):
    tray = _make_tray(qapp, tmp_path)
    scheduled: list[tuple[int, object]] = []

    def _fake_single_shot(ms: int, callback) -> None:
        scheduled.append((ms, callback))

    monkeypatch.setattr("neurabreak.ui.tray.QTimer.singleShot", _fake_single_shot)

    reason = QSystemTrayIcon.ActivationReason.Context
    tray._on_activated(reason)
    tray._on_activated(reason)
    tray._on_activated(reason)

    assert len(scheduled) == 1
    assert scheduled[0][0] == _CONTEXT_MENU_POPUP_DELAY_MS


def test_context_clicks_are_debounced_after_popup_runs(qapp, tmp_path, monkeypatch):
    tray = _make_tray(qapp, tmp_path)
    scheduled_calls: list[int] = []

    def _fake_show_context_menu() -> None:
        tray._menu_popup_pending = False

    def _fake_single_shot(ms: int, callback) -> None:
        scheduled_calls.append(ms)
        callback()

    monkeypatch.setattr(tray, "_show_context_menu", _fake_show_context_menu)
    monkeypatch.setattr("neurabreak.ui.tray.QTimer.singleShot", _fake_single_shot)

    reason = QSystemTrayIcon.ActivationReason.Context
    tray._on_activated(reason)
    tray._on_activated(reason)

    assert scheduled_calls == [_CONTEXT_MENU_POPUP_DELAY_MS]


def test_quit_action_uses_explicit_quit_callback(qapp, tmp_path):
    called: list[bool] = []

    tray = _make_tray(
        qapp,
        tmp_path,
        on_quit_requested=lambda: called.append(True),
    )
    tray._confirm_quit = lambda: True  # type: ignore[method-assign]

    tray._request_quit()
    assert called == [True]


def test_quit_action_falls_back_to_app_quit(qapp, tmp_path, monkeypatch):
    tray = _make_tray(qapp, tmp_path)
    called: list[bool] = []

    tray._confirm_quit = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(tray.app, "quit", lambda: called.append(True))
    tray._request_quit()

    assert called == [True]


def test_quit_action_cancelled_does_not_quit(qapp, tmp_path, monkeypatch):
    tray = _make_tray(qapp, tmp_path)
    called: list[bool] = []

    tray._confirm_quit = lambda: False  # type: ignore[method-assign]
    monkeypatch.setattr(tray.app, "quit", lambda: called.append(True))
    tray._request_quit()

    assert called == []
