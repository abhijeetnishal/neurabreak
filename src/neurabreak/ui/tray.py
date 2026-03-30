"""System tray icon — the main user-visible presence of NeuraBreak.

The icon colour reflects the current monitoring state:
  grey   — idle, no one detected at desk
  green  — monitoring, good posture
  yellow — posture warning
  red    — break due or bad posture alert
"""

from __future__ import annotations

from time import monotonic
from typing import TYPE_CHECKING, Callable

import structlog
from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from neurabreak.core.config import ConfigManager
from neurabreak.core.events import EventType, bus
from neurabreak.core.state_machine import AppState
from neurabreak.ui.branding import apply_window_icon

if TYPE_CHECKING:
    from neurabreak.data.journal import HealthJournalService
    from neurabreak.notifications.audio import AudioManager

log = structlog.get_logger()

_CONTEXT_MENU_POPUP_DELAY_MS = 120
_CONTEXT_MENU_DEBOUNCE_SEC = 0.25


class _TrayBridge(QObject):
    """Tiny QObject that lives on the main thread.

    Background threads (inference, escalation monitor) emit `state_change`;
    Qt's QueuedConnection delivers it to `NeuraBreakTray.set_state` on the
    main thread — no QTimer.singleShot needed.
    """
    state_change: Signal = Signal(str, str)  # (colour, status_text)

# Colour scheme for tray icon dots
_COLOURS: dict[str, QColor] = {
    "grey":   QColor("#95a5a6"),
    "green":  QColor("#2ecc71"),
    "yellow": QColor("#f39c12"),
    "red":    QColor("#e74c3c"),
    "blue":   QColor("#3498db"),
}


def _make_icon(colour_name: str = "grey") -> QIcon:
    """Draw a simple filled circle as the tray icon."""
    size = 64
    margin = 6

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(_COLOURS.get(colour_name, _COLOURS["grey"]))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(margin, margin, size - 2 * margin, size - 2 * margin)
    painter.end()

    return QIcon(pixmap)


class NeuraBreakTray:
    """System tray icon with a right-click context menu."""

    def __init__(
        self,
        config_manager: ConfigManager,
        app: QApplication,
        journal: HealthJournalService | None = None,
        audio_manager: AudioManager | None = None,
        on_quit_requested: Callable[[], None] | None = None,
    ) -> None:
        self.config_manager = config_manager
        self.app = app
        self._journal = journal
        self._audio = audio_manager
        self._on_quit_requested = on_quit_requested
        self._paused = False

        # Lazy-constructed windows
        self._dashboard: object | None = None
        self._settings_win: object | None = None
        self._preview_win: object | None = None
        self._about_dlg: object | None = None
        self._quit_confirm_dlg: object | None = None

        self._icon = QSystemTrayIcon()
        self._icon.setIcon(_make_icon("grey"))
        self._icon.setToolTip("NeuraBreak — idle")

        # Bridge for cross-thread → main-thread state updates (QueuedConnection)
        self._bridge = _TrayBridge()
        self._bridge.state_change.connect(self.set_state)

        self._menu = self._build_menu()
        self._menu_popup_pending = False
        self._last_context_popup_at = 0.0
        # Do NOT use setContextMenu — on Windows, Qt's internal SetForegroundWindow /
        # TrackPopupMenu mechanism injects spurious WM_QUIT into the message queue
        self._icon.activated.connect(self._on_activated)

        self._subscribe_to_events()

    def _subscribe_to_events(self) -> None:
        """Wire up bus events so the tray icon reacts to state changes.

        Events are published from the inference thread. We emit a Signal on
        `_bridge` (owned by main thread) so Qt automatically uses
        QueuedConnection — no QTimer.singleShot required.
        """
        def _emit(colour: str, text: str) -> None:
            """Only color the icon when color-coding is enabled."""
            if not self.config_manager.config.ui.tray_icon_color_coding:
                colour = "grey"
            self._bridge.state_change.emit(colour, text)

        bus.subscribe(
            EventType.SESSION_STARTED,
            lambda _: _emit("green", "monitoring"),
        )
        bus.subscribe(
            EventType.MODEL_LOADING,
            lambda _: _emit("blue", "loading model…"),
        )
        bus.subscribe(
            EventType.MODEL_LOADED,
            lambda _: _emit("grey", "idle — waiting for presence"),
        )
        bus.subscribe(
            EventType.SESSION_PAUSED,
            lambda _: _emit("grey", "smart pause active"),
        )
        bus.subscribe(
            EventType.SESSION_RESUMED,
            lambda _: _emit("green", "monitoring"),
        )
        bus.subscribe(
            EventType.POSTURE_ALERT,
            lambda e: _emit(
                "yellow",
                "posture alert: " + e.data.get("posture", "").replace("posture_", "").replace("_", " "),
            ),
        )
        bus.subscribe(
            EventType.POSTURE_RESTORED,
            lambda _: _emit("green", "monitoring"),
        )
        bus.subscribe(
            EventType.BREAK_DUE,
            lambda _: _emit("red", "break time!"),
        )
        bus.subscribe(
            EventType.BREAK_STARTED,
            lambda _: _emit("blue", "on break"),
        )
        bus.subscribe(
            EventType.BREAK_ENDED,
            lambda _: _emit("green", "monitoring"),
        )
        bus.subscribe(
            EventType.UPDATE_AVAILABLE,
            lambda e: QTimer.singleShot(0, lambda: self._on_update_available(e.data)),
        )

    def _on_update_available(self, data: dict) -> None:
        """Show a tray balloon pointing to the GitHub releases page."""
        version = data.get("version", "")
        url = data.get("url", "https://github.com/abhijeetnishal/neurabreak/releases/latest")

        # Add a "Download update" entry to the menu (lazy — only when update found)
        if not hasattr(self, "_update_action"):
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl

            self._update_action = self._menu.addAction(f"Update available: v{version}")
            self._update_action.triggered.connect(
                lambda: QDesktopServices.openUrl(QUrl(url))
            )
            # Insert it near the top, just after the status label
            self._menu.insertAction(self._pause_action, self._update_action)

        self._icon.showMessage(
            "NeuraBreak — Update available",
            f"Version {version} is available on GitHub.\n"
            "Click the tray icon menu to download.",
            QSystemTrayIcon.MessageIcon.Information,
            8000,
        )

    def _build_menu(self) -> QMenu:
        menu = QMenu()

        # Non-interactive status label at the top
        self._status_action = menu.addAction("Status: Idle")
        self._status_action.setEnabled(False)
        menu.addSeparator()

        # Pause / Resume toggle
        self._pause_action = menu.addAction("Pause Monitoring")
        self._pause_action.triggered.connect(self._toggle_pause)

        menu.addSeparator()

        # Store every action reference on self — local-variable QAction wrappers
        self._dashboard_action = menu.addAction("Dashboard")
        self._dashboard_action.triggered.connect(self._show_dashboard)

        self._settings_action = menu.addAction("Settings")
        self._settings_action.triggered.connect(self._show_settings)

        self._preview_action = menu.addAction("Show Camera Preview")
        self._preview_action.triggered.connect(self._toggle_preview)

        menu.addSeparator()

        self._about_action = menu.addAction("About NeuraBreak…")
        self._about_action.triggered.connect(self._show_about)

        menu.addSeparator()

        self._quit_action = menu.addAction("Quit")
        self._quit_action.triggered.connect(self._request_quit)

        return menu

    def show(self) -> None:
        self._icon.show()

    def hide(self) -> None:
        self._icon.hide()

    @Slot(str, str)
    def set_state(self, colour: str, status_text: str) -> None:
        """Update the icon colour and tooltip.

        Safe to call from any thread via the _bridge signal (QueuedConnection)
        or directly from the Qt main thread.
        """
        self._icon.setIcon(_make_icon(colour))
        self._icon.setToolTip(f"NeuraBreak — {status_text}")
        self._status_action.setText(f"Status: {status_text.capitalize()}")

    #  Action handlers

    def _show_dashboard(self) -> None:
        if self._journal is None:
            QMessageBox.information(
                None,
                "Dashboard",
                "The health journal is not available.\n"
                "Make sure sqlalchemy is installed (`uv sync --extra data`).",
            )
            return
        if self._dashboard is None:
            from neurabreak.ui.dashboard import DashboardWindow
            self._dashboard = DashboardWindow(self._journal)
        self._dashboard.show()

    def _show_settings(self) -> None:
        from neurabreak.ui.settings import SettingsWindow
        # Store on self (not a local var) so PySide6/Qt don't GC the C++ QDialog
        self._settings_win = SettingsWindow(
            config_manager=self.config_manager,
            audio_manager=self._audio,
        )
        self._settings_win.exec()

    def _toggle_preview(self) -> None:
        if self._preview_win is None:
            from neurabreak.ui.preview import PreviewWindow
            self._preview_win = PreviewWindow()
            # Keep the menu label in sync
            self._preview_win.closed.connect(
                lambda: self._preview_action.setText("Show Camera Preview")
            )

        if self._preview_win.isVisible():
            self._preview_win.hide()
            self._preview_action.setText("Show Camera Preview")
        else:
            self._preview_win.show()
            self._preview_action.setText("Hide Camera Preview")

    def get_preview_window(self) -> object | None:
        """Return the live PreviewWindow if it has been created, else None."""
        return self._preview_win

    def _toggle_pause(self) -> None:
        from neurabreak.core.events import Event

        self._paused = not self._paused
        if self._paused:
            self._pause_action.setText("Resume Monitoring")
            self.set_state("grey", "paused")
            # Tell the state machine to stop processing (it will move to IDLE)
            bus.publish(Event(EventType.MONITORING_PAUSED_MANUAL))
        else:
            self._pause_action.setText("Pause Monitoring")
            self.set_state("grey", "waiting for presence…")
            # Tell the state machine it may now re-enter MONITORING on presence
            bus.publish(Event(EventType.MONITORING_RESUMED_MANUAL))

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Context:
            # Queue menu popup to avoid re-entrant right-click storms on Windows.
            self._request_context_menu_popup()
        elif reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._icon.showMessage(
                "NeuraBreak",
                "Right-click the icon for options.",
                QSystemTrayIcon.MessageIcon.Information,
                3000,
            )

    def _request_context_menu_popup(self) -> None:
        now = monotonic()

        if self._menu_popup_pending or self._menu.isVisible():
            return
        if (now - self._last_context_popup_at) < _CONTEXT_MENU_DEBOUNCE_SEC:
            return

        self._menu_popup_pending = True
        self._last_context_popup_at = now
        QTimer.singleShot(_CONTEXT_MENU_POPUP_DELAY_MS, self._show_context_menu)

    def _show_context_menu(self) -> None:
        self._menu_popup_pending = False
        if self._menu.isVisible():
            return

        from PySide6.QtGui import QCursor
        self._menu.popup(QCursor.pos())

    def _request_quit(self) -> None:
        if not self._confirm_quit():
            return

        if self._on_quit_requested is not None:
            self._on_quit_requested()
            return
        self.app.quit()

    def _confirm_quit(self) -> bool:
        self._quit_confirm_dlg = QMessageBox()
        apply_window_icon(self._quit_confirm_dlg)
        self._quit_confirm_dlg.setWindowTitle("Quit NeuraBreak?")
        self._quit_confirm_dlg.setText("Do you want to quit NeuraBreak?")
        self._quit_confirm_dlg.setInformativeText(
            "Monitoring and reminders will stop until you open the app again."
        )
        self._quit_confirm_dlg.setIcon(QMessageBox.Icon.Question)
        self._quit_confirm_dlg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        self._quit_confirm_dlg.setDefaultButton(QMessageBox.StandardButton.No)

        return (
            self._quit_confirm_dlg.exec()
            == QMessageBox.StandardButton.Yes
        )

    def _show_about(self) -> None:
        # Store on self so the QMessageBox C++ object outlives its exec() call;
        self._about_dlg = QMessageBox()
        apply_window_icon(self._about_dlg)
        self._about_dlg.setWindowTitle("About NeuraBreak")
        self._about_dlg.setText(
            "<b>NeuraBreak v0.1.0</b><br><br>"
            "AI-powered break &amp; posture guardian.<br>"
            "100% local — no cloud, no tracking, no account.<br><br>"
        )
        self._about_dlg.setIcon(QMessageBox.Icon.Information)
        self._about_dlg.exec()
