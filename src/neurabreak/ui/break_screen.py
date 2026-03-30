"""Break screen overlay — shown when it's time to take a break.

A semi-transparent full-screen-ish QDialog with:
  - 20-20-20 countdown (look 20 feet away for 20 seconds)
  - Simple stretching and movement instructions
  - [Take Break] and [Snooze N min] buttons

This is Level 3 in the escalation ladder. The app.py shows it by calling
`screen.show_break()` from the Qt main thread.

The same widget also exposes `show_eye_break()` — a compact, auto-closing
20-second countdown overlay for the periodic 20-20-20 eye-rest reminder.
"""

from __future__ import annotations

from typing import Callable

import structlog
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from neurabreak.ui.branding import apply_window_icon, logo_pixmap

log = structlog.get_logger()

# Cycling exercise instructions shown during the break
_EXERCISES = [
    "Look at something 20 feet away for 20 seconds — let your eyes relax.",
    "Stand up and take a short walk — even 2 minutes of movement boosts focus and blood flow.",
    "Roll your shoulders back three times. Feel the tension release.",
    "Gently tilt your head left, then right. Hold for 5 seconds each side.",
    "Stretch your arms overhead and take three deep breaths — hydrate while you're at it.",
    "Rotate your wrists 5 times each direction to reduce wrist strain.",
    "Walk to the kitchen, step outside, or do 10 slow squats — get your circulation going.",
]


class BreakScreen(QDialog):
    """Full-screen break overlay (Level 3 escalation).

    Semi-transparent, always on top. User can take the break or snooze.
    Also handles the 20-20-20 eye break (compact auto-close overlay)
    via show_eye_break().
    """

    def __init__(self, config: object, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self._state_machine: object | None = None
        self._notif_manager: object | None = None
        self._countdown_sec = 20
        self._exercise_index = 0
        self._mode = "break"  # 'break' | 'eye' | 'walk'

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

        self._on_break_taken_cb: Callable | None = None
        self._on_snoozed_cb: Callable[..., None] | None = None

        self._build_ui()

    # Public API

    def show_break(
        self,
        state_machine: object | None = None,
        notif_manager: object | None = None,
    ) -> None:
        """Display the break screen and start the 20-second countdown."""
        self._state_machine = state_machine
        self._notif_manager = notif_manager
        self._mode = "break"
        self._countdown_sec = 20
        self._countdown_label.setText("20")
        self._exercise_index = 0
        self._exercise_label.setText(_EXERCISES[0])
        self._title_label.setText("🧘 Time for a Break")
        self._title_label.setStyleSheet("color: #e2e8f0; background: transparent;")
        self._countdown_label.setStyleSheet("color: #60a5fa;")
        self._take_btn.setText("✓  I'm taking my break")
        self._take_btn.show()
        self._snooze_btn.show()
        self.setMinimumSize(540, 340)
        self._timer.start()
        self.show()
        self.raise_()
        self.activateWindow()

    def show_eye_break(self, duration_sec: int = 20) -> None:
        """Show the compact 20-20-20 eye-rest overlay.

        Displays a focused countdown and auto-closes when it reaches zero.
        Does not interact with the state machine — the user just looks away
        and the overlay dismisses itself.
        """
        # Don't interrupt an active break overlay
        if self._mode == "break" and self.isVisible():
            return
        self._state_machine = None
        self._notif_manager = None
        self._mode = "eye"
        self._countdown_sec = duration_sec
        self._countdown_label.setText(str(duration_sec))
        self._exercise_label.setText(
            "Look at something 20 feet (6 m) away and let your eyes relax completely."
        )
        self._title_label.setText("👁️ 20-20-20 Eye Break")
        self._title_label.setStyleSheet("color: #86efac; background: transparent;")  # green tint
        self._countdown_label.setStyleSheet("color: #4ade80;")
        self._take_btn.setText("✓  Done")
        self._snooze_btn.hide()
        self.setMinimumSize(480, 260)
        self._timer.start()
        self.show()
        self.raise_()
        self.activateWindow()


    def on_break_taken(self, callback: Callable) -> None:
        self._on_break_taken_cb = callback

    def on_snoozed(self, callback: Callable[..., None]) -> None:
        self._on_snoozed_cb = callback

    # Internal

    def _build_ui(self) -> None:
        self.setWindowTitle("NeuraBreak — Break Time")
        apply_window_icon(self)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
        )
        # Prevent this dialog from triggering app-quit when closed
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setMinimumSize(540, 340)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #0f172a;
                border-radius: 16px;
            }
            QLabel {
                color: #e2e8f0;
                background: transparent;
            }
            QPushButton {
                background-color: #1e40af;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton#snooze {
                background-color: #374151;
            }
            QPushButton#snooze:hover {
                background-color: #4b5563;
            }
            """
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(48, 40, 48, 40)
        outer.setSpacing(20)

        logo = logo_pixmap(72)
        if not logo.isNull():
            logo_lbl = QLabel()
            logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo_lbl.setPixmap(logo)
            outer.addWidget(logo_lbl)

        # Title
        self._title_label = QLabel("\U0001f9d8 Time for a Break")
        self._title_label.setFont(QFont("", 20, QFont.Weight.Bold))
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._title_label)

        # Exercise instruction (updates every 5 seconds of countdown)
        self._exercise_label = QLabel(_EXERCISES[0])
        self._exercise_label.setWordWrap(True)
        self._exercise_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._exercise_label.setFont(QFont("", 13))
        outer.addWidget(self._exercise_label)

        # Big countdown number
        self._countdown_label = QLabel("20")
        self._countdown_label.setFont(QFont("", 56, QFont.Weight.Bold))
        self._countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._countdown_label.setStyleSheet("color: #60a5fa;")
        outer.addWidget(self._countdown_label)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(16)

        self._take_btn = QPushButton("\u2713  I'm taking my break")
        self._take_btn.setObjectName("take")
        self._take_btn.clicked.connect(self._on_take_break)
        btn_row.addWidget(self._take_btn)

        snooze_min = self._snooze_minutes()
        self._snooze_btn = QPushButton(f"Snooze {snooze_min} min")
        self._snooze_btn.setObjectName("snooze")
        self._snooze_btn.clicked.connect(lambda: self._on_snooze(snooze_min))
        self._snooze_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        btn_row.addWidget(self._snooze_btn)

        outer.addLayout(btn_row)

    def _snooze_minutes(self) -> int:
        try:
            return self.config.escalation.snooze_options[1]  # type: ignore
        except (AttributeError, IndexError):
            return 10

    def _tick(self) -> None:
        self._countdown_sec -= 1
        if self._mode == "eye":
            # Show remaining seconds; auto-close at zero
            self._countdown_label.setText(str(max(0, self._countdown_sec)))
            if self._countdown_sec <= 0:
                self._timer.stop()
                self._countdown_label.setText("\U0001f31f")
                QTimer.singleShot(800, self.hide)  # brief flash of star then close
            return

        # Cycle exercise text every 5 seconds
        if self._countdown_sec > 0 and self._countdown_sec % 5 == 0:
            self._exercise_index = (self._exercise_index + 1) % len(_EXERCISES)
            self._exercise_label.setText(_EXERCISES[self._exercise_index])

        if self._countdown_sec <= 0:
            self._timer.stop()
            self._countdown_label.setText("🌟")

    def _on_take_break(self) -> None:
        self._timer.stop()

        if self._state_machine is not None:
            try:
                self._state_machine.start_break()  # type: ignore
            except Exception:
                pass
        if self._on_break_taken_cb:
            self._on_break_taken_cb()
        self.hide()

    def _on_snooze(self, minutes: int) -> None:
        self._timer.stop()
        
        if self._notif_manager is not None:
            try:
                self._notif_manager.snooze(minutes)  # type: ignore
            except Exception:
                pass
        if self._on_snoozed_cb:
            self._on_snoozed_cb(minutes)
        self.hide()
