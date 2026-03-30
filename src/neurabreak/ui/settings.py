"""Settings window — exposes all config options in a tabbed GUI.

Tabs:
  General       — break interval, duration, smart pause
  Notifications — escalation delays, mandatory break, snooze options
  Audio         — per-level sound picker, master volume, live test button
  About         — version, credits, open-source info
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from neurabreak.core.config import AppConfig, ConfigManager
    from neurabreak.notifications.audio import AudioManager

from neurabreak.ui.branding import apply_window_icon, logo_pixmap
from neurabreak.core.events import Event, EventType, bus

log = structlog.get_logger()

_STYLE = """
QDialog, QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-size: 12px;
}
QTabWidget::pane {
    border: 1px solid #3a3a3a;
    border-radius: 4px;
}
QTabBar::tab {
    background: #2d2d2d;
    border: 1px solid #3a3a3a;
    padding: 7px 18px;
    color: #aaaaaa;
}
QTabBar::tab:selected {
    background: #3a3a3a;
    color: #ffffff;
}
QTabBar::tab:hover { background: #333333; }
QGroupBox {
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 8px;
    color: #aaaaaa;
    font-size: 11px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton {
    background-color: #2d2d2d;
    border: 1px solid #444;
    border-radius: 5px;
    padding: 5px 14px;
    color: #e0e0e0;
}
QPushButton:hover { background-color: #3a3a3a; }
QPushButton:pressed { background-color: #222; }
QSpinBox, QComboBox {
    background-color: #2d2d2d;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 3px 24px 3px 8px;
    color: #e0e0e0;
    min-width: 80px;
}
QSpinBox:focus, QComboBox:focus { border-color: #3498db; }
QSpinBox::up-button, QSpinBox::down-button {
    subcontrol-origin: border;
    width: 18px;
    background-color: #3a3a3a;
    border-left: 1px solid #555;
}
QSpinBox::up-button {
    subcontrol-position: top right;
    border-top-right-radius: 3px;
    border-bottom: 1px solid #555;
}
QSpinBox::down-button {
    subcontrol-position: bottom right;
    border-bottom-right-radius: 3px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background-color: #4a4a4a; }
QSpinBox::up-button:pressed, QSpinBox::down-button:pressed { background-color: #222; }
QSpinBox::up-arrow {
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #c8c8c8;
}
QSpinBox::down-arrow {
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #c8c8c8;
}
QSpinBox::up-arrow:disabled { border-bottom-color: #555; }
QSpinBox::down-arrow:disabled { border-top-color: #555; }
QSlider::groove:horizontal {
    height: 4px;
    background: #444;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #3498db;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #3498db; border-radius: 2px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #555; border-radius: 3px; }
QCheckBox::indicator:checked { background: #3498db; border-color: #3498db; }
QLabel { color: #e0e0e0; }
QLabel[role="hint"] { color: #888; font-size: 10px; }
QDialogButtonBox QPushButton[text="Save"] { background-color: #2980b9; border-color: #2980b9; color: #fff; }
QDialogButtonBox QPushButton[text="Save"]:hover { background-color: #3498db; }
"""


def _hint(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "hint")
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: #888888; font-size: 10px;")
    return lbl


class SettingsWindow(QDialog):
    """Tabbed settings dialog.

    All changes are held in-memory until the user clicks Save.
    Clicking Cancel discards everything.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        audio_manager: AudioManager | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.config_manager = config_manager
        self._audio = audio_manager
        self._config = config_manager.config.model_copy(deep=True)

        self.setWindowTitle("NeuraBreak — Settings")
        apply_window_icon(self)
        self.setMinimumSize(520, 400)
        self.setStyleSheet(_STYLE)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # Cap the dialog to 85 % of the available screen height
        screen = QApplication.primaryScreen()
        if screen:
            avail_h = screen.availableGeometry().height()
            self.resize(580, min(640, int(avail_h * 0.85)))
        else:
            self.resize(580, 640)

        self._build_ui()

    #  UI construction

    @staticmethod
    def _scrollable(widget: QWidget) -> QScrollArea:
        """Wrap a tab widget in a frameless scroll area so tall tabs don't
        push the dialog's title bar off-screen."""
        area = QScrollArea()
        area.setWidget(widget)
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return area

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(10)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        self._tabs.addTab(self._scrollable(self._tab_general()),       "General")
        self._tabs.addTab(self._scrollable(self._tab_notifications()), "Notifications")
        self._tabs.addTab(self._scrollable(self._tab_audio()),         "Audio")
        self._tabs.addTab(self._scrollable(self._tab_about()),         "About")

        # Save / Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # Tab: General

    def _tab_general(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        group = QGroupBox("Break Timing")
        form = QFormLayout(group)
        form.setSpacing(10)

        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(5, 180)
        self._interval_spin.setSuffix("  min")
        self._interval_spin.setValue(self._config.breaks.interval_min)
        form.addRow("Break every:", self._interval_spin)
        form.addRow("", _hint("How long between break reminders (5–180 min)."))

        self._duration_spin = QSpinBox()
        self._duration_spin.setRange(1, 30)
        self._duration_spin.setSuffix("  min")
        self._duration_spin.setValue(self._config.breaks.duration_min)
        form.addRow("Break duration:", self._duration_spin)

        self._posture_alert_spin = QSpinBox()
        self._posture_alert_spin.setRange(10, 300)
        self._posture_alert_spin.setSuffix("  sec")
        self._posture_alert_spin.setValue(self._config.breaks.posture_alert_sec)
        form.addRow("Posture alert after:", self._posture_alert_spin)
        form.addRow("", _hint("Alert when bad posture has lasted this many seconds."))

        self._smart_pause_spin = QSpinBox()
        self._smart_pause_spin.setRange(5, 300)
        self._smart_pause_spin.setSuffix("  sec")
        self._smart_pause_spin.setValue(self._config.breaks.smart_pause_sec)
        form.addRow("Smart pause after:", self._smart_pause_spin)
        form.addRow("", _hint("Pause the session timer when no one is at the desk for this long."))

        self._eye_interval_spin = QSpinBox()
        self._eye_interval_spin.setRange(5, 60)
        self._eye_interval_spin.setSuffix("  min")
        self._eye_interval_spin.setValue(self._config.breaks.eye_break_interval_min)
        form.addRow("Eye break every:", self._eye_interval_spin)
        form.addRow("", _hint("20-20-20 rule: look 20 ft away for 20 sec every N minutes."))

        self._eye_duration_spin = QSpinBox()
        self._eye_duration_spin.setRange(5, 120)
        self._eye_duration_spin.setSuffix("  sec")
        self._eye_duration_spin.setValue(self._config.breaks.eye_break_duration_sec)
        form.addRow("Eye break duration:", self._eye_duration_spin)

        layout.addWidget(group)

        ui_group = QGroupBox("Interface")
        ui_form = QFormLayout(ui_group)
        ui_form.setSpacing(10)

        self._start_minimized_cb = QCheckBox()
        self._start_minimized_cb.setChecked(self._config.ui.start_minimized)
        ui_form.addRow("Start minimized to tray:", self._start_minimized_cb)

        from neurabreak.core import startup as _startup
        self._startup_cb: QCheckBox | None = None
        if _startup.is_windows():
            self._startup_cb = QCheckBox()
            self._startup_cb.setChecked(_startup.is_startup_enabled())
            ui_form.addRow("Start with Windows:", self._startup_cb)
            ui_form.addRow("", _hint("Launch NeuraBreak automatically at login (no admin rights needed)."))

        self._tray_colors_cb = QCheckBox()
        self._tray_colors_cb.setChecked(self._config.ui.tray_icon_color_coding)
        ui_form.addRow("Tray icon color coding:", self._tray_colors_cb)

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["system", "dark", "light"])
        self._theme_combo.setCurrentText(self._config.ui.theme)
        ui_form.addRow("Theme:", self._theme_combo)

        layout.addWidget(ui_group)
        layout.addStretch()
        return w

    # Tab: Notifications

    def _tab_notifications(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        group = QGroupBox("Escalation")
        form = QFormLayout(group)
        form.setSpacing(10)

        self._lvl2_spin = QSpinBox()
        self._lvl2_spin.setRange(1, 30)
        self._lvl2_spin.setSuffix("  min")
        self._lvl2_spin.setValue(self._config.escalation.level_2_delay_min)
        form.addRow("Level 2 (balloon) after:", self._lvl2_spin)

        self._lvl3_spin = QSpinBox()
        self._lvl3_spin.setRange(1, 30)
        self._lvl3_spin.setSuffix("  min")
        self._lvl3_spin.setValue(self._config.escalation.level_3_delay_min)
        form.addRow("Level 3 (overlay) after:", self._lvl3_spin)

        self._mandatory_cb = QCheckBox()
        self._mandatory_cb.setChecked(self._config.escalation.mandatory_break)
        form.addRow("Mandatory break:", self._mandatory_cb)
        form.addRow("", _hint("When on, the overlay cannot be dismissed until the break is taken."))

        self._focus_mode_cb = QCheckBox()
        self._focus_mode_cb.setChecked(self._config.escalation.respect_focus_mode)
        form.addRow("Respect focus mode:", self._focus_mode_cb)
        form.addRow("", _hint("Holds notifications when OS Focus Assist / Do Not Disturb is active."))

        layout.addWidget(group)

        dark_group = QGroupBox("Dark Hours")
        dark_form = QFormLayout(dark_group)
        dark_form.setSpacing(10)

        self._dark_enabled_cb = QCheckBox()
        self._dark_enabled_cb.setChecked(self._config.dark_hours.enabled)
        dark_form.addRow("Enable dark hours:", self._dark_enabled_cb)

        self._dark_start_spin = QSpinBox()
        self._dark_start_spin.setRange(0, 23)
        self._dark_start_spin.setSuffix(" :00")
        self._dark_start_spin.setValue(self._config.dark_hours.start_hour)
        dark_form.addRow("Start hour:", self._dark_start_spin)

        self._dark_end_spin = QSpinBox()
        self._dark_end_spin.setRange(0, 23)
        self._dark_end_spin.setSuffix(" :00")
        self._dark_end_spin.setValue(self._config.dark_hours.end_hour)
        dark_form.addRow("End hour:", self._dark_end_spin)

        self._dark_volume_cb = QCheckBox()
        self._dark_volume_cb.setChecked(self._config.dark_hours.reduce_volume)
        dark_form.addRow("Reduce volume:", self._dark_volume_cb)

        layout.addWidget(dark_group)
        layout.addStretch()
        return w

    # Tab: Audio

    def _tab_audio(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        master_group = QGroupBox("Master Volume")
        master_form = QFormLayout(master_group)

        self._audio_enabled_cb = QCheckBox()
        self._audio_enabled_cb.setChecked(self._config.audio.enabled)
        master_form.addRow("Audio enabled:", self._audio_enabled_cb)

        vol_row = QHBoxLayout()
        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(self._config.audio.volume)
        self._volume_lbl = QLabel(f"{self._config.audio.volume}%")
        self._volume_lbl.setFixedWidth(36)
        self._volume_slider.valueChanged.connect(lambda v: self._volume_lbl.setText(f"{v}%"))
        vol_row.addWidget(self._volume_slider)
        vol_row.addWidget(self._volume_lbl)
        master_form.addRow("Volume:", vol_row)

        layout.addWidget(master_group)

        sounds_group = QGroupBox("Sounds per Level")
        sounds_layout = QVBoxLayout(sounds_group)
        sounds_layout.setSpacing(10)

        self._sound_rows: dict[str, QLabel] = {}
        sound_levels = [
            ("level_1", "Level 1 (gentle nudge):",  self._config.audio.sounds.level_1),
            ("level_2", "Level 2 (balloon):",        self._config.audio.sounds.level_2),
            ("level_3", "Level 3 (overlay):",        self._config.audio.sounds.level_3),
            ("break_end", "Break end chime:",        self._config.audio.sounds.break_end),
        ]
        for key, label_text, current_val in sound_levels:
            row = QHBoxLayout()
            row.setSpacing(8)
            label = QLabel(label_text)
            label.setFixedWidth(180)
            val_lbl = QLabel(current_val)
            val_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
            val_lbl.setMinimumWidth(120)
            self._sound_rows[key] = val_lbl

            pick_btn = QPushButton("…")
            pick_btn.setFixedWidth(28)
            pick_btn.setToolTip("Browse for a sound file")
            pick_btn.clicked.connect(lambda checked, k=key, l=val_lbl: self._pick_sound(k, l))

            test_btn = QPushButton("▶")
            test_btn.setFixedWidth(28)
            test_btn.setToolTip("Preview this sound")
            test_btn.clicked.connect(lambda checked, k=key: self._test_sound(k))

            row.addWidget(label)
            row.addWidget(val_lbl, stretch=1)
            row.addWidget(pick_btn)
            row.addWidget(test_btn)
            sounds_layout.addLayout(row)

        layout.addWidget(sounds_group)
        layout.addStretch()
        return w

    def _pick_sound(self, key: str, label: QLabel) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Choose sound for {key}",
            str(Path.home()),
            "Audio files (*.wav *.mp3 *.ogg)",
        )
        if path:
            label.setText(path)
            label.setToolTip(path)
            setattr(self._config.audio.sounds, key, path)

    def _test_sound(self, key: str) -> None:
        if not self._audio:
            QMessageBox.information(self, "Audio", "Audio manager not available.")
            return
        sound_val = getattr(self._config.audio.sounds, key, "")
        if sound_val.startswith("builtin:"):
            name = sound_val.removeprefix("builtin:")
            self._audio.play_builtin(name)
        elif sound_val:
            self._audio.play(key, Path(sound_val))

    # Tab: About

    def _tab_about(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)

        logo = logo_pixmap(84)
        if not logo.isNull():
            logo_lbl = QLabel()
            logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo_lbl.setPixmap(logo)
            layout.addWidget(logo_lbl)

        title = QLabel("NeuraBreak")
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        tagline = QLabel("Your personal health guardian for long desk sessions")
        tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tagline.setStyleSheet("color: #aaa; font-size: 12px; font-style: italic;")
        tagline.setWordWrap(True)
        layout.addWidget(tagline)

        version = QLabel("Version 0.1.0")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(version)

        sep = QLabel("─" * 48)
        sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sep.setStyleSheet("color: #3a3a3a; margin: 4px 0;")
        layout.addWidget(sep)

        features_html = (
            "<table style='width:100%; border-spacing: 0 6px; color:#c8c8c8; font-size:11px;'>"
            "<tr><td style='width:22px'>👁️</td><td><b>20-20-20 Eye Breaks</b> — auto-close overlay every 20 min; look 20 ft away for 20 sec</td></tr>"
            "<tr><td>🧠</td><td><b>AI Posture Detection</b> — YOLO-based (nano/small/medium); face &amp; posture class detection</td></tr>"
            "<tr><td>🕐</td><td><b>Smart Break Scheduling</b> — configurable session timer with full-screen countdown overlay</td></tr>"
            "<tr><td>🚶</td><td><b>Smart Pause</b> — session timer stops the moment you step away; resumes when you return</td></tr>"
            "<tr><td>🔔</td><td><b>Escalating Notifications</b> — four levels: gentle → balloon → persistent → overlay</td></tr>"
            "<tr><td>📊</td><td><b>Health Dashboard</b> — SQLite journal, posture charts, break compliance, active-today counter</td></tr>"
            "<tr><td>🔒</td><td><b>100 % Local</b> — no cloud, no account, no data leaves your machine. Ever.</td></tr>"
            "<tr><td>🔄</td><td><b>Auto-update Checker</b> — background GitHub Releases check, never blocks the UI</td></tr>"
            "</table>"
        )
        features_lbl = QLabel(features_html)
        features_lbl.setTextFormat(Qt.TextFormat.RichText)
        features_lbl.setWordWrap(True)
        features_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        features_lbl.setStyleSheet("background: transparent;")
        layout.addWidget(features_lbl)

        sep2 = QLabel("─" * 48)
        sep2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sep2.setStyleSheet("color: #3a3a3a; margin: 4px 0;")
        layout.addWidget(sep2)

        footer = QLabel(
            'MIT License — <a style="color:#3498db;" href="https://github.com/abhijeetnishal/neurabreak">'
            "github.com/abhijeetnishal/neurabreak</a>"
        )
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setTextFormat(Qt.TextFormat.RichText)
        footer.setOpenExternalLinks(True)
        footer.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(footer)

        layout.addStretch()
        return w

    #  Save logic

    def _on_save(self) -> None:
        """Write changed values back to config and persist to disk."""
        cfg = self._config

        # General
        cfg.breaks.interval_min = self._interval_spin.value()
        cfg.breaks.duration_min = self._duration_spin.value()
        cfg.breaks.posture_alert_sec = self._posture_alert_spin.value()
        cfg.breaks.smart_pause_sec = self._smart_pause_spin.value()
        cfg.breaks.eye_break_interval_min = self._eye_interval_spin.value()
        cfg.breaks.eye_break_duration_sec = self._eye_duration_spin.value()
        cfg.ui.start_minimized = self._start_minimized_cb.isChecked()
        cfg.ui.tray_icon_color_coding = self._tray_colors_cb.isChecked()
        cfg.ui.theme = self._theme_combo.currentText()

        # Windows autostart — write registry immediately on save
        if self._startup_cb is not None:
            from neurabreak.core.startup import set_startup
            set_startup(self._startup_cb.isChecked())

        # Notifications
        cfg.escalation.level_2_delay_min = self._lvl2_spin.value()
        cfg.escalation.level_3_delay_min = self._lvl3_spin.value()
        cfg.escalation.mandatory_break = self._mandatory_cb.isChecked()
        cfg.escalation.respect_focus_mode = self._focus_mode_cb.isChecked()
        cfg.dark_hours.enabled = self._dark_enabled_cb.isChecked()
        cfg.dark_hours.start_hour = self._dark_start_spin.value()
        cfg.dark_hours.end_hour = self._dark_end_spin.value()
        cfg.dark_hours.reduce_volume = self._dark_volume_cb.isChecked()

        # Audio
        cfg.audio.enabled = self._audio_enabled_cb.isChecked()
        cfg.audio.volume = self._volume_slider.value()

        try:
            self._write_config_to_disk(cfg)
            self.config_manager._config = cfg
            bus.publish(Event(EventType.CONFIG_CHANGED, {"config": cfg}))
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _write_config_to_disk(self, cfg: AppConfig) -> None:
        path = self.config_manager._config_path
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build TOML string from model — simple manual serialisation so we don't need an extra library
        lines = [
            "# NeuraBreak Configuration",
            "# Saved by the Settings window.\n",
            "[breaks]",
            f'interval_min           = {cfg.breaks.interval_min}',
            f'duration_min           = {cfg.breaks.duration_min}',
            f'posture_alert_sec      = {cfg.breaks.posture_alert_sec}',
            f'smart_pause_sec        = {cfg.breaks.smart_pause_sec}',
            f'eye_break_interval_min = {cfg.breaks.eye_break_interval_min}',
            f'eye_break_duration_sec = {cfg.breaks.eye_break_duration_sec}',
            "",
            "[escalation]",
            f'level_2_delay_min  = {cfg.escalation.level_2_delay_min}',
            f'level_3_delay_min  = {cfg.escalation.level_3_delay_min}',
            f'mandatory_break    = {str(cfg.escalation.mandatory_break).lower()}',
            f'respect_focus_mode = {str(cfg.escalation.respect_focus_mode).lower()}',
            "",
            "[audio]",
            f'enabled = {str(cfg.audio.enabled).lower()}',
            f'volume  = {cfg.audio.volume}',
            "",
            "[audio.sounds]",
            f'level_1   = "{cfg.audio.sounds.level_1}"',
            f'level_2   = "{cfg.audio.sounds.level_2}"',
            f'level_3   = "{cfg.audio.sounds.level_3}"',
            f'break_end = "{cfg.audio.sounds.break_end}"',
            "",
            "[privacy]",
            f'store_detections    = {str(cfg.privacy.store_detections).lower()}',
            "",
            "[ui]",
            f'start_minimized         = {str(cfg.ui.start_minimized).lower()}',
            f'tray_icon_color_coding  = {str(cfg.ui.tray_icon_color_coding).lower()}',
            f'theme                   = "{cfg.ui.theme}"',
            "",
            "[dark_hours]",
            f'enabled         = {str(cfg.dark_hours.enabled).lower()}',
            f'start_hour      = {cfg.dark_hours.start_hour}',
            f'end_hour        = {cfg.dark_hours.end_hour}',
            f'reduce_volume   = {str(cfg.dark_hours.reduce_volume).lower()}',
            f'stricter_posture = {str(cfg.dark_hours.stricter_posture).lower()}',
        ]

        path.write_text("\n".join(lines), encoding="utf-8")

