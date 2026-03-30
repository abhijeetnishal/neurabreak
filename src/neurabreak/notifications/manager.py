"""Notification orchestrator — decides what to show and when.

Manages the escalation ladder: OS toast → balloon reminder → break screen.
Delegates to EscalationTimer for timing and AudioManager for sounds.
UI callbacks (on_level2, on_level3) let app.py hook in Qt UI updates from
the main thread safely.
"""

from __future__ import annotations

import threading
from typing import Callable

import structlog

from neurabreak.core.config import AppConfig
from neurabreak.core.events import Event, EventType, bus
from neurabreak.notifications.escalation import EscalationTimer

log = structlog.get_logger()


class NotificationManager:
    """Orchestrates the four-level alert escalation ladder.

    Level 1 — OS toast + soft chime
    Level 2 — tray balloon reminder + louder chime  (+level_2_delay_min ignored)
    Level 3 — break screen overlay + ambient audio  (+level_3_delay_min more)
    Level 4 — mandatory (optional, user must enable in config)
    """

    def __init__(
        self,
        config: AppConfig,
        audio_manager: object | None = None,
    ) -> None:
        self.config = config
        self.audio = audio_manager
        self._escalation = EscalationTimer(
            level_2_delay_min=config.escalation.level_2_delay_min,
            level_3_delay_min=config.escalation.level_3_delay_min,
        )

        self._current_level = 0
        self._current_trigger: str | None = None  # 'break' | 'posture'
        self._current_message = ""
        self._level_lock = threading.Lock()  # guards _current_level
        self._snooze_timer: threading.Timer | None = None

        # UI layer registers callbacks here to show overlays from Qt main thread
        self._level1_cbs: list[Callable[[str], None]] = []
        self._level2_cbs: list[Callable[[str], None]] = []
        self._level3_cbs: list[Callable[[str], None]] = []

        # Callbacks for the 20-20-20 eye break
        self._eye_break_cbs: list[Callable[[str], None]] = []

        # Background thread that polls escalation and escalates if ignored
        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(
            target=self._escalation_monitor,
            name="EscalationMonitor",
            daemon=True,
        )

        bus.subscribe(EventType.BREAK_DUE, self._on_break_due)
        bus.subscribe(EventType.POSTURE_ALERT, self._on_posture_alert)
        bus.subscribe(EventType.BREAK_STARTED, lambda _: self.dismiss())
        bus.subscribe(EventType.BREAK_ENDED, lambda _: self.dismiss())
        bus.subscribe(EventType.POSTURE_RESTORED, self._on_posture_restored)
        bus.subscribe(EventType.SESSION_PAUSED, lambda _: self.dismiss())
        bus.subscribe(EventType.EYE_BREAK_DUE, self._on_eye_break_due)

    # Lifecycle

    def start(self) -> None:
        self._monitor_thread.start()
        log.info("notification_manager_started")

    def stop(self) -> None:
        self._stop_event.set()

    def apply_config(self, config: AppConfig) -> None:
        """Apply updated config values while the manager is running."""
        self.config = config
        with self._level_lock:
            active_level = self._current_level
        self._escalation = EscalationTimer(
            level_2_delay_min=config.escalation.level_2_delay_min,
            level_3_delay_min=config.escalation.level_3_delay_min,
        )
        if active_level > 0:
            self._escalation.start()

    # Callback registration (called by app.py before Qt event loop starts)

    def on_level1(self, callback: Callable[[str], None]) -> None:
        self._level1_cbs.append(callback)

    def on_level2(self, callback: Callable[[str], None]) -> None:
        self._level2_cbs.append(callback)

    def on_level3(self, callback: Callable[[str], None]) -> None:
        self._level3_cbs.append(callback)

    def on_eye_break(self, callback: Callable[[str], None]) -> None:
        """Register a callback invoked when the 20-20-20 eye break fires."""
        self._eye_break_cbs.append(callback)

    # Public control API

    def dismiss(self) -> None:
        """User took a break or posture was corrected — reset everything."""
        if self._snooze_timer is not None:
            self._snooze_timer.cancel()
            self._snooze_timer = None
        self._escalation.reset()
        with self._level_lock:
            self._current_level = 0
        self._current_trigger = None
        self._current_message = ""

    def snooze(self, minutes: int) -> None:
        """Defer the reminder by `minutes` minutes then re-fire BREAK_DUE."""
        self._escalation.reset()
        with self._level_lock:
            self._current_level = 0
        if self._snooze_timer is not None:
            self._snooze_timer.cancel()
        self._snooze_timer = threading.Timer(
            minutes * 60,
            lambda: bus.publish(Event(EventType.BREAK_DUE, {"reason": "snooze_expired"})),
        )
        self._snooze_timer.start()

    # Event handlers (called from the inference thread via the bus)

    def _on_break_due(self, event: Event) -> None:
        elapsed = event.data.get("session_elapsed_sec", 0)
        elapsed_min = int(elapsed // 60)
        self._current_trigger = "break"
        self._current_message = (
            f"You've been working for {elapsed_min} minutes. Time for a break!"
            if elapsed_min > 0
            else "Time for a break!"
        )
        self._trigger_level(1)

    def _on_posture_alert(self, event: Event) -> None:
        posture = event.data.get("posture", "")
        label = posture.replace("posture_", "").replace("_", " ")
        self._current_trigger = "posture"
        self._current_message = f"Posture check: {label}" if label else "Check your posture!"
        self._trigger_level(1)

    def _on_posture_restored(self, event: Event) -> None:
        if self._current_trigger == "posture":
            self.dismiss()

    def _on_eye_break_due(self, event: Event) -> None:
        """20-20-20 rule: gentle eye-rest reminder every eye_break_interval_min minutes.

        Fires independently of the escalation ladder — does NOT change _current_level
        so it cannot interfere with an in-progress posture or break alert.
        """
        interval = event.data.get("interval_min", 20)
        count = event.data.get("count", 1)
        message = (
            f"\U0001f441\ufe0f 20-20-20 Eye Break #{count}: "
            f"Look at something 20 feet away for 20 seconds to rest your eyes."
            if interval == 20
            else (
                f"\U0001f441\ufe0f Eye Break #{count}: "
                f"Look away from the screen for 20 seconds — let your eyes relax."
            )
        )
        
        self._send_toast("\U0001f441\ufe0f Eye Break", message)
        self._play("level_1")
        for cb in self._eye_break_cbs:
            try:
                cb(message)
            except Exception as exc:
                log.error("eye_break_callback_error", error=str(exc))

    # Core notification dispatch

    def _trigger_level(self, level: int) -> None:
        # Ignore stale escalation ticks after the flow was dismissed.
        if self._current_trigger is None:
            return

        with self._level_lock:
            if self._current_trigger is None:
                return
            if level <= self._current_level:
                return  # already at this level or higher
            self._current_level = level

        if level == 1:
            self._escalation.start()
            self._send_toast("NeuraBreak", self._current_message)
            self._play("level_1")
            for cb in self._level1_cbs:
                try:
                    cb(self._current_message)
                except Exception as exc:
                    log.error("level1_callback_error", error=str(exc))

        elif level == 2:
            self._send_toast("NeuraBreak Reminder", self._current_message)
            self._play("level_2")
            for cb in self._level2_cbs:
                try:
                    cb(self._current_message)
                except Exception as exc:
                    log.error("level2_callback_error", error=str(exc))

        elif level == 3:
            self._play("level_3")
            for cb in self._level3_cbs:
                try:
                    cb(self._current_message)
                except Exception as exc:
                    log.error("level3_callback_error", error=str(exc))

        elif level == 4 and self.config.escalation.mandatory_break:
            # Mandatory mode — same as level 3 for now
            for cb in self._level3_cbs:
                try:
                    cb(self._current_message)
                except Exception as exc:
                    log.error("level4_callback_error", error=str(exc))

    def _send_toast(self, title: str, message: str) -> None:
        # Suppress OS toasts when Windows Focus Assist is active
        if self._is_focus_mode_active():
            log.debug("toast_suppressed_focus_mode", title=title)
            return

        import platform

        if platform.system() == "Windows":
            from neurabreak.notifications.platforms.windows import send_toast
            send_toast(title, message)
        elif platform.system() == "Darwin":
            from neurabreak.notifications.platforms.macos import send_toast  # type: ignore
            send_toast(title, message)
        elif platform.system() == "Linux":
            from neurabreak.notifications.platforms.linux import send_toast  # type: ignore
            send_toast(title, message)

    def _play(self, sound_key: str) -> None:
        if self.audio is None:
            return
        from neurabreak.notifications.audio import AudioManager

        if isinstance(self.audio, AudioManager):
            # Respect dark-hours reduce_volume: scale down to 30 % of master volume
            dh = self.config.dark_hours
            volume_override: int | None = None
            if dh.enabled and self._is_dark_hours() and dh.reduce_volume:
                volume_override = max(1, int(self.config.audio.volume * 0.30))
            self.audio.play_configured(sound_key, self.config, volume_override=volume_override)

    # Dark-hours helpers

    def _is_dark_hours(self) -> bool:
        """Return True if the current wall-clock time falls inside the configured dark window."""
        import datetime

        dh = self.config.dark_hours
        if not dh.enabled:
            return False
        hour = datetime.datetime.now().hour
        start, end = dh.start_hour, dh.end_hour
        # Ranges that cross midnight (e.g. 22 – 07) need the OR form.
        if start > end:
            return hour >= start or hour < end
        return start <= hour < end

    # ————— Focus-mode helper (Windows only) ————————————————————

    def _is_focus_mode_active(self) -> bool:
        """Return True when Windows Focus Assist is in Priority-only or Alarms-only mode.

        Only runs on Windows. Fails safely on any error (returns False).
        """
        if not self.config.escalation.respect_focus_mode:
            return False
        try:
            import platform
            if platform.system() != "Windows":
                return False
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Explorer\FocusAssistSettings",
            ) as key:
                mode, _ = winreg.QueryValueEx(key, "FocusAssistMode")
                # 0 = Focus Assist off, 1 = priority only, 2 = alarms only
                return int(mode) in (1, 2)
        except Exception:
            return False

    # Escalation monitor — background thread

    def _escalation_monitor(self) -> None:
        """Polls the escalation timer every ~15 seconds and escalates if needed."""
        while not self._stop_event.wait(timeout=15.0):
            with self._level_lock:
                current_level = self._current_level
            if current_level == 0:
                continue  # nothing active
            new_level = self._escalation.check()
            if new_level > current_level:
                self._trigger_level(new_level)
        log.debug("escalation_monitor_stopped")
