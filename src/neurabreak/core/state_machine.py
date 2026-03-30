"""Posture state machine — governs all app state transitions.

States and their meanings:
  IDLE          — No one at the desk (or monitoring paused).
  MONITORING    — Person present, session timer running normally.
  POSTURE_ALERT — Bad posture detected for too long, alert in progress.
  BREAK_DUE     — Session time limit reached, break reminder issued.
  BREAK_ACTIVE  — Person is taking a break.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Callable

import structlog

from neurabreak.core.events import Event, EventType, bus

log = structlog.get_logger()


class AppState(Enum):
    IDLE = "idle"
    MONITORING = "monitoring"
    POSTURE_ALERT = "posture_alert"
    BREAK_DUE = "break_due"
    BREAK_ACTIVE = "break_active"


class PostureStateMachine:
    """Drives all state transitions for NeuraBreak.

    Call process() on every inference result (~5 FPS). The machine
    maintains the session timer, tracks consecutive frame counts, and
    publishes events on the global bus whenever state transitions happen.
    """

    def __init__(self, fps: int = 5, break_interval_min: int = 45,
                 posture_alert_sec: int = 60, smart_pause_sec: int = 30,
                 eye_break_interval_min: int = 20,
                 break_duration_min: int = 5) -> None:
        self.fps = fps
        self.break_interval_min = break_interval_min
        self.break_duration_min = break_duration_min
        self.posture_alert_sec = posture_alert_sec
        self.smart_pause_sec = smart_pause_sec
        self.eye_break_interval_min = eye_break_interval_min

        self.state = AppState.IDLE

        # Session timing
        self.session_started_at: float | None = None
        self.session_elapsed_sec: float = 0.0

        # Frame counters (reset on every direction change)
        self.consecutive_present: int = 0
        self.consecutive_absent: int = 0

        # Frame counter for consecutive bad posture
        self.bad_posture_frames: int = 0

        # 20-20-20 rule: counts how many eye breaks have fired in this session
        self._eye_break_count: int = 0

        # Break-active tracking
        self._break_active_started_at: float | None = None
        self._break_was_absent: bool = False
        self._break_due_absence_score: int = 0

        # Manual pause via tray "Pause Monitoring" button — prevents session auto-start
        self._manually_paused: bool = False

        # Optional callbacks for UI (e.g. tray icon color change)
        self._on_transition: list[Callable[[AppState, AppState], None]] = []

        # Subscribe to manual pause/resume events so the tray button actually works
        bus.subscribe(EventType.MONITORING_PAUSED_MANUAL, lambda _: self._on_manual_pause())
        bus.subscribe(EventType.MONITORING_RESUMED_MANUAL, lambda _: self._on_manual_resume())

    def on_state_change(self, callback: Callable[[AppState, AppState], None]) -> None:
        """Register a callback that fires on every state transition."""
        self._on_transition.append(callback)

    def apply_runtime_config(
        self,
        *,
        break_interval_min: int | None = None,
        break_duration_min: int | None = None,
        posture_alert_sec: int | None = None,
        smart_pause_sec: int | None = None,
        eye_break_interval_min: int | None = None,
    ) -> None:
        """Update timing thresholds while the app is running.

        This allows settings changes to take effect immediately without
        restarting the detection thread or the app.
        """
        if break_interval_min is not None:
            self.break_interval_min = max(1, int(break_interval_min))
        if break_duration_min is not None:
            self.break_duration_min = max(1, int(break_duration_min))
        if posture_alert_sec is not None:
            self.posture_alert_sec = max(1, int(posture_alert_sec))
        if smart_pause_sec is not None:
            self.smart_pause_sec = max(1, int(smart_pause_sec))
        if eye_break_interval_min is not None:
            self.eye_break_interval_min = max(0, int(eye_break_interval_min))
            # Rebase to "already-fired" count so changing the interval does
            # not immediately backfill old reminders from earlier in session.
            if self.eye_break_interval_min <= 0:
                self._eye_break_count = 0
            else:
                interval_sec = self.eye_break_interval_min * 60
                self._eye_break_count = int(self.session_elapsed_sec // interval_sec)


    def process(self, present: bool, posture_class: str | None, confidence: float) -> None:
        """Process one inference result. Called at ~fps rate.

        Args:
            present:      Whether a person (face) was detected.
            posture_class: Class label from model, e.g. "posture_good".
                           None when person is absent.
            confidence:   Model confidence score (0.0-1.0).
        """
        now = time.monotonic()

        # Update consecutive frame counters
        if present:
            self.consecutive_present += 1
            self.consecutive_absent = 0
        else:
            self.consecutive_absent += 1
            self.consecutive_present = 0

        # Dispatch to the appropriate handler for the current state
        if self.state == AppState.IDLE:
            self._from_idle(now, present)
        elif self.state == AppState.MONITORING:
            self._from_monitoring(now, present, posture_class)
        elif self.state == AppState.POSTURE_ALERT:
            self._from_posture_alert(now, present, posture_class)
        elif self.state == AppState.BREAK_DUE:
            self._from_break_due(present)
        elif self.state == AppState.BREAK_ACTIVE:
            self._from_break_active(now, present)


    def _from_idle(self, now: float, present: bool) -> None:
        # Don't auto-start a new session when user has manually paused monitoring.
        if self._manually_paused:
            return
        # Need ~2 seconds of continuous presence before we start a session.
        # At 5 fps that's 10 consecutive frames; at 3 fps it's 6 frames.
        needed = self.fps * 2
        if self.consecutive_present >= needed:
            self.session_started_at = now
            self.session_elapsed_sec = 0.0
            self._transition(AppState.MONITORING)
            bus.publish(Event(EventType.SESSION_STARTED))

    def _from_monitoring(self, now: float, present: bool, posture_class: str | None) -> None:
        # Smart pause: if we haven't seen anyone for long enough, pause the session
        pause_threshold = self.fps * self.smart_pause_sec
        if not present and self.consecutive_absent >= pause_threshold:
            self._transition(AppState.IDLE)
            bus.publish(Event(EventType.SESSION_PAUSED))
            return

        if not present:
            return  # brief absence, don't change state yet

        # Tick the session timer
        if self.session_started_at is not None:
            self.session_elapsed_sec = now - self.session_started_at

        # Is it time for a break?
        if self.session_elapsed_sec >= self.break_interval_min * 60:
            self._break_due_absence_score = 0
            self._transition(AppState.BREAK_DUE)
            bus.publish(Event(EventType.BREAK_DUE, {"session_elapsed_sec": self.session_elapsed_sec}))
            return

        # 20-20-20 eye break
        # Fire once per N-minute interval. Does not change app state.
        # Suppressed within 2 minutes of the scheduled work break to avoid
        # stacking a gentle eye reminder on top of an imminent full-break alert.
        if self.eye_break_interval_min > 0:
            time_to_break_sec = self.break_interval_min * 60 - self.session_elapsed_sec
            near_scheduled_break = time_to_break_sec <= 120
            interval_sec = self.eye_break_interval_min * 60
            expected_count = int(self.session_elapsed_sec // interval_sec)
            if expected_count > self._eye_break_count:
                self._eye_break_count = expected_count
                if not near_scheduled_break:
                    bus.publish(Event(EventType.EYE_BREAK_DUE, {
                        "interval_min": self.eye_break_interval_min,
                        "count": self._eye_break_count,
                    }))

        # Posture check — use frame counts
        good_classes = {"posture_good", "face_present", None}
        if posture_class not in good_classes:
            self.bad_posture_frames += 1
            if self.bad_posture_frames >= self.fps * self.posture_alert_sec:
                self._transition(AppState.POSTURE_ALERT)
                bus.publish(Event(EventType.POSTURE_ALERT, {"posture": posture_class}))
        else:
            # Posture is fine — reset the bad-posture counter
            self.bad_posture_frames = 0

    def _from_posture_alert(self, now: float, present: bool, posture_class: str | None) -> None:
        # Person left — smart pause takes precedence
        pause_threshold = self.fps * self.smart_pause_sec
        if not present and self.consecutive_absent >= pause_threshold:
            self._transition(AppState.IDLE)
            bus.publish(Event(EventType.SESSION_PAUSED))
            return
        # Brief absence (below pause threshold)
        if not present:
            return
        if self.session_started_at is not None:
            self.session_elapsed_sec = now - self.session_started_at

        _neutral_classes = {"posture_good", "face_present", None}
        if posture_class in _neutral_classes:
            self.bad_posture_frames = 0
            self._transition(AppState.MONITORING)
            bus.publish(Event(EventType.POSTURE_RESTORED))

    def _from_break_due(self, present: bool) -> None:
        # Notification manager calls start_break() / snooze() directly.
        # If the person steps away, auto-start the break
        absence_threshold = max(1, self.fps * min(5, self.smart_pause_sec))
        if not present:
            self._break_due_absence_score = min(
                absence_threshold,
                self._break_due_absence_score + 1,
            )
        else:
            decay = max(1, self.fps // 2)  # ~0.5s of presence clears some score
            self._break_due_absence_score = max(0, self._break_due_absence_score - decay)

        if self._break_due_absence_score >= absence_threshold:
            self.start_break()

    def _from_break_active(self, now: float, present: bool) -> None:
        """Handle inference ticks while the user is on a break.

        Transitions back to MONITORING once the user has stepped away and
        returned, or after the configured break duration elapses (prevents
        the machine from getting stuck in BREAK_ACTIVE if the user never
        truly walked away).
        """
        if self._break_active_started_at is None:
            self._break_active_started_at = now

        if not present:
            self._break_was_absent = True
            return

        # Person is back at the desk
        elapsed = now - self._break_active_started_at
        presence_threshold = self.fps * 3  # 3 seconds of stable presence
        break_duration_sec = self.break_duration_min * 60

        if self.consecutive_present >= presence_threshold:
            # End break if user was absent at any point, OR if the full break
            # duration has elapsed (handles the case where they never walked away).
            if self._break_was_absent or elapsed >= break_duration_sec:
                self.end_break()


    def start_break(self) -> None:
        """Called when the user acknowledges a break reminder (or auto-triggered)."""
        if self.state == AppState.BREAK_ACTIVE:
            return
        self._break_due_absence_score = 0
        self._break_active_started_at = time.monotonic()
        self._break_was_absent = False
        self._transition(AppState.BREAK_ACTIVE)
        bus.publish(Event(EventType.BREAK_STARTED))

    def end_break(self) -> None:
        """Called when the break is over — resets session and resumes monitoring."""
        if self.state != AppState.BREAK_ACTIVE:
            return  # safeguard against double-calls
        self.session_started_at = time.monotonic()
        self.session_elapsed_sec = 0.0
        self.bad_posture_frames = 0
        # Reset eye-break counter so it fires again in the new session
        self._eye_break_count = 0
        self._break_due_absence_score = 0
        self._break_active_started_at = None
        self._break_was_absent = False
        self._transition(AppState.MONITORING)
        bus.publish(Event(EventType.BREAK_ENDED))

    def reset(self) -> None:
        """Full reset — used on manual pause or app restart."""
        self._transition(AppState.IDLE)
        self.session_elapsed_sec = 0.0
        self.session_started_at = None
        self.bad_posture_frames = 0
        self.consecutive_present = 0
        self.consecutive_absent = 0
        self._eye_break_count = 0
        self._break_due_absence_score = 0
        self._break_active_started_at = None
        self._break_was_absent = False
        self._manually_paused = False

    def _on_manual_pause(self) -> None:
        """Called when the tray's Pause Monitoring button is clicked."""
        self._manually_paused = True
        self._break_due_absence_score = 0
        # Push to IDLE so the current session timer stops and any alert is dismissed.
        if self.state not in (AppState.IDLE,):
            self._transition(AppState.IDLE)
            bus.publish(Event(EventType.SESSION_PAUSED))

    def _on_manual_resume(self) -> None:
        """Called when the tray's Resume Monitoring button is clicked."""
        self._manually_paused = False
        # State machine re-enters MONITORING naturally on next presence detection.


    def _transition(self, new_state: AppState) -> None:
        old_state = self.state
        self.state = new_state

        for cb in self._on_transition:
            try:
                cb(old_state, new_state)
            except Exception as e:
                log.error("state_transition_callback_error", error=str(e))
