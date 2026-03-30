"""Escalation timer — tracks how long each notification level has been ignored."""

from __future__ import annotations

import time

import structlog

log = structlog.get_logger()


class EscalationTimer:
    """Tracks how long a notification has been ignored and escalates accordingly.

    Level 1 is triggered by the state machine.
    Level 2 fires after level_2_delay_min of being ignored.
    Level 3 fires after level_3_delay_min more minutes of being ignored.
    """

    def __init__(self, level_2_delay_min: int = 2, level_3_delay_min: int = 5) -> None:
        self.level_2_delay_sec = level_2_delay_min * 60
        self.level_3_delay_sec = level_3_delay_min * 60
        self._triggered_at: float | None = None
        self._current_level = 0

    def start(self) -> None:
        """Call when a level-1 notification fires."""
        self._triggered_at = time.monotonic()
        self._current_level = 1

    def check(self) -> int:
        """Return the current escalation level based on elapsed time."""
        if self._triggered_at is None:
            return 0
        elapsed = time.monotonic() - self._triggered_at
        if elapsed >= self.level_2_delay_sec + self.level_3_delay_sec:
            self._current_level = 3
        elif elapsed >= self.level_2_delay_sec:
            self._current_level = 2
        return self._current_level

    def reset(self) -> None:
        """Call when user takes a break or dismisses the notification."""
        self._triggered_at = None
        self._current_level = 0
