"""Integration smoke tests — notification flow end-to-end."""

from __future__ import annotations

import pytest


class TestNotificationImports:
    def test_notification_modules_importable(self):
        from neurabreak.notifications.audio import AudioManager
        from neurabreak.notifications.escalation import EscalationTimer
        from neurabreak.notifications.manager import NotificationManager

    def test_escalation_timer_starts_at_level_0(self):
        from neurabreak.notifications.escalation import EscalationTimer

        timer = EscalationTimer(level_2_delay_min=2, level_3_delay_min=5)
        assert timer.check() == 0

    def test_escalation_timer_starts_at_level_1_after_start(self):
        from neurabreak.notifications.escalation import EscalationTimer

        timer = EscalationTimer(level_2_delay_min=999, level_3_delay_min=999)
        timer.start()
        # Right after starting: level 1, not enough time for level 2
        assert timer.check() == 1

    def test_escalation_timer_resets(self):
        from neurabreak.notifications.escalation import EscalationTimer

        timer = EscalationTimer()
        timer.start()
        timer.reset()
        assert timer.check() == 0


class TestNotificationManagerEvents:
    def test_notification_manager_subscribes_to_events(self):
        from neurabreak.core.config import AppConfig
        from neurabreak.core.events import bus, EventType
        from neurabreak.notifications.manager import NotificationManager

        cfg = AppConfig()
        manager = NotificationManager(config=cfg)

        # Should not raise — manager is subscribed to BREAK_DUE and POSTURE_ALERT
        from neurabreak.core.events import Event

        bus.publish(Event(EventType.BREAK_DUE, {"session_elapsed_sec": 2700}))
        bus.publish(Event(EventType.POSTURE_ALERT, {"posture": "posture_bad"}))
