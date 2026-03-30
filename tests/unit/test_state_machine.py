"""Tests for the posture state machine.

These test FSM transitions without any camera or model — just mock inputs.
Tests are written based on the state transition.
"""

from __future__ import annotations

import time

import pytest

from neurabreak.core.state_machine import AppState, PostureStateMachine


def make_sm(**kwargs) -> PostureStateMachine:
    """Helper — create a state machine with fast thresholds for testing."""
    defaults = {
        "fps": 5,
        "break_interval_min": 1,  # 1 minute so tests don't have to wait
        "posture_alert_sec": 2,   # 2 seconds
        "smart_pause_sec": 2,     # 2 seconds
    }
    defaults.update(kwargs)
    return PostureStateMachine(**defaults)


class TestIdleToMonitoring:
    def test_transitions_on_2s_of_continuous_presence(self):
        sm = make_sm(fps=5)
        # Need 5 fps * 2 sec = 10 consecutive frames
        for _ in range(9):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)
        assert sm.state == AppState.IDLE  # not yet

        sm.process(present=True, posture_class="posture_good", confidence=0.9)
        assert sm.state == AppState.MONITORING

    def test_brief_presence_doesnt_start_session(self):
        sm = make_sm(fps=5)
        for _ in range(5):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)
        assert sm.state == AppState.IDLE


class TestSmartPause:
    def test_transitions_to_idle_on_prolonged_absence(self):
        sm = make_sm(fps=5, smart_pause_sec=2)
        # Get to MONITORING first (fps*2 = 10 frames required)
        for _ in range(10):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)
        assert sm.state == AppState.MONITORING

        # 5 fps * 2 sec = 10 absent frames
        for _ in range(10):
            sm.process(present=False, posture_class=None, confidence=0.0)
        assert sm.state == AppState.IDLE

    def test_brief_absence_doesnt_pause(self):
        sm = make_sm(fps=5, smart_pause_sec=10)
        for _ in range(15):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)

        for _ in range(5):  # only 5 absent frames, threshold is 50
            sm.process(present=False, posture_class=None, confidence=0.0)
        assert sm.state == AppState.MONITORING


class TestPostureAlert:
    def test_bad_posture_triggers_alert_after_threshold(self):
        sm = make_sm(fps=5, posture_alert_sec=2)
        # Get to MONITORING
        for _ in range(15):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)

        # Slouch for 2 seconds = 10 frames
        for _ in range(10):
            sm.process(present=True, posture_class="posture_bad", confidence=0.9)
        assert sm.state == AppState.POSTURE_ALERT

    def test_posture_restored_goes_back_to_monitoring(self):
        sm = make_sm(fps=5, posture_alert_sec=2)
        for _ in range(15):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)
        for _ in range(10):
            sm.process(present=True, posture_class="posture_bad", confidence=0.9)
        assert sm.state == AppState.POSTURE_ALERT

        sm.process(present=True, posture_class="posture_good", confidence=0.9)
        assert sm.state == AppState.MONITORING

    def test_good_posture_resets_bad_posture_clock(self):
        sm = make_sm(fps=5, posture_alert_sec=2)
        for _ in range(15):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)

        # Slouch for 1 second (5 frames), then sit straight
        for _ in range(5):
            sm.process(present=True, posture_class="posture_bad", confidence=0.9)
        sm.process(present=True, posture_class="posture_good", confidence=0.9)
        assert sm.bad_posture_frames == 0
        assert sm.state == AppState.MONITORING


class TestBreakDue:
    def test_state_callback_is_called_on_transition(self):
        sm = make_sm()
        transitions_seen = []
        sm.on_state_change(lambda old, new: transitions_seen.append((old, new)))

        for _ in range(15):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)

        assert (AppState.IDLE, AppState.MONITORING) in transitions_seen

    def test_reset_returns_to_idle(self):
        sm = make_sm()
        for _ in range(15):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)
        assert sm.state == AppState.MONITORING

        sm.reset()
        assert sm.state == AppState.IDLE
        assert sm.session_elapsed_sec == 0.0

    def test_start_and_end_break(self):
        sm = make_sm()
        for _ in range(15):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)
        sm._transition(AppState.BREAK_DUE)

        sm.start_break()
        assert sm.state == AppState.BREAK_ACTIVE

        sm.end_break()
        assert sm.state == AppState.MONITORING
        assert sm.session_elapsed_sec == pytest.approx(0.0, abs=0.1)


class TestRuntimeConfig:
    def test_break_interval_change_applies_without_restart(self):
        sm = make_sm(break_interval_min=45)

        # Enter MONITORING first.
        for _ in range(10):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)
        assert sm.state == AppState.MONITORING

        # Pretend this session has already run for 6 minutes.
        sm.session_started_at = time.monotonic() - (6 * 60)
        sm.apply_runtime_config(break_interval_min=5)

        # Next frame should evaluate against the new 5-minute threshold.
        sm.process(present=True, posture_class="posture_good", confidence=0.9)
        assert sm.state == AppState.BREAK_DUE


class TestBreakDueAutoStart:
    def test_away_user_auto_starts_break(self):
        sm = make_sm(fps=5, smart_pause_sec=30)

        for _ in range(10):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)
        sm._transition(AppState.BREAK_DUE)

        # BREAK_DUE should auto-start break after ~5 seconds of absence.
        for _ in range(25):
            sm.process(present=False, posture_class=None, confidence=0.0)

        assert sm.state == AppState.BREAK_ACTIVE

    def test_break_due_tolerates_brief_presence_flicker(self):
        sm = make_sm(fps=5, smart_pause_sec=30)

        for _ in range(10):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)
        sm._transition(AppState.BREAK_DUE)

        # Mostly absent with a couple of noisy present frames.
        pattern = [False] * 10 + [True] + [False] * 10 + [True] + [False] * 10
        for present in pattern:
            sm.process(
                present=present,
                posture_class="face_present" if present else None,
                confidence=0.9 if present else 0.0,
            )

        assert sm.state == AppState.BREAK_ACTIVE

    def test_break_due_does_not_auto_start_while_present(self):
        sm = make_sm(fps=5, smart_pause_sec=30)

        for _ in range(10):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)
        sm._transition(AppState.BREAK_DUE)

        for _ in range(40):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)

        assert sm.state == AppState.BREAK_DUE
