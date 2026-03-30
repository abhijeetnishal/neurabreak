"""Tests for HealthJournalService against a real in-memory SQLite database.

We use SQLAlchemy's in-memory SQLite engine — this is fast, isolated per
test, and exercises all the real SQL without touching the filesystem.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed; run `uv sync --extra data`")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from neurabreak.data.database import Database
from neurabreak.data.journal import HealthJournalService
from neurabreak.data.models import Base, Break, Detection, Session


#  Fixture: in-memory database

@pytest.fixture()
def db(tmp_path):
    """Real Database backed by a temp-file SQLite (in-memory breaks across
    sessions, so we use a tmpdir file instead)."""
    db_path = tmp_path / "test_journal.db"
    database = Database(db_path)
    database.connect()
    yield database
    database.disconnect()


@pytest.fixture()
def journal(db):
    return HealthJournalService(db)


#  Session lifecycle

class TestSessionLifecycle:
    def test_start_session_creates_row(self, db, journal):
        journal.start_session()
        assert journal._current_session_id is not None

        with db.session() as sess:
            row = sess.get(Session, journal._current_session_id)
            assert row is not None
            assert row.start_time is not None

    def test_end_session_stamps_end_time(self, db, journal):
        journal.start_session()
        session_id = journal._current_session_id

        journal.end_session()

        # session_id should be cleared after ending
        assert journal._current_session_id is None

        with db.session() as sess:
            row = sess.get(Session, session_id)
            assert row.end_time is not None
            assert row.total_active_secs >= 0

    def test_end_session_without_start_is_safe(self, journal):
        # Should not raise even if called before start_session
        journal.end_session()

    def test_multiple_sessions_are_independent(self, db, journal):
        journal.start_session()
        id1 = journal._current_session_id
        journal.end_session()

        journal.start_session()
        id2 = journal._current_session_id
        journal.end_session()

        assert id1 != id2

        with db.session() as sess:
            assert sess.get(Session, id1) is not None
            assert sess.get(Session, id2) is not None


#  Detection recording

class TestDetectionRecording:
    def test_record_detection_inserts_row(self, db, journal):
        journal.start_session()
        journal.record_detection(
            posture_class="posture_good",
            confidence=0.93,
            is_face_present=True,
            inference_ms=28.5,
        )

        with db.session() as sess:
            dets = sess.query(Detection).all()
            assert len(dets) == 1
            det = dets[0]
            assert det.posture_class == "posture_good"
            assert abs(det.confidence - 0.93) < 1e-4
            assert det.is_face_present == 1
            assert det.phone_detected == 0
            assert det.inference_ms == 28.5

    def test_record_detection_phone_flag(self, db, journal):
        journal.start_session()
        journal.record_detection(
            posture_class="posture_good",
            confidence=0.8,
            is_face_present=True,
            phone_detected=True,
        )

        with db.session() as sess:
            det = sess.query(Detection).first()
            assert det.phone_detected == 1

    def test_record_detection_without_session_does_not_crash(self, db, journal):
        # session_id will be None — should still insert (nullable FK)
        journal.record_detection(
            posture_class="posture_bad",
            confidence=0.75,
            is_face_present=True,
        )
        with db.session() as sess:
            det = sess.query(Detection).first()
            assert det is not None
            assert det.session_id is None


#  Break recording

class TestBreakRecording:
    def test_record_break_returns_positive_id(self, journal):
        journal.start_session()
        break_id = journal.record_break("timer")
        assert break_id > 0

    def test_mark_break_taken(self, db, journal):
        journal.start_session()
        bid = journal.record_break("posture")
        journal.mark_break_taken(bid)

        with db.session() as sess:
            brk = sess.get(Break, bid)
            assert brk.acknowledged == 1
            assert brk.started_at is not None

    def test_mark_break_ended_calculates_duration(self, db, journal):
        journal.start_session()
        bid = journal.record_break("timer")
        journal.mark_break_taken(bid)
        journal.mark_break_ended(bid)

        with db.session() as sess:
            brk = sess.get(Break, bid)
            assert brk.ended_at is not None
            assert brk.duration_secs is not None
            assert brk.duration_secs >= 0

    def test_mark_break_invalid_id_is_safe(self, journal):
        journal.mark_break_taken(-1)
        journal.mark_break_ended(-1)


#  Aggregation queries

class TestAggregation:
    def test_get_today_summary_no_data_returns_none(self, journal):
        summary = journal.get_today_summary()
        assert summary is None

    def test_get_today_summary_with_data(self, db, journal):
        journal.start_session()
        for i in range(10):
            journal.record_detection(
                posture_class="posture_good" if i % 2 == 0 else "posture_bad",
                confidence=0.9,
                is_face_present=True,
            )
        journal.end_session()

        summary = journal.get_today_summary()
        assert summary is not None
        assert summary.total_active_secs >= 0

    def test_get_daily_scores_empty(self, journal):
        scores = journal.get_daily_scores(days=7)
        assert scores == []

    def test_get_daily_scores_groups_by_day(self, db, journal):
        journal.start_session()
        journal.record_detection("posture_good", 0.9, True)
        journal.end_session()

        scores = journal.get_daily_scores(days=7)
        assert len(scores) >= 1
        assert all(hasattr(s, "avg_score") for s in scores)

    def test_get_hourly_posture_today_empty(self, journal):
        data = journal.get_hourly_posture_today()
        assert data == []

    def test_get_hourly_posture_today_with_detections(self, db, journal):
        journal.start_session()
        for _ in range(5):
            journal.record_detection("posture_good", 0.9, True)
        for _ in range(5):
            journal.record_detection("posture_bad", 0.8, True)

        data = journal.get_hourly_posture_today()
        assert len(data) >= 1
        hour_entry = data[0]
        assert 0 <= hour_entry.good_pct <= 100
        assert 0 <= hour_entry.bad_pct <= 100

    def test_get_break_compliance_no_breaks(self, journal):
        stats = journal.get_break_compliance(days=7)
        assert stats == {"triggered": 0, "taken": 0, "skipped": 0}

    def test_get_break_compliance_partial(self, journal):
        journal.start_session()
        b1 = journal.record_break("timer")
        b2 = journal.record_break("timer")
        journal.mark_break_taken(b1)

        stats = journal.get_break_compliance(days=7)
        assert stats["triggered"] == 2
        assert stats["taken"] == 1
        assert stats["skipped"] == 1

    def test_posture_score_calculation(self, db, journal):
        journal.start_session()
        session_id = journal._current_session_id

        # 8 good + 2 bad = 80%
        for _ in range(8):
            journal.record_detection("posture_good", 0.9, True)
        for _ in range(2):
            journal.record_detection("posture_bad", 0.75, True)

        with db.session() as sess:
            score = journal._calc_posture_score(sess, session_id)
        assert abs(score - 0.8) < 0.01


#  Data export

class TestDataExport:
    def test_export_csv_creates_file(self, db, journal, tmp_path):
        journal.start_session()
        journal.record_detection("posture_good", 0.9, True)
        journal.record_detection("posture_bad", 0.8, True)

        out = str(tmp_path / "export.csv")
        journal.export_csv(out)

        assert Path(out).exists()
        content = Path(out).read_text()
        assert "posture_class" in content
        assert "posture_good" in content

    def test_export_csv_empty_does_not_raise(self, db, journal, tmp_path):
        # No data — should log a warning but not crash
        journal.export_csv(str(tmp_path / "empty.csv"))

    def test_export_json_creates_file(self, db, journal, tmp_path):
        journal.start_session()
        journal.end_session()

        out = str(tmp_path / "export.json")
        journal.export_json(out)

        data = json.loads(Path(out).read_text())
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "session_id" in data[0]

