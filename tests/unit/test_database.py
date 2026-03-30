"""Tests for the Database connection layer."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed; run `uv sync --extra data`")

from neurabreak.data.database import Database
from neurabreak.data.models import Base, Session


class TestDatabase:
    def test_connect_creates_tables(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.connect()

        assert db.is_connected
        assert (tmp_path / "test.db").exists()

        db.disconnect()

    def test_disconnect_marks_not_connected(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.connect()
        db.disconnect()
        assert not db.is_connected

    def test_session_context_manager_commits(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.connect()

        with db.session() as sess:
            row = Session(
                start_time=__import__("datetime").datetime.now(),
                total_active_secs=0,
                breaks_taken=0,
                avg_posture_score=0.0,
            )
            sess.add(row)

        # Row should persist after the context manager exits
        with db.session() as sess:
            count = sess.query(Session).count()
            assert count == 1

        db.disconnect()

    def test_session_rolls_back_on_error(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.connect()

        try:
            with db.session() as sess:
                row = Session(
                    start_time=__import__("datetime").datetime.now(),
                    total_active_secs=0,
                    breaks_taken=0,
                    avg_posture_score=0.0,
                )
                sess.add(row)
                raise RuntimeError("simulated error")
        except RuntimeError:
            pass

        # The rolled-back row should NOT appear
        with db.session() as sess:
            assert sess.query(Session).count() == 0

        db.disconnect()

    def test_session_without_connect_raises(self, tmp_path):
        db = Database(tmp_path / "test.db")
        with pytest.raises(RuntimeError, match="not connected"):
            with db.session() as sess:
                pass

    def test_disconnect_without_connect_is_safe(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.disconnect()  # should not raise

    def test_double_connect_reconnects(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.connect()
        db.connect()  # second connect — should not raise or corrupt
        assert db.is_connected
        db.disconnect()

    def test_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "test.db"
        db = Database(deep_path)
        db.connect()
        assert deep_path.exists()
        db.disconnect()
