"""Health journal — persists detection events and session summaries."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import func, select

from neurabreak.data.models import Break, Detection, Session

if TYPE_CHECKING:
    from neurabreak.data.database import Database

log = structlog.get_logger()


@dataclass
class SessionSummary:
    session_id: int
    start_time: datetime
    end_time: datetime | None
    total_active_secs: int
    breaks_taken: int
    avg_posture_score: float


@dataclass
class HourlyPosture:
    """Aggregated posture stats for one hour of the day."""
    hour: int          # 0-23
    good_pct: float    # percentage of frames with good posture
    bad_pct: float
    total_samples: int


@dataclass
class DailyScore:
    """Posture score summary for a single calendar day."""
    day: date
    avg_score: float   # 0.0 – 1.0, where 1.0 = perfect posture all day
    total_active_min: int
    breaks_taken: int


class HealthJournalService:
    """Records every detection event and exposes aggregated stats.

    All data stays local in SQLite — nothing goes to the network.
    The dashboard reads from here for charts.
    """

    _GOOD_POSTURE_CLASSES = {"posture_good", "face_present"}

    def __init__(self, db: Database) -> None:
        self._db = db
        self._current_session_id: int | None = None


    #  Session lifecycle

    def start_session(self) -> None:
        """Creates a new session row and caches its ID."""
        with self._db.session() as sess:
            session_row = Session(
                start_time=datetime.now(),
                total_active_secs=0,
                breaks_taken=0,
                avg_posture_score=0.0,
            )
            sess.add(session_row)
            sess.flush()  # populate the auto-increment id before commit
            self._current_session_id = session_row.id

    def end_session(self) -> None:
        """Stamps the end time and recalculates totals for the current session."""
        if self._current_session_id is None:
            return
        with self._db.session() as sess:
            session_row = sess.get(Session, self._current_session_id)
            if session_row is None:
                return

            now = datetime.now()
            session_row.end_time = now

            # Recalculate active time from the accumulated detections
            elapsed = (now - session_row.start_time).total_seconds()
            session_row.total_active_secs = int(elapsed)

            # Recalculate average posture score
            score = self._calc_posture_score(sess, self._current_session_id)
            session_row.avg_posture_score = score

            # Recount breaks taken (acknowledged) from the Break table
            breaks_taken = sess.scalar(
                select(func.count(Break.id)).where(
                    Break.session_id == self._current_session_id,
                    Break.acknowledged == 1,
                )
            ) or 0
            session_row.breaks_taken = breaks_taken

        self._current_session_id = None

    #  Write events

    def record_detection(
        self,
        posture_class: str,
        confidence: float,
        is_face_present: bool,
        phone_detected: bool = False,
        inference_ms: float | None = None,
    ) -> None:
        """Persist one detection result. Called at the configured FPS rate."""
        with self._db.session() as sess:
            det = Detection(
                session_id=self._current_session_id,
                timestamp=datetime.now(),
                posture_class=posture_class,
                confidence=confidence,
                inference_ms=inference_ms,
                is_face_present=int(is_face_present),
                phone_detected=int(phone_detected),
            )
            sess.add(det)

    def record_break(self, trigger_type: str) -> int:
        """Log a break being triggered. Returns the new break ID."""
        with self._db.session() as sess:
            brk = Break(
                session_id=self._current_session_id,
                trigger_type=trigger_type,
                triggered_at=datetime.now(),
            )
            sess.add(brk)
            sess.flush()
            break_id = brk.id
            log.debug("journal_break_recorded", trigger=trigger_type, id=break_id)
        return break_id

    def mark_break_taken(self, break_id: int) -> None:
        """Stamp a break as actually started by the user."""
        if break_id < 0:
            return
        with self._db.session() as sess:
            brk = sess.get(Break, break_id)
            if brk:
                brk.started_at = datetime.now()
                brk.acknowledged = 1

    def mark_break_ended(self, break_id: int) -> None:
        """Stamp a break as completed and record how long it lasted."""
        if break_id < 0:
            return
        with self._db.session() as sess:
            brk = sess.get(Break, break_id)
            if brk:
                ended = datetime.now()
                brk.ended_at = ended
                if brk.started_at:
                    brk.duration_secs = int((ended - brk.started_at).total_seconds())

    #  Internal helpers

    def _live_active_secs(self, row: "Session") -> int:  # type: ignore[name-defined]
        """Return the active seconds for a session row.

        Only the CURRENT in-progress session (id matches ``_current_session_id``)
        gets a live elapsed time computed from ``start_time``.  All other rows
        use their persisted ``total_active_secs`` value, which means:

        * Completed sessions (``end_time`` is set): correct stored duration.
        * Orphaned sessions from a previous run that was force-killed
          (``end_time=None`` but not the current session): returns their stored
          ``total_active_secs`` (0 for true orphans) instead of a growing
          wall-clock delta — prevents stale rows from inflating the total.
        """
        if row.end_time is None and row.id == self._current_session_id:
            return int((datetime.now() - row.start_time).total_seconds())
        return row.total_active_secs

    #  Read / aggregation — used by the dashboard

    def get_today_summary(self) -> SessionSummary | None:
        """Aggregate today's sessions into one summary row for the dashboard."""
        today = date.today()
        day_start = datetime.combine(today, datetime.min.time())
        day_end = day_start + timedelta(days=1)

        with self._db.session() as sess:
            rows = sess.scalars(
                select(Session).where(
                    Session.start_time >= day_start,
                    Session.start_time < day_end,
                )
            ).all()

            if not rows:
                return None

            total_active = sum(self._live_active_secs(r) for r in rows)
            total_breaks = sum(r.breaks_taken for r in rows)
            scores = [r.avg_posture_score for r in rows if r.avg_posture_score > 0]
            avg_score = sum(scores) / len(scores) if scores else 0.0

            first = rows[0]
            last = rows[-1]

            return SessionSummary(
                session_id=first.id,
                start_time=first.start_time,
                end_time=last.end_time,
                total_active_secs=total_active,
                breaks_taken=total_breaks,
                avg_posture_score=avg_score,
            )

    def get_daily_scores(self, days: int = 7) -> list[DailyScore]:
        """Return posture score and activity stats for the past N days."""
        cutoff = datetime.now() - timedelta(days=days)

        # Use a namedtuple-style dict so we don't hold live ORM objects outside
        # the session — accessing detached attributes raises DetachedInstanceError.
        by_day: dict[date, list[tuple[int, int, float]]] = {}
        with self._db.session() as sess:
            rows = sess.scalars(
                select(Session).where(Session.start_time >= cutoff).order_by(Session.start_time)
            ).all()
            for r in rows:
                day = r.start_time.date()
                by_day.setdefault(day, []).append(
                    (self._live_active_secs(r), r.breaks_taken, r.avg_posture_score)
                )

        result = []
        for day, entries in sorted(by_day.items()):
            active_min = sum(e[0] for e in entries) // 60
            breaks = sum(e[1] for e in entries)
            scores = [e[2] for e in entries if e[2] > 0]
            avg = sum(scores) / len(scores) if scores else 0.0
            result.append(DailyScore(day=day, avg_score=avg, total_active_min=active_min, breaks_taken=breaks))

        return result

    def get_hourly_posture_today(self) -> list[HourlyPosture]:
        """Break today's detections down by hour — for the heatmap chart."""
        today = date.today()
        day_start = datetime.combine(today, datetime.min.time())
        day_end = day_start + timedelta(days=1)

        # Bucket per hour — read everything we need while the session is open
        hourly: dict[int, dict[str, int]] = {h: {"good": 0, "total": 0} for h in range(24)}
        with self._db.session() as sess:
            detections = sess.scalars(
                select(Detection).where(
                    Detection.timestamp >= day_start,
                    Detection.timestamp < day_end,
                )
            ).all()
            for det in detections:
                hour = det.timestamp.hour
                is_good = det.posture_class in self._GOOD_POSTURE_CLASSES
                hourly[hour]["total"] += 1
                if is_good:
                    hourly[hour]["good"] += 1

        result = []
        for h in range(24):
            total = hourly[h]["total"]
            good = hourly[h]["good"]
            if total == 0:
                continue
            result.append(HourlyPosture(
                hour=h,
                good_pct=good / total * 100,
                bad_pct=(total - good) / total * 100,
                total_samples=total,
            ))
        return result

    def get_break_compliance(self, days: int = 7) -> dict[str, int]:
        """Count triggered vs taken breaks over the past N days."""
        cutoff = datetime.now() - timedelta(days=days)

        with self._db.session() as sess:
            breaks = sess.scalars(
                select(Break).where(Break.triggered_at >= cutoff)
            ).all()
            triggered = len(breaks)
            taken = sum(1 for b in breaks if b.acknowledged)

        return {
            "triggered": triggered,
            "taken": taken,
            "skipped": triggered - taken,
        }

    #  Data export

    def export_csv(self, output_path: str) -> None:
        """Export all detection data to a CSV file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._db.session() as sess:
            detections = sess.scalars(select(Detection).order_by(Detection.timestamp)).all()
            rows = [
                {
                    "id": d.id,
                    "session_id": d.session_id,
                    "timestamp": d.timestamp.isoformat(),
                    "posture_class": d.posture_class,
                    "confidence": round(d.confidence, 4),
                    "inference_ms": d.inference_ms,
                    "is_face_present": bool(d.is_face_present),
                    "phone_detected": bool(d.phone_detected),
                }
                for d in detections
            ]

        if not rows:
            log.warning("export_csv_empty")
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        log.info("export_csv_done", path=str(path), rows=len(rows))

    def export_json(self, output_path: str) -> None:
        """Export all sessions + detections + breaks to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._db.session() as sess:
            sessions = sess.scalars(select(Session).order_by(Session.start_time)).all()
            payload = []
            for s in sessions:
                payload.append({
                    "session_id": s.id,
                    "start_time": s.start_time.isoformat(),
                    "end_time": s.end_time.isoformat() if s.end_time else None,
                    "total_active_secs": s.total_active_secs,
                    "breaks_taken": s.breaks_taken,
                    "avg_posture_score": round(s.avg_posture_score, 4),
                })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        log.info("export_json_done", path=str(path), sessions=len(payload))

    #  Internal helpers

    def _calc_posture_score(self, sess, session_id: int) -> float:
        """Fraction of detections in this session with good posture."""
        total = sess.scalar(
            select(func.count(Detection.id)).where(Detection.session_id == session_id)
        ) or 0
        if total == 0:
            return 0.0
        good = sess.scalar(
            select(func.count(Detection.id)).where(
                Detection.session_id == session_id,
                Detection.posture_class.in_(list(self._GOOD_POSTURE_CLASSES)),
            )
        ) or 0
        return good / total

