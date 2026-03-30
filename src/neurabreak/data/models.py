"""SQLAlchemy ORM models.

These are defined up front so the rest of the codebase can import them
even before the database is connected (lazy connection is fine with SQLAlchemy).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[datetime | None] = mapped_column(DateTime)
    total_active_secs: Mapped[int] = mapped_column(Integer, default=0)
    breaks_taken: Mapped[int] = mapped_column(Integer, default=0)
    avg_posture_score: Mapped[float] = mapped_column(Float, default=0.0)

    detections: Mapped[list["Detection"]] = relationship(back_populates="session")
    breaks: Mapped[list["Break"]] = relationship(back_populates="session")

    def __repr__(self) -> str:
        return f"<Session id={self.id} started={self.start_time.isoformat()}>"


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("sessions.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    posture_class: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    inference_ms: Mapped[float | None] = mapped_column(Float)
    is_face_present: Mapped[int] = mapped_column(Integer, nullable=False)  # 0 or 1
    phone_detected: Mapped[int] = mapped_column(Integer, default=0)

    session: Mapped["Session | None"] = relationship(back_populates="detections")


class Break(Base):
    __tablename__ = "breaks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("sessions.id"))
    trigger_type: Mapped[str] = mapped_column(String(20), nullable=False)  # posture | timer | manual
    triggered_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)   # when user actually took it
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    duration_secs: Mapped[int | None] = mapped_column(Integer)
    acknowledged: Mapped[int] = mapped_column(Integer, default=0)

    session: Mapped["Session | None"] = relationship(back_populates="breaks")

