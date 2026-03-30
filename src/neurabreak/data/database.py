"""Database connection and schema management."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import structlog
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from neurabreak.data.models import Base

log = structlog.get_logger()


# Enable WAL mode for SQLite so reads and writes don't block each other.
# The dashboard reads while the inference thread writes.
@event.listens_for(Engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _connection_record) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class Database:
    """Manages the SQLAlchemy engine and session factory.

    Usage:
        db = Database(db_path)
        db.connect()
        with db.session() as session:
            session.add(...)
        db.disconnect()
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._engine: Engine | None = None
        self._SessionLocal: sessionmaker | None = None

    def connect(self) -> None:
        """Open the database and create tables if they don't exist yet."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        url = f"sqlite:///{self.db_path}"
        self._engine = create_engine(
            url,
            echo=False,
            # Allow the same connection to be used across threads — safe
            # SQLAlchemy manages locking internally.
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self._engine)
        self._SessionLocal = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )
        log.info("database_connected", path=str(self.db_path))

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Yield a transactional session that auto-commits or rolls back."""
        if self._SessionLocal is None:
            raise RuntimeError("Database is not connected — call connect() first")
        db = self._SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def disconnect(self) -> None:
        if self._engine:
            self._engine.dispose()
            self._engine = None
        log.info("database_disconnected")

    @property
    def is_connected(self) -> bool:
        return self._engine is not None
