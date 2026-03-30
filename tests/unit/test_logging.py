"""Regression tests for startup logging bootstrap behavior."""

from __future__ import annotations

import logging
import sys

from neurabreak.core.logging import setup_logging


class _DummyStream:
    def __init__(self, fail_isatty: bool = False):
        self._fail_isatty = fail_isatty

    def write(self, _msg: str) -> int:
        return 0

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        if self._fail_isatty:
            raise RuntimeError("isatty unavailable")
        return False


def _restore_root_handlers(original_handlers: list[logging.Handler]) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    for handler in original_handlers:
        root.addHandler(handler)


def test_setup_logging_handles_missing_stdio(monkeypatch):
    """PyInstaller windowed mode can expose stdout/stderr as None."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)

    try:
        monkeypatch.setattr(sys, "stdout", None)
        monkeypatch.setattr(sys, "stderr", None)

        setup_logging()

        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.NullHandler)
    finally:
        _restore_root_handlers(original_handlers)


def test_setup_logging_handles_broken_isatty(monkeypatch):
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    stream = _DummyStream(fail_isatty=True)

    try:
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", None)

        setup_logging()

        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream is stream
    finally:
        _restore_root_handlers(original_handlers)
