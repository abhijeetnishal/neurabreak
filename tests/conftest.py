"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture
def qapp() -> QApplication: # type: ignore
    """Return a QApplication instance for Qt widget tests.

    This mirrors the small part of pytest-qt's qapp fixture that these tests
    need, while keeping the suite runnable in environments where the plugin is
    not installed.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(["neurabreak-tests"])

    app.setQuitOnLastWindowClosed(False)
    yield app

    app.processEvents()
    for widget in QApplication.topLevelWidgets():
        widget.close()
    app.processEvents()
