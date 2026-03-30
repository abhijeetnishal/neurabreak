"""Shared branding utilities for NeuraBreak UI surfaces."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QWidget

_LOGO_REL_PATH = Path("assets") / "logo" / "neurabreak_logo.svg"


def _repo_root() -> Path:
    # src/neurabreak/ui/branding.py -> repo root
    return Path(__file__).resolve().parents[3]


def _candidate_logo_paths() -> list[Path]:
    candidates: list[Path] = [
        _repo_root() / _LOGO_REL_PATH,
        Path(__file__).resolve().parent / _LOGO_REL_PATH,
    ]

    if getattr(sys, "frozen", False):
        meipass = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        candidates = [
            meipass / _LOGO_REL_PATH,
            meipass / "neurabreak" / "ui" / _LOGO_REL_PATH,
        ] + candidates

    return candidates


@lru_cache(maxsize=1)
def get_logo_path() -> Path | None:
    for candidate in _candidate_logo_paths():
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def get_app_icon() -> QIcon:
    logo_path = get_logo_path()
    if logo_path is None:
        return QIcon()
    return QIcon(str(logo_path))


def apply_window_icon(widget: QWidget) -> None:
    icon = get_app_icon()
    if not icon.isNull():
        widget.setWindowIcon(icon)


def logo_pixmap(size: int = 64) -> QPixmap:
    icon = get_app_icon()
    if icon.isNull():
        return QPixmap()
    return icon.pixmap(size, size)
