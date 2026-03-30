"""Auto-update checker using GitHub Releases.

Runs once per app session in a background daemon thread — never blocks the UI.
If a newer tag is found on GitHub it fires `UPDATE_AVAILABLE` on the event bus.
The tray icon listens for that event and shows a notification with a download link.

Rate limits / failures are swallowed silently — a missing update check should
never crash the app or annoy the user.

Design notes:
  - Uses only stdlib (urllib) — no requests dependency.
  - Compares versions as tuples of ints so "1.10.0" > "1.9.0" works correctly.
  - Has a hard 5-second timeout on the network call.
  - Respects a session-level flag so we only ever check once per run.
"""

from __future__ import annotations

import json
import re
import threading
from urllib.error import URLError
from urllib.request import Request, urlopen

import structlog

from neurabreak import __version__
from neurabreak.core.events import Event, EventType, bus

log = structlog.get_logger()

# Change this to your actual GitHub username/repo once published.
_GITHUB_REPO = "abhijeetnishal/neurabreak"
_API_URL = f"https://api.github.com/repos/{_GITHUB_REPO}/releases/latest"
_RELEASES_URL = f"https://github.com/{_GITHUB_REPO}/releases/latest"

_TIMEOUT_SECS = 5
_check_done = False  # guard: run at most once per process
_check_lock = threading.Lock()  # protect _check_done check-then-set


def _parse_version(tag: str) -> tuple[int, ...]:
    """Turn 'v1.2.3' or '1.2.3' into (1, 2, 3). Unknown parts default to 0."""
    tag = tag.lstrip("v")
    parts = re.findall(r"\d+", tag)[:3]
    return tuple(int(p) for p in parts)


def _fetch_latest_release() -> dict | None:
    """Hit GitHub API and return the parsed JSON, or None on any failure."""
    req = Request(
        _API_URL,
        headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": f"NeuraBreak/{__version__}",
        },
    )
    try:
        with urlopen(req, timeout=_TIMEOUT_SECS) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode())
    except (URLError, OSError, ValueError):
        return None


def _do_check() -> None:
    global _check_done
    with _check_lock:
        if _check_done:
            return
        _check_done = True

    data = _fetch_latest_release()
    if not data:
        log.debug("update_check_skipped", reason="no_response")
        return

    tag = data.get("tag_name", "")
    if not tag:
        return

    latest = _parse_version(tag)
    current = _parse_version(__version__)

    log.debug("update_check_result", current=__version__, latest=tag)

    if latest > current:
        html_url = data.get("html_url", _RELEASES_URL)
        body = data.get("body", "")
        # Truncate release notes so the tooltip stays readable
        notes = body.strip()[:300] if body else ""
        bus.publish(
            Event(
                EventType.UPDATE_AVAILABLE,
                {
                    "version": tag.lstrip("v"),
                    "tag": tag,
                    "url": html_url,
                    "notes": notes,
                },
            )
        )


def check_for_updates_async() -> None:
    """Spawn a daemon thread to check for updates without blocking startup."""
    t = threading.Thread(target=_do_check, name="update-checker", daemon=True)
    t.start()
