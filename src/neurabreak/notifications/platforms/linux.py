"""Linux desktop notifications via libnotify / D-Bus."""

from __future__ import annotations

import structlog

log = structlog.get_logger()


def send_toast(title: str, message: str, urgency: str = "normal") -> None:
    """Show a desktop notification via libnotify.

    Args:
        title:   Notification title.
        message: Notification body.
        urgency: "low" | "normal" | "critical"
    """
    try:
        import notify2  # type: ignore[import-untyped]

        notify2.init("NeuraBreak")
        n = notify2.Notification(title, message)
        n.set_urgency({"low": 0, "normal": 1, "critical": 2}.get(urgency, 1))
        n.show()
    except Exception as e:
        log.warning("linux_toast_failed", error=str(e))


