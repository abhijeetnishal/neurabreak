"""macOS-native notifications via NSUserNotification / UNUserNotification."""

from __future__ import annotations

import structlog

log = structlog.get_logger()


def send_toast(title: str, message: str) -> None:
    """Show a macOS notification in Notification Centre."""
    # TODO:
    # from Foundation import NSUserNotification, NSUserNotificationCenter
    # notification = NSUserNotification.alloc().init()
    # notification.setTitle_(title)
    # notification.setInformativeText_(message)
    # center = NSUserNotificationCenter.defaultUserNotificationCenter()
    # center.deliverNotification_(notification)
