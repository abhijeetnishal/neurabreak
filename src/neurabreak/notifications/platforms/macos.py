"""macOS-native notifications.

Resolution order:
  1. pyobjc UNUserNotificationCenter  (macOS 10.14+ — modern, sandboxed API)
  2. osascript display notification    (always available, zero extra deps)
  3. plyer                             (cross-platform fallback)
"""

from __future__ import annotations

import subprocess

import structlog

log = structlog.get_logger()


def send_toast(title: str, message: str) -> None:
    """Show a macOS notification in Notification Centre."""
    if _send_via_pyobjc(title, message):
        return
    if _send_via_osascript(title, message):
        return
    if _send_via_plyer(title, message):
        return
    log.warning("macos_toast_all_methods_failed", title=title)


def _send_via_pyobjc(title: str, message: str) -> bool:
    """Use UNUserNotificationCenter via pyobjc (macOS 10.14+, recommended)."""
    try:
        import uuid

        from UserNotifications import (  # type: ignore[import-untyped]
            UNMutableNotificationContent,
            UNNotificationRequest,
            UNUserNotificationCenter,
        )

        center = UNUserNotificationCenter.currentNotificationCenter()
        content = UNMutableNotificationContent.alloc().init()
        content.setTitle_(title)
        content.setBody_(message)
        request = UNNotificationRequest.requestWithIdentifier_content_trigger_(
            str(uuid.uuid4()), content, None
        )
        center.addNotificationRequest_withCompletionHandler_(request, None)
        return True
    except ImportError:
        return False
    except Exception as exc:
        log.warning("pyobjc_toast_failed", error=str(exc))
        return False


def _send_via_osascript(title: str, message: str) -> bool:
    """Use AppleScript — always available on macOS, no extra dependencies."""
    try:
        # Sanitise to prevent AppleScript injection: strip/escape double-quotes
        safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
        safe_msg = message.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{safe_msg}" with title "{safe_title}"',
            ],
            check=False,
            capture_output=True,
            timeout=5,
        )
        return True
    except Exception as exc:
        log.warning("osascript_toast_failed", error=str(exc))
        return False


def _send_via_plyer(title: str, message: str) -> bool:
    try:
        from plyer import notification  # type: ignore[import-untyped]

        notification.notify(
            title=title,
            message=message,
            app_name="NeuraBreak",
            timeout=8,
        )
        return True
    except ImportError:
        return False
    except Exception as exc:
        log.warning("plyer_toast_failed", error=str(exc))
        return False
