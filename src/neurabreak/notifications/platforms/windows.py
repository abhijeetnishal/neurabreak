"""Windows-native toast notifications.

Tries windows-toasts first (Action Center integration), then falls back
to plyer which uses win32 balloon tips. Both are optional — if neither
works, we just log a warning and move on.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger()


def send_toast(title: str, message: str, icon_path: str | None = None) -> None:
    """Show a Windows desktop notification.

    Args:
        title:     Notification title.
        message:   Notification body text.
        icon_path: Optional path to a .png icon (used when supported).
    """
    if _send_via_windows_toasts(title, message):
        return
    if _send_via_plyer(title, message):
        return
    log.warning("toast_all_methods_failed", title=title)


def _send_via_windows_toasts(title: str, message: str) -> bool:
    try:
        from windows_toasts import Toast, WindowsToaster  # type: ignore

        toaster = WindowsToaster("NeuraBreak")
        toast = Toast()
        toast.text_fields = [title, message]
        toaster.show_toast(toast)
        return True
    except ImportError:
        return False
    except Exception as e:
        log.warning("windows_toasts_failed", error=str(e))
        return False


def _send_via_plyer(title: str, message: str) -> bool:
    try:
        from plyer import notification  # type: ignore

        notification.notify(
            title=title,
            message=message,
            app_name="NeuraBreak",
            timeout=8,
        )
        return True
    except ImportError:
        return False
    except Exception as e:
        log.warning("plyer_toast_failed", error=str(e))
        return False
