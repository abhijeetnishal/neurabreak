"""Windows startup registry management.

Allows NeuraBreak to optionally launch when Windows starts, entirely via the
per-user Run key — no admin rights required. On non-Windows platforms every
function is a no-op so the rest of the codebase can call this module safely.

Registry key:
    HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run
    value name: "NeuraBreak"
    value data: "<path to exe>" --minimized
"""

from __future__ import annotations

import sys
from pathlib import Path

import structlog

log = structlog.get_logger()

_REG_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_VALUE_NAME = "NeuraBreak"


def is_windows() -> bool:
    return sys.platform == "win32"


def get_startup_exe_path() -> str:
    """Return the path that should be written to the registry.

    When running as a PyInstaller bundle, sys.executable is the .exe we want.
    In dev mode we build a wrapper: 'pythonw.exe -m neurabreak --minimized'.
    """
    if getattr(sys, "frozen", False):
        # Packaged — sys.executable is NeuraBreak.exe
        return f'"{sys.executable}" --minimized'
    # Dev mode — use pythonw so no console window flashes
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        pythonw = Path(sys.executable)
    return f'"{pythonw}" -m neurabreak --minimized'


def is_startup_enabled() -> bool:
    """Return True if NeuraBreak is registered to run at Windows login."""
    if not is_windows():
        return False
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _REG_KEY, 0, winreg.KEY_READ) as key:
            value, _ = winreg.QueryValueEx(key, _VALUE_NAME)
            return bool(value)
    except FileNotFoundError:
        return False
    except OSError as exc:
        log.warning("startup_read_failed", error=str(exc))
        return False


def enable_startup() -> bool:
    """Write the autostart registry entry. Returns True on success."""
    if not is_windows():
        return False
    try:
        import winreg
        cmd = get_startup_exe_path()
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _REG_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.SetValueEx(key, _VALUE_NAME, 0, winreg.REG_SZ, cmd)
        log.info("startup_enabled", command=cmd)
        return True
    except OSError as exc:
        log.error("startup_enable_failed", error=str(exc))
        return False


def disable_startup() -> bool:
    """Remove the autostart registry entry. Returns True on success."""
    if not is_windows():
        return False
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _REG_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.DeleteValue(key, _VALUE_NAME)
        log.info("startup_disabled")
        return True
    except FileNotFoundError:
        # Already gone — that's fine
        return True
    except OSError as exc:
        log.error("startup_disable_failed", error=str(exc))
        return False


def set_startup(enabled: bool) -> bool:
    """Toggle the autostart entry on or off."""
    return enable_startup() if enabled else disable_startup()
