"""Login-item / autostart management for Windows and macOS.

Windows: per-user HKCU Run registry key — no admin rights required.
macOS:   per-user LaunchAgent plist in ~/Library/LaunchAgents/ — no admin
         rights required.
Other platforms: every function is a no-op so callers need no platform guards.

Windows registry key:
    HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run
    value name: "NeuraBreak"
    value data: "<path to exe>" --minimized

macOS LaunchAgent plist:
    ~/Library/LaunchAgents/com.neurabreak.app.plist
"""

from __future__ import annotations

import sys
from pathlib import Path

import structlog

log = structlog.get_logger()

# Windows constants
_REG_KEY    = r"Software\Microsoft\Windows\CurrentVersion\Run"
_VALUE_NAME = "NeuraBreak"

# macOS constants
_LAUNCH_AGENT_DIR = Path.home() / "Library" / "LaunchAgents"
_PLIST_LABEL      = "com.neurabreak.app"
_PLIST_PATH       = _LAUNCH_AGENT_DIR / f"{_PLIST_LABEL}.plist"


# Platform guards

def is_windows() -> bool:
    return sys.platform == "win32"


def is_macos() -> bool:
    return sys.platform == "darwin"


# Windows helpers

def get_startup_exe_path() -> str:
    """Return the command string written to the Windows registry.

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


# macOS helpers

def _macos_exe_args() -> list[str]:
    """Return the argv list for the LaunchAgent ProgramArguments key."""
    if getattr(sys, "frozen", False):
        # Inside .app bundle — sys.executable is .../Contents/MacOS/NeuraBreak
        return [sys.executable, "--minimized"]
    # Dev mode: run via current interpreter
    return [sys.executable, "-m", "neurabreak", "--minimized"]


def _macos_plist_xml() -> str:
    """Render a minimal LaunchAgent plist that starts NeuraBreak at login."""
    args_xml = "\n".join(
        f"        <string>{arg}</string>" for arg in _macos_exe_args()
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"\n'
        '    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        '<dict>\n'
        '    <key>Label</key>\n'
        f'    <string>{_PLIST_LABEL}</string>\n'
        '    <key>ProgramArguments</key>\n'
        '    <array>\n'
        f'{args_xml}\n'
        '    </array>\n'
        '    <key>RunAtLoad</key>\n'
        '    <true/>\n'
        '    <key>KeepAlive</key>\n'
        '    <false/>\n'
        '</dict>\n'
        '</plist>\n'
    )


# Public API

def is_startup_enabled() -> bool:
    """Return True if NeuraBreak is registered to run at login."""
    if is_windows():
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
    if is_macos():
        return _PLIST_PATH.exists()
    return False


def enable_startup() -> bool:
    """Enable autostart at login. Returns True on success."""
    if is_windows():
        try:
            import winreg
            cmd = get_startup_exe_path()
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, _REG_KEY, 0, winreg.KEY_SET_VALUE
            ) as key:
                winreg.SetValueEx(key, _VALUE_NAME, 0, winreg.REG_SZ, cmd)
                
            return True
        except OSError as exc:
            log.error("startup_enable_failed", error=str(exc))
            return False
    if is_macos():
        try:
            _LAUNCH_AGENT_DIR.mkdir(parents=True, exist_ok=True)
            _PLIST_PATH.write_text(_macos_plist_xml(), encoding="utf-8")

            return True
        except OSError as exc:
            log.error("macos_startup_enable_failed", error=str(exc))
            return False
    return False


def disable_startup() -> bool:
    """Disable autostart at login. Returns True on success."""
    if is_windows():
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
    if is_macos():
        try:
            if _PLIST_PATH.exists():
                _PLIST_PATH.unlink()
            log.info("macos_startup_disabled")
            return True
        except OSError as exc:
            log.error("macos_startup_disable_failed", error=str(exc))
            return False
    return False


def set_startup(enabled: bool) -> bool:
    """Toggle the autostart entry on or off."""
    return enable_startup() if enabled else disable_startup()
