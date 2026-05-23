# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for NeuraBreak — macOS one-folder bundle (.app).

Usage (from repo root):
    uv run pyinstaller packaging/macos/neurabreak.spec --noconfirm

Output:
    dist/NeuraBreak.app   <-- drag to /Applications

Then run packaging/macos/build_dmg.sh to produce a distributable DMG.

Notes:
  - UPX is disabled on the main executable — it breaks macOS code signatures
    and is flagged by Gatekeeper on both Intel and Apple Silicon.
  - argv_emulation is False — the tray app does not use drag-and-drop and the
    Apple Event handler installed by argv_emulation conflicts with PySide6.
  - The training data/ directory is intentionally excluded from the bundle.
  - Requires models/neurabreak.onnx — export with `python training/export.py`.
"""

import re as _re
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

# SPECPATH is the directory containing this file (packaging/macos/)
ROOT = Path(SPECPATH).parent.parent  # repo root
SRC  = ROOT / "src" / "neurabreak"

# Version — read from pyproject.toml so it stays in sync
_toml_text   = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
_ver_match   = _re.search(r'^version\s*=\s*"([^"]+)"', _toml_text, _re.MULTILINE)
VERSION      = _ver_match.group(1) if _ver_match else "0.1.0"

# ── ONNX model — required; fail fast if missing (matches Windows behaviour) ─
_onnx_model = ROOT / "models" / "neurabreak.onnx"
if not _onnx_model.exists():
    raise SystemExit(
        "ERROR: macOS packaging requires models/neurabreak.onnx. "
        "Export it with `python training/export.py --format onnx` before building."
    )

# Data files: model, sounds, logo
# NOTE: the training data/ directory is intentionally NOT included here.
datas = [
    (str(_onnx_model), "models"),
]

_sounds_dir = SRC / "ui" / "assets" / "sounds"
if _sounds_dir.exists():
    datas.append((str(_sounds_dir), "neurabreak/ui/assets/sounds"))

_assets_dir = SRC / "ui" / "assets"
for _child in _assets_dir.iterdir():
    if _child.is_file():
        datas.append((str(_child), "neurabreak/ui/assets"))

_logo_svg = ROOT / "assets" / "logo" / "neurabreak_logo.svg"
if _logo_svg.exists():
    datas.append((str(_logo_svg), "assets/logo"))

# onnxruntime data files (schema, config JSON, etc.)
datas += collect_data_files("onnxruntime", include_py_files=False)

# Hidden imports
hidden = [
    # NeuraBreak sub-packages
    "neurabreak",
    "neurabreak.core",
    "neurabreak.core.config",
    "neurabreak.core.events",
    "neurabreak.core.logging",
    "neurabreak.core.runtime_control",
    "neurabreak.core.startup",
    "neurabreak.core.state_machine",
    "neurabreak.core.updater",
    "neurabreak.ai",
    "neurabreak.ai.camera",
    "neurabreak.ai.detection_service",
    "neurabreak.ai.engine",
    "neurabreak.ai.postprocessor",
    "neurabreak.data",
    "neurabreak.data.database",
    "neurabreak.data.journal",
    "neurabreak.data.models",
    "neurabreak.notifications",
    "neurabreak.notifications.audio",
    "neurabreak.notifications.escalation",
    "neurabreak.notifications.manager",
    "neurabreak.notifications.platforms",
    "neurabreak.notifications.platforms.linux",
    "neurabreak.notifications.platforms.macos",
    "neurabreak.notifications.platforms.windows",
    "neurabreak.ui",
    "neurabreak.ui.app",
    "neurabreak.ui.branding",
    "neurabreak.ui.break_screen",
    "neurabreak.ui.dashboard",
    "neurabreak.ui.preview",
    "neurabreak.ui.settings",
    "neurabreak.ui.tray",
    # Core dependencies
    "structlog",
    "pydantic",
    "pydantic.v1",
    "pydantic_core",
    "pydantic_settings",
    "transitions",
    "transitions.extensions",
    # PySide6 / Qt — platform + image plugins PyInstaller often misses
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtNetwork",
    # ONNX Runtime
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi._pybind_state",
    # OpenCV
    "cv2",
    # Audio
    "sounddevice",
    "soundfile",
    "cffi",
    "_cffi_backend",
    # Data layer
    "sqlalchemy",
    "sqlalchemy.dialects.sqlite",
    "sqlalchemy.orm",
    "sqlalchemy.pool",
    # Notifications
    "plyer",
    "plyer.platforms",
    "plyer.platforms.macosx",
    "plyer.platforms.macosx.notification",
    # System / misc
    "psutil",
    "pynput",
    "pynput.keyboard",
    "pynput.mouse",
    "tomllib",
    "tomli_w",
    "urllib.request",
    "urllib.error",
    "json",
    "threading",
    "queue",
]

# Excluded modules: training stack + things not needed at runtime
excluded = [
    "torch",
    "torchvision",
    "torchaudio",
    "ultralytics",
    "wandb",
    "albumentations",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "roboflow",
    "tkinter",
    "unittest",
    "xmlrpc",
    "pydoc",
    "doctest",
    "test",
    "tests",
]

# Icon — fall back gracefully if .icns not yet generated
# Generate with: iconutil -c icns packaging/macos/icon.iconset
_icon_path = ROOT / "packaging" / "macos" / "icon.icns"
_icon = str(_icon_path) if _icon_path.exists() else None

# ── Deduplicate binaries (prevents occasional PySide6/onnxruntime clashes) ───
a = Analysis(
    [str(ROOT / "src" / "neurabreak" / "__main__.py")],
    pathex=[str(ROOT), str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[str(ROOT / "packaging" / "macos" / "hooks")],
    runtime_hooks=[
        str(ROOT / "packaging" / "macos" / "hooks" / "rthook_pyside6.py"),
    ],
    excludes=excluded,
    noarchive=False,
    optimize=1,
)

_seen: set[str] = set()
_unique_bins = []
for _name, _path, _kind in a.binaries:
    _key = _name.lower()
    if _key not in _seen:
        _seen.add(_key)
        _unique_bins.append((_name, _path, _kind))
a.binaries = _unique_bins

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NeuraBreak",
    debug=False,
    strip=False,
    # UPX disabled — breaks code signatures required by Gatekeeper
    upx=False,
    console=False,
    # False: tray app has no drag-and-drop; argv_emulation conflicts with PySide6
    argv_emulation=False,
    target_arch=None,   # matches build machine (x86_64 or arm64)
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="NeuraBreak",
)

app = BUNDLE(
    coll,
    name="NeuraBreak.app",
    icon=_icon,
    bundle_identifier="com.neurabreak.app",
    version=VERSION,
    info_plist={
        "CFBundleName":                          "NeuraBreak",
        "CFBundleDisplayName":                   "NeuraBreak",
        "CFBundleIdentifier":                    "com.neurabreak.app",
        "CFBundleVersion":                       VERSION,
        "CFBundleShortVersionString":            VERSION,
        # LSUIElement = True: app lives in the menu bar / tray, no Dock icon
        "LSUIElement":                           True,
        "LSMinimumSystemVersion":                "12.0",
        "NSHighResolutionCapable":               True,
        "NSCameraUsageDescription": (
            "NeuraBreak uses your camera to monitor posture and detect "
            "when you step away. No video is stored or transmitted."
        ),
        "NSSupportsAutomaticGraphicsSwitching":  True,
    },
)
