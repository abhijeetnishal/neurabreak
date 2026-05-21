import sys
from pathlib import Path

ROOT   = Path(SPECPATH).parent.parent
SRC    = ROOT / "src" / "neurabreak"
ASSETS = ROOT / "assets"
DATA   = ROOT / "data"
MODEL  = ROOT / "yolo26n.pt"

block_cipher = None

a = Analysis(
    [str(ROOT / "src" / "neurabreak" / "__main__.py")],
    pathex=[str(ROOT), str(ROOT / "src")],
    binaries=[],
    datas=[
        (str(MODEL),  "models"),
        (str(ASSETS), "assets"),
        (str(DATA),   "data"),
    ],
    hiddenimports=[
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "sqlalchemy.dialects.sqlite",
        "sqlalchemy.pool",
        "ultralytics",
        "ultralytics.nn.tasks",
        "ultralytics.utils",
        "ultralytics.models.yolo",
        "cv2",
        "sounddevice",
        "soundfile",
        "pydantic",
        "pydantic.v1",
        "tomllib",
        "tomli_w",
        "structlog",
        "neurabreak.notifications.platforms.macos",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython"],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NeuraBreak",
    debug=False,
    strip=False,
    upx=True,
    console=False,
    argv_emulation=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="NeuraBreak",
)

app = BUNDLE(
    coll,
    name="NeuraBreak.app",
    icon=None,
    bundle_identifier="com.neurabreak.app",
    version="1.0.0",
    info_plist={
        "CFBundleName":               "NeuraBreak",
        "CFBundleDisplayName":        "NeuraBreak",
        "CFBundleIdentifier":         "com.neurabreak.app",
        "CFBundleVersion":            "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "LSUIElement":                True,
        "LSMinimumSystemVersion":     "12.0",
        "NSHighResolutionCapable":    True,
        "NSCameraUsageDescription":
            "NeuraBreak uses your camera to monitor posture and detect "
            "when you step away. No video is stored or transmitted.",
        "NSSupportsAutomaticGraphicsSwitching": True,
    },
)
