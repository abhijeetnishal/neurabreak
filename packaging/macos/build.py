"""
macOS app bundle builder — PyInstaller.

Produces: dist/NeuraBreak.app  (drag to /Applications)

Usage (run from repo root):
    python packaging/macos/build.py

Prerequisites:
    uv sync --extra packaging    # installs pyinstaller
    python training/export.py --format onnx   # produces models/neurabreak.onnx
    # On macOS only — requires Xcode Command Line Tools
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SPEC = ROOT / "packaging" / "macos" / "neurabreak.spec"


def main() -> None:
    if sys.platform != "darwin":
        print("ERROR: macOS build must run on macOS.")
        sys.exit(1)

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyinstaller>=6.3.0"],
            check=True,
        )

    cmd = [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm"]
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print("PyInstaller build failed.")
        sys.exit(result.returncode)

    app_path = ROOT / "dist" / "NeuraBreak.app"
    if app_path.exists():
        print(f"\nBuild successful: {app_path}")
        print("Next step: run packaging/macos/build_dmg.sh to create a distributable DMG.")
    else:
        print("\nBuild may have succeeded; check dist/")


if __name__ == "__main__":
    main()
