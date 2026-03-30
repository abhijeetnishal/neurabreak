"""
macOS app bundle builder using Python-Briefcase.

Produces: dist/NeuraBreak.app (can be dragged to Applications)

Usage:
    python packaging/macos/build.py

Prerequisites:
    pip install briefcase
    # On macOS only — requires Xcode Command Line Tools
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def main() -> None:
    if sys.platform != "darwin":
        print("ERROR: macOS build must run on macOS.")
        sys.exit(1)

    try:
        import briefcase  # noqa: F401
    except ImportError:
        print("Installing briefcase...")
        subprocess.run([sys.executable, "-m", "pip", "install", "briefcase"], check=True)

    # briefcase reads pyproject.toml [tool.briefcase] section
    # Run from repo root so it picks up pyproject.toml
    commands = [
        ["briefcase", "create", "macOS"],
        ["briefcase", "build", "macOS"],
    ]

    for cmd in commands:
        print(f"\n$ {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            sys.exit(result.returncode)

    app_path = ROOT / "dist" / "macOS" / "app" / "NeuraBreak" / "NeuraBreak.app"
    if app_path.exists():
        print(f"\nBuild successful: {app_path}")
        print("To create a DMG: briefcase package macOS --update-support")
    else:
        print("\nBuild may have succeeded; check dist/macOS/")


if __name__ == "__main__":
    main()
