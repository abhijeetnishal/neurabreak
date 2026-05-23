"""Runtime hook: configure Qt plugin paths inside the frozen macOS bundle.

PyInstaller bundles PySide6's Qt plugins (including the 'cocoa' platform
plugin) into the _MEIPASS directory, but the Qt loader still needs to be
told where to find them at runtime.  Without this hook the app crashes on
launch with:
    "could not find or load the Qt platform plugin 'cocoa'"
"""

import os
import sys

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _bundle = sys._MEIPASS

    # Primary Qt plugin tree bundled by PyInstaller
    _plugins = os.path.join(_bundle, "PySide6", "Qt", "plugins")
    if os.path.isdir(_plugins):
        os.environ.setdefault("QT_PLUGIN_PATH", _plugins)

    # Explicit platform-plugin path so Qt finds libqcocoa.dylib
    _platforms = os.path.join(_plugins, "platforms")
    if os.path.isdir(_platforms):
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", _platforms)
