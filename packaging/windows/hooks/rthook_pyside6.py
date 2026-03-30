# PyInstaller runtime hook for PySide6.
#
# Runs before any app code at process startup when the frozen bundle executes.
# Adds the Qt plugins directory to QT_PLUGIN_PATH so PySide6 can locate
# platform/style/imageformat plugins that live inside the _MEIPASS bundle.

import os
import sys

# Runtime hooks run at module-level by design; code here executes before main.
if getattr(sys, "frozen", False):
    _bundle_dir = getattr(sys, "_MEIPASS", None)
    if _bundle_dir is not None:
        _plugins_dir = os.path.join(_bundle_dir, "PySide6", "Qt", "plugins")
        if not os.path.isdir(_plugins_dir):
            # Alternate location used by some PySide6 wheel layouts
            _plugins_dir = os.path.join(_bundle_dir, "PySide6", "plugins")

        if os.path.isdir(_plugins_dir):
            os.environ["QT_PLUGIN_PATH"] = _plugins_dir

        # Point QML to its bundled import path if we ever add QML
        _qml_dir = os.path.join(_bundle_dir, "PySide6", "qml")
        if os.path.isdir(_qml_dir):
            os.environ.setdefault("QML2_IMPORT_PATH", _qml_dir)

        # Silence spurious high-DPI warnings that appear in some Qt builds
        os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
