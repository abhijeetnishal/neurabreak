"""Entry point — run with: python -m neurabreak"""

import sys


def main() -> int:
    # Handle special flags injected by the installer / autostart registry entry
    args = sys.argv[1:]

    # --quit: ask any running instance to exit (used by the uninstaller).
    # We just exit immediately — the uninstaller kills the process next anyway.
    if "--quit" in args:
        return 0

    # --minimized: passed by the Windows autostart registry entry.
    # Stored as a global flag; app.py reads it when building the tray.
    start_minimized_override = "--minimized" in args

    # Set up logging before touching anything else so we catch early failures
    from neurabreak.core.logging import setup_logging

    setup_logging()

    import structlog

    log = structlog.get_logger()
    log.info("neurabreak_starting", version="0.1.0", minimized=start_minimized_override)

    try:
        from neurabreak.core.config import ConfigManager

        config_manager = ConfigManager.load()
        if start_minimized_override:
            config_manager.config.ui.start_minimized = True
    except Exception as e:
        import structlog

        structlog.get_logger().error("config_load_failed", error=str(e))
        return 1

    try:
        from neurabreak.ui.app import NeuraBreakApp

        app = NeuraBreakApp(config_manager)
        return app.run()
    except ImportError as e:
        import structlog

        structlog.get_logger().error(
            "ui_unavailable",
            error=str(e),
            hint="Make sure PySide6 is installed: pip install PySide6",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
