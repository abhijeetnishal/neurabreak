"""Structured logging setup for NeuraBreak.

Uses structlog for human-readable console output during development.
The setup is intentionally simple — no log rotation, no remote sinks.
Everything goes to stdout and can be piped/captured by the OS.
"""

import logging
import sys

import structlog


def _pick_output_stream() -> object | None:
    """Return a writable stream when available.

    In windowed PyInstaller builds (console=False), both stdout and stderr can
    be None. Returning None lets setup fall back to a NullHandler safely.
    """

    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "write"):
            return stream
    return None


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog. Call this once at startup before anything else."""

    # Processors shared between structlog and stdlib logging
    shared_processors: list[structlog.types.Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.contextvars.merge_contextvars,
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Use colors if we're on an interactive terminal, plain text otherwise.
    # On Windows, colors require colorama; it is listed as a dependency but we
    # guard with a try/import so a missing install degrades gracefully.
    stream = _pick_output_stream()

    renderer: structlog.types.Processor
    _want_colors = False
    isatty = getattr(stream, "isatty", None)
    if callable(isatty):
        try:
            _want_colors = bool(isatty())
        except Exception:
            _want_colors = False

    if _want_colors and sys.platform == "win32":
        try:
            import colorama  # noqa: F401
            _want_colors = True
        except ImportError:
            _want_colors = False
    renderer = structlog.dev.ConsoleRenderer(colors=_want_colors)

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    root = logging.getLogger()
    root.handlers.clear()

    if stream is None:
        root.addHandler(logging.NullHandler())
    else:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for noisy in ("PIL", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
