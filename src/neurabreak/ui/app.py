"""QApplication wrapper — the top-level app object.

Sets up Qt, initialises all services (camera, inference, notifications,
journal), wires their events to the tray icon and break screen, then
hands control to the Qt event loop.

Threading model:
  - Qt main thread:  UI (tray, break screen, settings, dashboard)
  - InferenceThread: camera capture → model inference → state machine
  - EscalationMonitor thread: notification escalation polling
  - Audio threads:   short-lived daemon threads per sound playback
"""

from __future__ import annotations

import sys
from pathlib import Path

import structlog
from PySide6.QtCore import QEvent, QObject

from neurabreak.core.config import ConfigManager, DB_FILE
from neurabreak.ui.branding import get_app_icon

log = structlog.get_logger()

# Assets directory relative to this file (ui/assets/)
_ASSETS_DIR = Path(__file__).parent / "assets"


class _QuitEventGuard(QObject):
    """Block unexpected Qt quit events unless explicitly allowed.

    On Windows, tray interactions may occasionally inject a stray quit event.
    This guard prevents accidental app shutdown and only lets quit events pass
    when triggered through the tray's explicit Quit action.
    """

    def __init__(self, parent: object) -> None:
        super().__init__(parent)
        self._allow_next_quit = False

    def allow_next_quit(self) -> None:
        self._allow_next_quit = True

    def reset(self) -> None:
        self._allow_next_quit = False

    def eventFilter(self, watched: object, event: object) -> bool:  # noqa: N802
        if isinstance(event, QEvent) and event.type() == QEvent.Type.Quit:
            if self._allow_next_quit:
                self._allow_next_quit = False
                return False
            log.warning("unexpected_quit_event_blocked")
            return True
        return False


class NeuraBreakApp:
    """Owns the Qt application lifetime and wires all services together."""

    def __init__(self, config_manager: ConfigManager) -> None:
        self.config_manager = config_manager
        self._qt_app = None
        self._tray = None
        self._detection_service = None
        self._notification_manager = None
        self._break_screen = None
        self._journal = None
        self._db = None
        self._quit_guard = None
        self._explicit_quit_requested = False

    def run(self) -> int:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import QApplication, QSystemTrayIcon

        # Performance: limit PyTorch CPU threads before any model loads
        try:
            import torch
            torch.set_num_threads(4)
            torch.set_num_interop_threads(2)
        except ImportError:
            pass

        # Disable Ultralytics telemetry and update checks (eliminates outbound network calls during inference).
        try:
            import os as _os
            _os.environ.setdefault("YOLO_VERBOSE", "False")
            from ultralytics.utils import SETTINGS as _YOLO_SETTINGS
            _YOLO_SETTINGS.update({"sync": False})
        except Exception:
            pass

        self._qt_app = QApplication.instance() or QApplication(sys.argv)
        self._qt_app.setQuitOnLastWindowClosed(False)
        self._qt_app.setApplicationName("NeuraBreak")
        self._qt_app.setApplicationVersion("0.1.0")
        app_icon = get_app_icon()
        if not app_icon.isNull():
            self._qt_app.setWindowIcon(app_icon)
        self._qt_app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        self._quit_guard = _QuitEventGuard(self._qt_app)
        self._qt_app.installEventFilter(self._quit_guard)

        if not QSystemTrayIcon.isSystemTrayAvailable():
            log.error("no_system_tray", message="System tray not available on this platform.")
            return 1

        config = self.config_manager.config

        # Auto-tune FPS: drop to 3 on CPU-only to halve CPU load vs the 5 FPS default.
        try:
            from neurabreak.ai.engine import select_best_device
            _effective_device = select_best_device(
                getattr(config.detection, "device", "auto")
            )
            if _effective_device == "cpu" and config.detection.fps > 3:
                log.info(
                    "fps_auto_reduced_for_cpu",
                    from_fps=config.detection.fps,
                    to_fps=3,
                    hint="Install CUDA/MPS PyTorch to remove this limit",
                )
                config.detection.fps = 3
        except Exception:
            pass

        # Database & journal
        journal = None
        try:
            from neurabreak.data.database import Database
            from neurabreak.data.journal import HealthJournalService

            self._db = Database(DB_FILE)
            self._db.connect()
            journal = HealthJournalService(self._db)
            self._journal = journal
        except ImportError:
            log.warning(
                "journal_unavailable",
                hint="Run `uv sync --extra data` to enable the health journal.",
            )

        from neurabreak.notifications.audio import AudioManager

        audio = AudioManager(
            assets_dir=_ASSETS_DIR,
            volume=config.audio.volume,
            enabled=config.audio.enabled,
        )

        from neurabreak.notifications.manager import NotificationManager

        self._notification_manager = NotificationManager(config=config, audio_manager=audio)

        from neurabreak.core.state_machine import PostureStateMachine

        state_machine = PostureStateMachine(
            fps=config.detection.fps,
            break_interval_min=config.breaks.interval_min,
            break_duration_min=config.breaks.duration_min,
            posture_alert_sec=config.breaks.posture_alert_sec,
            smart_pause_sec=config.breaks.smart_pause_sec,
            eye_break_interval_min=config.breaks.eye_break_interval_min,
        )

        # Tray icon
        from neurabreak.ui.tray import NeuraBreakTray

        self._tray = NeuraBreakTray(
            config_manager=self.config_manager,
            app=self._qt_app,
            journal=journal,
            audio_manager=audio,
            on_quit_requested=self._request_app_quit,
        )
        self._tray.show()

        # Break screen
        from neurabreak.ui.break_screen import BreakScreen

        self._break_screen = BreakScreen(config=config)

        def _show_break_screen(message: str) -> None:
            QTimer.singleShot(0, lambda: self._break_screen.show_break(
                state_machine=state_machine,
                notif_manager=self._notification_manager,
            ))

        self._notification_manager.on_level3(_show_break_screen)

        def _show_level2_balloon(message: str) -> None:
            if self._tray:
                _msg = message
                QTimer.singleShot(0, lambda: self._tray._icon.showMessage(
                    "NeuraBreak Reminder",
                    _msg,
                    QSystemTrayIcon.MessageIcon.Warning,
                    8000,
                ))

        self._notification_manager.on_level2(_show_level2_balloon)

        def _show_eye_break_balloon(message: str) -> None:
            """Compact eye-rest overlay + tray tip for the 20-20-20 eye break."""
            _msg = message
            if self._break_screen is not None:
                duration = getattr(self.config_manager.config.breaks, "eye_break_duration_sec", 20)
                QTimer.singleShot(0, lambda: self._break_screen.show_eye_break(duration))
            if self._tray:
                QTimer.singleShot(0, lambda: self._tray._icon.showMessage(
                    "\U0001f441\ufe0f Eye Break",
                    _msg,
                    QSystemTrayIcon.MessageIcon.Information,
                    6000,
                ))

        self._notification_manager.on_eye_break(_show_eye_break_balloon)

        self._break_screen.on_break_taken(lambda: audio.play_builtin("break_end"))

        from neurabreak.ai.camera import FrameCaptureService
        from neurabreak.ai.detection_service import DetectionService
        from neurabreak.ai.engine import InferenceEngine

        camera = FrameCaptureService(
            camera_index=0,
            fps=config.detection.fps,
        )
        engine = InferenceEngine(
            model_path=config.detection.model_path,
            confidence_threshold=config.detection.confidence_threshold,
            model_variant=config.detection.model_variant,
            device=getattr(config.detection, "device", "auto"),
            use_half=getattr(config.detection, "use_half", True),
            imgsz=getattr(config.detection, "imgsz", 320),
        )

        self._detection_service = DetectionService(
            camera=camera,
            engine=engine,
            state_machine=state_machine,
            config=config,
            journal=journal,
        )

        from neurabreak.core.events import Event, EventType, bus

        def _apply_runtime_config(event: Event) -> None:
            updated_cfg = event.data.get("config")
            if updated_cfg is None:
                updated_cfg = self.config_manager.config

            self.config_manager._config = updated_cfg
            self._detection_service.config = updated_cfg
            self._notification_manager.apply_config(updated_cfg)
            state_machine.apply_runtime_config(
                break_interval_min=updated_cfg.breaks.interval_min,
                break_duration_min=updated_cfg.breaks.duration_min,
                posture_alert_sec=updated_cfg.breaks.posture_alert_sec,
                smart_pause_sec=updated_cfg.breaks.smart_pause_sec,
                eye_break_interval_min=updated_cfg.breaks.eye_break_interval_min,
            )

            audio.enabled = updated_cfg.audio.enabled
            audio.set_volume(updated_cfg.audio.volume)

            if self._break_screen is not None:
                self._break_screen.config = updated_cfg

        bus.subscribe(EventType.CONFIG_CHANGED, _apply_runtime_config)

        # Wire live-preview frames: the inference thread calls this closure
        # which routes the frame to the preview window (if open) on the main thread.
        _tray_ref = self._tray

        def _preview_sink(frame, boxes: list) -> None:
            preview = _tray_ref.get_preview_window()
            if preview is not None:
                preview.update_frame(frame, boxes)

        self._detection_service.set_frame_sink(_preview_sink)

        # Journal session lifecycle via event bus
        if journal:
            from neurabreak.core.events import Event, EventType, bus

            bus.subscribe(
                EventType.SESSION_STARTED,
                lambda _: journal.start_session(),
            )
            bus.subscribe(
                EventType.SESSION_ENDED,
                lambda _: journal.end_session(),
            )
            # Also end the session when smart-pause kicks in (person stepped away).
            # SESSION_ENDED is only published on explicit session end; SESSION_PAUSED
            # is the common path, so we close the journal record here too.
            bus.subscribe(
                EventType.SESSION_PAUSED,
                lambda _: journal.end_session(),
            )
            # Resume journal session when smart-pause ends and monitoring restarts.
            bus.subscribe(
                EventType.SESSION_RESUMED,
                lambda _: journal.start_session(),
            )

            # Break lifecycle — track triggered vs taken for the compliance chart
            _active_break_id: list[int] = [-1]

            def _on_break_due_journal(event: Event) -> None:
                reason = event.data.get("reason", "break_interval")
                _active_break_id[0] = journal.record_break(reason)

            def _on_break_started_journal(_: Event) -> None:
                journal.mark_break_taken(_active_break_id[0])

            def _on_break_ended_journal(_: Event) -> None:
                journal.mark_break_ended(_active_break_id[0])
                _active_break_id[0] = -1

            bus.subscribe(EventType.BREAK_DUE, _on_break_due_journal)
            bus.subscribe(EventType.BREAK_STARTED, _on_break_started_journal)
            bus.subscribe(EventType.BREAK_ENDED, _on_break_ended_journal)


        # Hide the break screen when the user steps away (smart-pause) or when
        # the break ends — avoids the overlay lingering after state transitions.
        from neurabreak.core.events import EventType, bus as _bus

        _break_screen_ref = self._break_screen

        def _hide_break_screen_on_pause(_: object) -> None:                                                     
            if _break_screen_ref is not None and _break_screen_ref.isVisible():
                QTimer.singleShot(0, _break_screen_ref.hide)

        _bus.subscribe(EventType.SESSION_PAUSED, _hide_break_screen_on_pause)
        _bus.subscribe(EventType.BREAK_ENDED, _hide_break_screen_on_pause)

        # Start services after Qt event loop is pumping
        QTimer.singleShot(500, self._start_services)

        self._qt_app.aboutToQuit.connect(self._on_about_to_quit)

        return self._run_main_loop()

    def _run_main_loop(self) -> int:
        if self._qt_app is None:
            return 1

        unexpected_quit_count = 0
        while True:
            exit_code = self._qt_app.exec()

            if self._explicit_quit_requested:
                self._on_quit()
                return exit_code

            unexpected_quit_count += 1
            log.warning(
                "qt_event_loop_exit_unexpected",
                restart_attempt=unexpected_quit_count,
            )

            if unexpected_quit_count >= 3:
                log.error(
                    "qt_event_loop_exit_repeated",
                    action="shutting_down",
                )
                self._on_quit()
                return exit_code

            if self._quit_guard is not None:
                self._quit_guard.reset()

    def _start_services(self) -> None:
        self._detection_service.start()
        self._notification_manager.start()

        # Kick off the update check in a background thread, fails silently if there's no internet connection.
        try:
            from neurabreak.core.updater import check_for_updates_async
            check_for_updates_async()
        except Exception:
            pass

    def _on_quit(self) -> None:
        log.info("app_shutting_down")
        if self._detection_service:
            self._detection_service.stop()
        if self._journal:
            try:
                self._journal.end_session()
            except Exception:
                pass
        if self._notification_manager:
            self._notification_manager.stop()
        if self._db:
            self._db.disconnect()

    def _on_about_to_quit(self) -> None:
        log.info(
            "qt_about_to_quit",
            explicit=self._explicit_quit_requested,
        )

    def _request_app_quit(self) -> None:
        if self._qt_app is None:
            return
        self._explicit_quit_requested = True
        if self._quit_guard is not None:
            self._quit_guard.allow_next_quit()
        self._qt_app.quit()

