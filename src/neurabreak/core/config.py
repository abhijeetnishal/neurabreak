"""Configuration management for NeuraBreak.

Config lives at ~/.neurabreak/config.toml and is auto-created with sensible
defaults on first launch. The user gets a commented TOML file they can edit
directly — changes are hot-reloaded without a restart.

Schema is enforced by Pydantic v2, so bad values are caught at load time
with clear error messages rather than failures at runtime.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel, Field, field_validator, model_validator

log = structlog.get_logger()

# Standard XDG-ish config location
CONFIG_DIR = Path.home() / ".neurabreak"
CONFIG_FILE = CONFIG_DIR / "config.toml"
DB_FILE = CONFIG_DIR / "neurabreak.db"
PLUGINS_DIR = CONFIG_DIR / "plugins"


class DetectionConfig(BaseModel):
    fps: int = Field(default=5, ge=1, le=30, description="Webcam capture rate")
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    consecutive_frames: int = Field(
        default=10,
        ge=1,
        description=(
            "Number of consecutive inference frames the same bad-posture class must "
            "appear before it is forwarded to the state machine. Acts as a debounce "
            "filter — useful for smaller models (nano/small) that may flicker between "
            "classes on consecutive frames. Good posture clears the debounce immediately "
            "so posture correction is felt instantly. At 5 FPS, default=10 means a 2-second "
            "confirmation window before the posture-alert countdown begins."
        ),
    )
    model_variant: Literal["nano", "small", "medium"] = "nano"
    model_path: str = ""  # empty = use the bundled model

    # Compute / performance
    device: str = Field(
        default="auto",
        description=(
            "Inference device. 'auto' picks the best available: CUDA > MPS > CPU. "
            "Force with 'cuda', 'mps', or 'cpu'. "
            "For ONNX models onnxruntime also tries DirectML (AMD/Intel, Windows)."
        ),
    )
    use_half: bool = Field(
        default=True,
        description="Enable FP16 inference on GPU (CUDA only). Halves VRAM and speeds up ~2×.",
    )
    imgsz: int = Field(
        default=320,
        ge=160,
        le=1280,
        description="YOLO input resolution in pixels (320 = 4× cheaper than 640; good for posture).",
    )
    frame_skip_threshold: float = Field(
        default=8.0,
        ge=0.0,
        le=100.0,
        description=(
            "Mean pixel-difference threshold for frame skipping (0 = disabled). "
            "Frames where the scene barely changes skip inference, saving CPU/GPU. "
            "Recommended range: 5–15."
        ),
    )


class BreakConfig(BaseModel):
    interval_min: int = Field(default=45, ge=5, le=180)
    duration_min: int = Field(default=5, ge=1, le=30)
    posture_alert_sec: int = Field(default=60, ge=10)
    smart_pause_sec: int = Field(default=30, ge=5)

    eye_break_interval_min: int = Field(
        default=20, ge=5, le=60,
        description="Fire the 20-20-20 eye-rest reminder every N minutes (0 = disabled).",
    )
    eye_break_duration_sec: int = Field(
        default=20, ge=5, le=120,
        description="How long the eye-break countdown lasts (seconds).",
    )

    @field_validator("duration_min")
    @classmethod
    def duration_sanity_check(cls, v: int) -> int:
        if v > 40:
            log.warning("unusually_long_break", minutes=v, hint="Is this intentional?")
        return v


class EscalationConfig(BaseModel):
    level_2_delay_min: int = Field(default=2, ge=1)
    level_3_delay_min: int = Field(default=5, ge=1)
    mandatory_break: bool = False
    snooze_options: list[int] = [5, 10, 20]
    respect_focus_mode: bool = True


class AudioSoundsConfig(BaseModel):
    """Per-escalation-level sound config.

    Values are either "builtin:<name>" (for the bundled sound pack)
    or an absolute/relative path to any MP3, WAV, or OGG file.
    """

    level_1: str = "builtin:chime_soft"
    level_2: str = "builtin:tibetan_bowl"
    level_3: str = "builtin:nature_forest"
    break_end: str = "builtin:break_end"


class AudioConfig(BaseModel):
    enabled: bool = True
    volume: int = Field(default=70, ge=0, le=100)
    sounds: AudioSoundsConfig = Field(default_factory=AudioSoundsConfig)

    def resolve_sound_path(self, sound_key: str, assets_dir: Path) -> Path | None:
        """Turn a sound key into an absolute path."""
        val = getattr(self.sounds, sound_key, "")
        if not val:
            return None
        if val.startswith("builtin:"):
            name = val.removeprefix("builtin:")
            return assets_dir / "sounds" / f"{name}.wav"
        return Path(val).expanduser().resolve()


class PrivacyConfig(BaseModel):
    store_detections: bool = True
    encrypt_database: bool = True
    anonymous_telemetry: bool = False  # always off by default; user must explicitly opt in

    @model_validator(mode="after")
    def telemetry_is_opt_in(self) -> PrivacyConfig:
        # Just a reminder that this is intentional — telemetry is never
        # enabled without an explicit user action.
        return self


class UIConfig(BaseModel):
    start_minimized: bool = True
    show_preview_on_start: bool = False
    tray_icon_color_coding: bool = True
    theme: Literal["system", "dark", "light"] = "system"


class DarkHoursConfig(BaseModel):
    enabled: bool = True
    start_hour: int = Field(default=22, ge=0, le=23)
    end_hour: int = Field(default=7, ge=0, le=23)
    reduce_volume: bool = True
    stricter_posture: bool = True


class AppConfig(BaseModel):
    """Root config model — mirror of config.toml sections."""

    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    breaks: BreakConfig = Field(default_factory=BreakConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    dark_hours: DarkHoursConfig = Field(default_factory=DarkHoursConfig)


class ConfigManager:
    """Loads, caches, and hot-reloads the app config.

    Usage:
        manager = ConfigManager.load()
        cfg = manager.config           # get the current AppConfig
        manager.reload()               # re-read from disk (called by file watcher)
    """

    def __init__(self, config: AppConfig, config_path: Path) -> None:
        self._config = config
        self._config_path = config_path

    @classmethod
    def load(cls, path: Path | None = None) -> ConfigManager:
        config_path = path or CONFIG_FILE

        # Make sure our dirs exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        PLUGINS_DIR.mkdir(parents=True, exist_ok=True)

        if not config_path.exists():
            log.info("first_run_creating_config", path=str(config_path))
            config = AppConfig()
            _write_default_config(config_path, config)
            return cls(config, config_path)

        try:
            with open(config_path, "rb") as f:
                raw = tomllib.load(f)
            config = AppConfig.model_validate(raw)

            return cls(config, config_path)
        except Exception as e:
            log.warning(
                "config_invalid",
                error=str(e),
                path=str(config_path),
                action="falling back to defaults",
            )
            return cls(AppConfig(), config_path)

    @property
    def config(self) -> AppConfig:
        return self._config

    def reload(self) -> None:
        """Hot-reload the config from disk. Called by the file watcher."""
        try:
            with open(self._config_path, "rb") as f:
                raw = tomllib.load(f)
            self._config = AppConfig.model_validate(raw)
        except Exception as e:
            log.error("config_reload_failed", error=str(e))


def _write_default_config(path: Path, cfg: AppConfig) -> None:
    content = f"""\
    # NeuraBreak Configuration
    # Edit this file to customize the app. Changes apply without a restart.

    [detection]
    fps                  = {cfg.detection.fps}      # Webcam capture rate (1-30). 5 fps is plenty for posture.
    confidence_threshold = {cfg.detection.confidence_threshold}   # Only alert when model is at least this confident.
    consecutive_frames   = {cfg.detection.consecutive_frames}      # How many frames in a row before triggering.
    model_variant        = "{cfg.detection.model_variant}"  # "nano" (fast, less accurate) / "small" / "medium"

    [breaks]
    interval_min      = {cfg.breaks.interval_min}    # Trigger a full break reminder every N minutes.
    duration_min      = {cfg.breaks.duration_min}     # Minimum break duration in minutes.
    posture_alert_sec = {cfg.breaks.posture_alert_sec}    # Alert after bad posture lasts this long (seconds).
    smart_pause_sec   = {cfg.breaks.smart_pause_sec}    # Pause the session timer if away for N seconds.

    # 20-20-20 rule
    # Every eye_break_interval_min minutes, look 20 feet away for eye_break_duration_sec seconds.
    eye_break_interval_min  = {cfg.breaks.eye_break_interval_min}   # Minutes between eye-rest reminders.
    eye_break_duration_sec  = {cfg.breaks.eye_break_duration_sec}   # Duration of the eye rest (seconds).

    [escalation]
    level_2_delay_min  = {cfg.escalation.level_2_delay_min}     # Escalate to level 2 if level 1 is ignored for N min.
    level_3_delay_min  = {cfg.escalation.level_3_delay_min}     # Escalate to break screen after N more minutes.
    mandatory_break    = {str(cfg.escalation.mandatory_break).lower()}   # Block input until break is taken (strong!).
    respect_focus_mode = {str(cfg.escalation.respect_focus_mode).lower()}  # Suppresses reminders while OS focus assist is on.

    [audio]
    enabled = {str(cfg.audio.enabled).lower()}
    volume  = {cfg.audio.volume}   # 0-100. The app always respects the OS mute state.

    [audio.sounds]
    # Use "builtin:<name>" for the included sound pack, or give a file path:
    #   level_1 = "~/Music/my_chime.mp3"
    level_1   = "{cfg.audio.sounds.level_1}"       # gentle Level 1 nudge
    level_2   = "{cfg.audio.sounds.level_2}"     # persistent Level 2 reminder
    level_3   = "{cfg.audio.sounds.level_3}"    # ambient audio during break
    break_end = "{cfg.audio.sounds.break_end}"           # break-over signal

    [privacy]
    store_detections    = {str(cfg.privacy.store_detections).lower()}   # Save detection events to the local DB.
    encrypt_database    = {str(cfg.privacy.encrypt_database).lower()}   # AES-256 encrypt the health journal.
    anonymous_telemetry = {str(cfg.privacy.anonymous_telemetry).lower()}  # Send anonymous usage stats. Off by default.

    [ui]
    start_minimized         = {str(cfg.ui.start_minimized).lower()}   # Start in the background (no window on launch).
    tray_icon_color_coding  = {str(cfg.ui.tray_icon_color_coding).lower()}  # Green/yellow/red posture indicator in tray.
    theme                   = "{cfg.ui.theme}"    # "system" / "dark" / "light"

    [dark_hours]
    enabled          = {str(cfg.dark_hours.enabled).lower()}
    start_hour       = {cfg.dark_hours.start_hour}   # 24-hour format. 22 = 10 pm.
    end_hour         = {cfg.dark_hours.end_hour}    # 24-hour format. 7 = 7 am.
    reduce_volume    = {str(cfg.dark_hours.reduce_volume).lower()}   # Lower audio volume late at night.
    stricter_posture = {str(cfg.dark_hours.stricter_posture).lower()}  # Extra posture alerts during dark hours.
    """
    path.write_text(content, encoding="utf-8")
    log.info("default_config_written", path=str(path))
