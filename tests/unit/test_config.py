"""Tests for config loading, default generation, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from neurabreak.core.config import (
    AppConfig,
    AudioConfig,
    AudioSoundsConfig,
    BreakConfig,
    ConfigManager,
    DetectionConfig,
    _write_default_config,
)


class TestAppConfigDefaults:
    def test_default_config_is_valid(self):
        cfg = AppConfig()
        assert cfg.detection.fps == 5
        assert cfg.breaks.interval_min == 45
        assert cfg.audio.volume == 70
        assert cfg.audio.enabled is True
        assert cfg.privacy.anonymous_telemetry is False

    def test_detection_config_clamps_fps(self):
        with pytest.raises(Exception):
            DetectionConfig(fps=0)

    def test_detection_config_clamps_confidence(self):
        with pytest.raises(Exception):
            DetectionConfig(confidence_threshold=1.5)

    def test_audio_volume_clamped(self):
        with pytest.raises(Exception):
            AudioConfig(volume=101)

    def test_break_interval_minimum(self):
        with pytest.raises(Exception):
            BreakConfig(interval_min=4)

    def test_audio_sounds_defaults(self):
        sounds = AudioSoundsConfig()
        assert sounds.level_1 == "builtin:chime_soft"
        assert sounds.level_2 == "builtin:tibetan_bowl"
        assert sounds.break_end == "builtin:break_end"


class TestConfigManager:
    def test_load_creates_default_config_on_first_run(self, tmp_path):
        config_path = tmp_path / "config.toml"
        assert not config_path.exists()

        manager = ConfigManager.load(path=config_path)

        assert config_path.exists()
        assert isinstance(manager.config, AppConfig)
        assert manager.config.breaks.interval_min == 45

    def test_load_reads_existing_config(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            "[breaks]\ninterval_min = 30\n",
            encoding="utf-8",
        )

        manager = ConfigManager.load(path=config_path)
        assert manager.config.breaks.interval_min == 30

    def test_load_falls_back_to_defaults_on_invalid_toml(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text("this is not valid toml [[[", encoding="utf-8")

        # Should not raise — falls back to defaults
        manager = ConfigManager.load(path=config_path)
        assert isinstance(manager.config, AppConfig)

    def test_reload_picks_up_file_changes(self, tmp_path):
        config_path = tmp_path / "config.toml"
        manager = ConfigManager.load(path=config_path)
        assert manager.config.breaks.interval_min == 45

        config_path.write_text("[breaks]\ninterval_min = 20\n", encoding="utf-8")
        manager.reload()
        assert manager.config.breaks.interval_min == 20

    def test_privacy_telemetry_off_by_default(self, tmp_path):
        manager = ConfigManager.load(path=tmp_path / "config.toml")
        assert manager.config.privacy.anonymous_telemetry is False


class TestDefaultConfigFileContent:
    def test_default_config_file_is_valid_toml(self, tmp_path):
        import tomllib

        config_path = tmp_path / "config.toml"
        _write_default_config(config_path, AppConfig())

        with open(config_path, "rb") as f:
            parsed = tomllib.load(f)

        assert "detection" in parsed
        assert "breaks" in parsed
        assert "audio" in parsed
        assert "privacy" in parsed

    def test_audio_resolve_builtin_sound(self):
        cfg = AudioConfig()
        assets_dir = Path("/fake/assets")
        path = cfg.resolve_sound_path("level_1", assets_dir)
        assert path is not None
        assert "chime_soft.wav" in str(path)

    def test_audio_resolve_custom_sound(self, tmp_path):
        custom_sound = tmp_path / "my_sound.mp3"
        custom_sound.touch()
        cfg = AudioConfig()
        cfg.sounds.level_1 = str(custom_sound)
        path = cfg.resolve_sound_path("level_1", Path("/assets"))
        assert path == custom_sound.resolve()
