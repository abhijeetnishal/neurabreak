"""Unit tests for AudioManager.

Tests focus on the public API contract without actually playing audio
(sounddevice/soundfile are optional and might not be installed in CI).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestAudioManagerInit:
    def test_volume_is_clamped_at_construction(self, tmp_path):
        from neurabreak.notifications.audio import AudioManager

        am = AudioManager(assets_dir=tmp_path, volume=150)
        assert am.volume == 100

        am2 = AudioManager(assets_dir=tmp_path, volume=-10)
        assert am2.volume == 0

    def test_disabled_manager_skips_play(self, tmp_path):
        from neurabreak.notifications.audio import AudioManager

        am = AudioManager(assets_dir=tmp_path, volume=70, enabled=False)
        # Should return immediately with no threads spawned
        am.play("level_1", tmp_path / "nonexistent.wav")

    def test_assets_dir_and_sounds_created_on_init(self, tmp_path):
        from neurabreak.notifications.audio import AudioManager

        AudioManager(assets_dir=tmp_path)
        sounds_dir = tmp_path / "sounds"
        assert sounds_dir.exists()

    def test_builtin_sounds_generated_on_init(self, tmp_path):
        from neurabreak.notifications.audio import AudioManager, _BUILTIN_SPECS

        AudioManager(assets_dir=tmp_path)
        sounds_dir = tmp_path / "sounds"

        for filename in _BUILTIN_SPECS:
            assert (sounds_dir / filename).exists(), f"Missing: {filename}"

    def test_generated_wav_is_valid(self, tmp_path):
        """Generated placeholder sounds should be readable WAV files."""
        import wave

        from neurabreak.notifications.audio import AudioManager

        AudioManager(assets_dir=tmp_path)
        wav_path = tmp_path / "sounds" / "chime_soft.wav"
        assert wav_path.exists()

        with wave.open(str(wav_path), "r") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 44100
            assert wf.getnframes() > 0


class TestAudioManagerPlayback:
    def test_play_missing_file_falls_back_to_beep(self, tmp_path):
        from neurabreak.notifications.audio import AudioManager

        am = AudioManager(assets_dir=tmp_path)
        am._system_beep = MagicMock()

        # Run synchronously to test the logic
        am._play_file("level_1", tmp_path / "does_not_exist.wav")
        am._system_beep.assert_called_once()

    def test_play_none_path_falls_back_to_beep(self, tmp_path):
        from neurabreak.notifications.audio import AudioManager

        am = AudioManager(assets_dir=tmp_path)
        am._system_beep = MagicMock()
        am._play_file("level_1", None)
        am._system_beep.assert_called_once()

    def test_play_builtin_resolves_correct_path(self, tmp_path):
        from neurabreak.notifications.audio import AudioManager, _BUILTIN_NAMES

        am = AudioManager(assets_dir=tmp_path)
        played: list[tuple] = []
        am._play_file = lambda key, path, vol=None: played.append((key, path))  # type: ignore

        am.play_builtin("level_1")
        # Give the thread a moment (it's fire-and-forget)
        import time
        time.sleep(0.05)

        # The path should point to the chime_soft.wav in our sounds dir
        assert any("chime_soft.wav" in str(p) for _, p in played)

    def test_set_volume_clamps(self, tmp_path):
        from neurabreak.notifications.audio import AudioManager

        am = AudioManager(assets_dir=tmp_path, volume=50)
        am.set_volume(999)
        assert am.volume == 100
        am.set_volume(-5)
        assert am.volume == 0

    def test_play_configured_uses_config_path(self, tmp_path):
        from neurabreak.core.config import AppConfig
        from neurabreak.notifications.audio import AudioManager

        cfg = AppConfig()
        am = AudioManager(assets_dir=tmp_path)

        played_keys: list[str] = []
        am.play = lambda key, path, vol=None: played_keys.append(key)  # type: ignore

        am.play_configured("level_1", cfg)
        assert "level_1" in played_keys
