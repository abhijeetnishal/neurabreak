"""Audio manager — plays notification sounds without blocking the UI.

Supports two sound sources:
  - Built-in pack: bundled .wav files under ui/assets/sounds/
  - Custom sounds:  any .mp3 / .wav / .ogg the user provides in config.toml

Playback is fire-and-forget: each call spins up a short-lived daemon thread
so the Qt event loop is never stalled. Only one sound plays at a time —
if audio is already playing, the new request waits for the lock.
"""

from __future__ import annotations

import math
import struct
import threading
import wave
from pathlib import Path

import structlog

log = structlog.get_logger()

# Filename for each logical sound key — matches config.toml [audio.sounds] keys
_BUILTIN_NAMES: dict[str, str] = {
    "level_1": "chime_soft.wav",
    "level_2": "tibetan_bowl.wav",
    "level_3": "nature_forest.wav",
    "break_end": "break_end.wav",
}

# Simple sine-wave specs: (frequency_hz, duration_sec) for generated placeholders
_BUILTIN_SPECS: dict[str, tuple[int, float]] = {
    "chime_soft.wav":    (880, 0.8),
    "tibetan_bowl.wav":  (528, 1.5),
    "nature_forest.wav": (432, 2.0),
    "break_end.wav":     (660, 0.6),
}


class AudioManager:
    """Non-blocking audio playback for notification events."""

    def __init__(self, assets_dir: Path, volume: int = 70, enabled: bool = True) -> None:
        self.assets_dir = assets_dir
        self.volume = max(0, min(100, volume))
        self.enabled = enabled
        self._lock = threading.Lock()  # one-at-a-time playback
        self._ensure_builtin_sounds()

    # Public API

    def play(self, sound_key: str, sound_path: Path | None = None, volume_override: int | None = None) -> None:
        """Play a sound file asynchronously (fire and forget).

        Args:
            sound_key:       Logical identifier like "level_1" — used for logging.
            sound_path:      Resolved path to the file. Pass None to trigger system beep.
            volume_override: If given, use this volume (0–100) instead of self.volume.
        """
        if not self.enabled:
            return
        thread = threading.Thread(
            target=self._play_file,
            args=(sound_key, sound_path, volume_override),
            name=f"Audio-{sound_key}",
            daemon=True,
        )
        thread.start()

    def play_builtin(self, sound_key: str) -> None:
        """Play one of the bundled sounds by its logical key."""
        filename = _BUILTIN_NAMES.get(sound_key)
        if filename is None:
            log.warning("unknown_builtin_sound_key", key=sound_key)
            return
        self.play(sound_key, self.assets_dir / "sounds" / filename)

    def play_configured(self, sound_key: str, config: object, volume_override: int | None = None) -> None:
        """Resolve the path from config and play it."""
        from neurabreak.core.config import AppConfig

        if not isinstance(config, AppConfig):
            return
        path = config.audio.resolve_sound_path(sound_key, self.assets_dir)
        self.play(sound_key, path, volume_override)

    def set_volume(self, volume: int) -> None:
        self.volume = max(0, min(100, volume))

    def stop(self) -> None:
        """Stop whatever is currently playing."""
        try:
            import sounddevice as sd  # type: ignore
            sd.stop()
        except Exception:
            pass
    
    
    # Internal helpers

    def _play_file(self, sound_key: str, sound_path: Path | None, volume_override: int | None = None) -> None:
        with self._lock:
            if sound_path is None or not sound_path.exists():
                log.warning("audio_file_missing", key=sound_key, path=str(sound_path))
                self._system_beep()
                return
            try:
                import sounddevice as sd  # type: ignore
                import soundfile as sf  # type: ignore

                data, samplerate = sf.read(str(sound_path), dtype="float32")
                vol = (volume_override if volume_override is not None else self.volume)
                data = data * (max(0, min(100, vol)) / 100.0)
                sd.play(data, samplerate)
                sd.wait()
                log.debug("audio_played", key=sound_key)
            except ImportError:
                log.debug("sounddevice_not_installed", hint="install audio extras")
                # Installed audio extras are optional
                self._system_beep()
            except Exception as e:
                log.error("audio_playback_error", key=sound_key, error=str(e))
                self._system_beep()

    def _system_beep(self) -> None:
        try:
            import winsound
            winsound.MessageBeep()
        except ImportError:
            print("\a", end="", flush=True)

    def _ensure_builtin_sounds(self) -> None:
        """Generate placeholder WAV tones if the bundled sounds are missing.

        Keeps the app functional on first run without requiring any extra
        download step. Users can replace these with proper audio files by
        editing the config.toml [audio.sounds] paths.
        """
        sounds_dir = self.assets_dir / "sounds"
        sounds_dir.mkdir(parents=True, exist_ok=True)

        for filename, (freq, duration) in _BUILTIN_SPECS.items():
            path = sounds_dir / filename
            if not path.exists():
                self._generate_wav(path, freq=freq, duration=duration)

    def _generate_wav(
        self, path: Path, freq: int = 440, duration: float = 0.5, sample_rate: int = 44100
    ) -> None:
        """Write a simple fade-out sine-wave tone to a .wav file."""
        n = int(sample_rate * duration)
        try:
            with wave.open(str(path), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                for i in range(n):
                    t = i / sample_rate
                    # Smooth exponential fade-out so there's no click at the end
                    envelope = math.exp(-3.0 * t / duration)
                    val = int(0.35 * 32767 * envelope * math.sin(2 * math.pi * freq * t))
                    wf.writeframes(struct.pack("<h", val))
            log.debug("generated_placeholder_sound", path=str(path), freq=freq)
        except Exception as e:
            log.warning("sound_generation_failed", path=str(path), error=str(e))
