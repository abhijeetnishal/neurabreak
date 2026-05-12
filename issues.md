# NeuraBreak Issue Backlog

This backlog was created from a codebase audit covering the app runtime, AI pipeline, config/settings, notifications, journal, packaging, README, and tests.

## High Priority

### 1. Minimal install does not actually provide break reminders

The README says `uv sync` gives "tray icon + break reminders only", but the app always starts `FrameCaptureService` and `DetectionService`. Without `opencv-python`, camera capture exits and the state machine never receives ticks, so timed break reminders do not fire.

Relevant files:
- `README.md`
- `src/neurabreak/ui/app.py`
- `src/neurabreak/ai/camera.py`

Suggested fix:
- Add a non-camera timer path for minimal installs, or make AI/OpenCV a required runtime dependency.
- Update README install modes to match the real behavior.

### 2. Windows package likely ships AI in stub mode

The PyInstaller spec falls back to bundling `yolo26n.pt`, but explicitly excludes `torch` and `ultralytics`, which are required to load `.pt` models. Unless `models/neurabreak.onnx` exists at build time, packaged posture detection will not work.

Relevant files:
- `packaging/windows/build.spec`
- `src/neurabreak/ai/engine.py`

Suggested fix:
- Require an ONNX model for packaging and fail the build if it is missing, or include the PyTorch/Ultralytics runtime when bundling `.pt`.

### 3. ONNX inference/post-processing is probably broken for exported YOLO models

`training/export.py` exports ONNX without `nms=True`, but the runtime assumes detections are already `[x1, y1, x2, y2, conf, cls]`. The runtime also labels ONNX classes as `class_0`, `class_1`, etc., while posture parsing only accepts `face_present`, `posture_good`, `posture_bad`, and `person_absent`.

Relevant files:
- `training/export.py`
- `src/neurabreak/ai/engine.py`
- `src/neurabreak/ai/postprocessor.py`

Suggested fix:
- Either export ONNX with compatible NMS output or implement YOLO decode and NMS in runtime.
- Map ONNX class IDs through `CLASS_NAMES` before posture parsing.

### 4. Smart pause resets work-session time instead of pausing it

On absence, the state moves to `IDLE`. When presence returns, `_from_idle()` starts a brand-new session and resets elapsed time to `0.0`. This lets stepping away postpone breaks indefinitely, contrary to the "pause/resume" behavior.

Relevant files:
- `src/neurabreak/core/state_machine.py`
- `src/neurabreak/ui/app.py`
- `src/neurabreak/data/journal.py`

Suggested fix:
- Track accumulated active seconds separately from wall-clock session start.
- Publish a true resume event when returning from smart pause.
- Ensure the journal represents paused/resumed sessions consistently.

### 5. Mandatory break mode is not enforced

`mandatory_break` level 4 is currently "same as level 3 for now". The overlay can still be snoozed or hidden after pressing "I'm taking my break"; it does not block input or require the configured break duration.

Relevant files:
- `src/neurabreak/notifications/manager.py`
- `src/neurabreak/ui/break_screen.py`
- `src/neurabreak/core/state_machine.py`

Suggested fix:
- Define mandatory-break semantics clearly.
- Disable snooze/dismiss paths in mandatory mode.
- Keep the overlay active until break duration or absence/return criteria are satisfied.

### 6. Privacy config claims database encryption, but storage is plain SQLite

`encrypt_database` defaults to `True` and default config says AES-256 encryption, but `Database.connect()` creates a normal SQLite database with SQLAlchemy and no encryption layer.

Relevant files:
- `src/neurabreak/core/config.py`
- `src/neurabreak/data/database.py`

Suggested fix:
- Implement real encrypted storage, or remove/disable the encryption setting and update docs/config comments.

## Medium Priority

### 7. Settings save can destroy config and write invalid TOML paths

The settings writer omits the entire `[detection]` section and most privacy fields, so manually configured model/device/FPS settings are lost on save. It also writes custom sound paths inside basic TOML strings without escaping backslashes or quotes, so normal Windows paths can make the config invalid.

Relevant files:
- `src/neurabreak/ui/settings.py`
- `src/neurabreak/core/config.py`

Suggested fix:
- Serialize the full config model.
- Use a TOML writer or correct escaping for strings.
- Add regression tests for preserving detection and privacy config.

### 8. Built-in audio preview buttons are wired incorrectly

`_test_sound()` strips `builtin:` and passes names like `chime_soft` into `play_builtin()`, but `play_builtin()` expects logical keys like `level_1`. Previewing built-in sounds logs `unknown_builtin_sound_key`.

Relevant files:
- `src/neurabreak/ui/settings.py`
- `src/neurabreak/notifications/audio.py`

Suggested fix:
- For built-in preview, call `play_configured()` with the logical sound key or teach `play_builtin()` to accept both logical keys and builtin names.

### 9. Break compliance overcounts snoozes as skipped breaks

Every `BREAK_DUE` creates a new break row. Snooze publishes another `BREAK_DUE`, so a user who snoozes then takes the break gets one skipped break and one taken break.

Relevant files:
- `src/neurabreak/notifications/manager.py`
- `src/neurabreak/ui/app.py`
- `src/neurabreak/data/journal.py`

Suggested fix:
- Track a single active break reminder across snoozes.
- Record snooze separately from skipped/taken compliance.

### 10. macOS and Linux notifications are advertised but no-op

Both platform implementations contain only TODO comments, so non-Windows users get no OS toast even though README claims macOS/Linux native notifications.

Relevant files:
- `src/neurabreak/notifications/platforms/macos.py`
- `src/neurabreak/notifications/platforms/linux.py`
- `README.md`

Suggested fix:
- Implement platform notifications or update README to say they are not supported yet.

### 11. Outbound update check conflicts with "no cloud calls"

The app starts a GitHub releases check on startup with no config toggle, while the README says there are no cloud calls.

Relevant files:
- `src/neurabreak/ui/app.py`
- `src/neurabreak/core/updater.py`
- `README.md`

Suggested fix:
- Add an explicit opt-in/out setting for update checks.
- Update privacy wording to mention the GitHub request if enabled.

### 12. README/config schema mismatches can break user configs

README shows `theme = "auto"`, but the schema only allows `system`, `dark`, or `light`. README also shows `smart_pause = true`, but the real key is `smart_pause_sec`.

Relevant files:
- `README.md`
- `src/neurabreak/core/config.py`

Suggested fix:
- Align README examples with the schema, or update the schema to support the documented keys.

### 13. Several config fields are dead or contradictory

`eye_break_interval_min` says `0 = disabled`, but schema/UI enforce a minimum of `5`. `dark_hours.stricter_posture` and `ui.show_preview_on_start` are configured but not used.

Relevant files:
- `src/neurabreak/core/config.py`
- `src/neurabreak/ui/settings.py`
- `src/neurabreak/ui/app.py`

Suggested fix:
- Either implement these fields or remove them from config.
- If `0` should disable eye breaks, allow `ge=0` and expose `0` in the UI.

### 14. Phone detection pipeline is dead code

`DetectionResult.phone_detected` is published, journaled, and has an event hook, but the engine never sets it from model classes.

Relevant files:
- `src/neurabreak/ai/engine.py`
- `src/neurabreak/ai/detection_service.py`
- `src/neurabreak/data/journal.py`

Suggested fix:
- Add phone class support to the model/class map and parser, or remove phone detection event/journal fields until supported.

## Verification Notes

At audit time:

- `.\.venv\Scripts\python.exe -m pytest tests -q` reported `67 passed`, `4 skipped`, and `8 errors`.
- The 8 errors were all `fixture 'qapp' not found`, indicating the active virtualenv was missing `pytest-qt` or test fixture setup.
- `ruff` and `mypy` were not installed in the active virtualenv.
- `uv run` could not be used because the sandbox hit an access-denied error in the global uv cache.

