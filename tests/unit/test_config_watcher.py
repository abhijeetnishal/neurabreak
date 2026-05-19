from __future__ import annotations

from pathlib import Path

from neurabreak.core.config import ConfigManager
from neurabreak.core.events import EventType, bus
from neurabreak.ui.app import NeuraBreakApp


def test_external_config_change_triggers_event(tmp_path, qapp):
    config_path = tmp_path / "config.toml"
    manager = ConfigManager.load(path=config_path)

    app = NeuraBreakApp(manager)

    events: list = []
    bus.clear()
    bus.subscribe(EventType.CONFIG_CHANGED, lambda e: events.append(e))

    # initialize mtime tracker
    app._config_file_mtime = float(config_path.stat().st_mtime)

    # write a valid change
    config_path.write_text("[breaks]\ninterval_min = 10\n", encoding="utf-8")

    app._check_config_file()

    assert len(events) == 1
    cfg = events[0].data.get("config")
    assert cfg is not None
    assert cfg.breaks.interval_min == 10


def test_invalid_external_config_does_not_replace(tmp_path, qapp):
    config_path = tmp_path / "config.toml"
    manager = ConfigManager.load(path=config_path)

    app = NeuraBreakApp(manager)

    events: list = []
    bus.clear()
    bus.subscribe(EventType.CONFIG_CHANGED, lambda e: events.append(e))

    # initialize mtime tracker
    app._config_file_mtime = float(config_path.stat().st_mtime)

    # write invalid toml
    config_path.write_text("not a toml [[[", encoding="utf-8")

    app._check_config_file()

    # no event published and config unchanged
    assert len(events) == 0
    assert manager.config.breaks.interval_min == 45
