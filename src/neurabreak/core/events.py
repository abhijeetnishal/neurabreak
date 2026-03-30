"""Internal event bus for decoupled communication between modules.

All communication between the camera, inference engine, state machine,
and UI goes through here. This keeps the modules decoupled, which makes
testing easy (just subscribe + assert instead of mocking call chains).

Usage:
    from neurabreak.core.events import bus, Event, EventType

    bus.subscribe(EventType.BREAK_DUE, my_handler)
    bus.publish(Event(EventType.BREAK_DUE, {"session_elapsed": 2700}))
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import structlog

log = structlog.get_logger()


class EventType(Enum):
    # Detection pipeline
    DETECTION_COMPLETE = auto()

    # Posture
    POSTURE_ALERT = auto()
    POSTURE_RESTORED = auto()

    # Session lifecycle
    SESSION_STARTED = auto()
    SESSION_PAUSED = auto()   # smart pause — person stepped away
    SESSION_RESUMED = auto()  # person returned
    SESSION_ENDED = auto()

    # Breaks
    BREAK_DUE = auto()
    BREAK_STARTED = auto()
    BREAK_ENDED = auto()
    BREAK_SNOOZED = auto()

    # 20-20-20 eye break (every 20 min: look 20 ft away for 20 sec)
    EYE_BREAK_DUE = auto()

    # Misc
    PHONE_DETECTED = auto()
    CONFIG_CHANGED = auto()

    # App lifecycle & updates
    UPDATE_AVAILABLE = auto()   # data: {"version": "1.2.3", "url": "https://..."}

    # Model lifecycle — published from the inference thread during startup
    MODEL_LOADING = auto()   # fired just before engine.load() begins
    MODEL_LOADED = auto()    # fired as soon as engine.load() finishes

    # User-initiated monitoring pause / resume
    MONITORING_PAUSED_MANUAL = auto()
    MONITORING_RESUMED_MANUAL = auto()


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Event({self.type.name}, {self.data})"


class EventBus:
    """Thread-safe publish/subscribe event bus.

    Callbacks run synchronously on the publishing thread, so keep them fast.
    For anything that blocks (file I/O, network, heavy computation), dispatch
    to a background thread inside the callback.
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        with self._lock:
            subs = self._subscribers.get(event_type, [])
            if callback in subs:
                subs.remove(callback)

    def publish(self, event: Event) -> None:
        """Publish an event. All subscribers are called before this returns."""
        with self._lock:
            # Copy the list so subscribers can unsubscribe from within their callback
            callbacks = list(self._subscribers.get(event.type, []))

        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                # A bad subscriber should never crash the bus
                log.error(
                    "event_subscriber_error",
                    event_type=event.type.name,
                    error=str(e),
                    exc_info=True,
                )

    def clear(self) -> None:
        """Remove all subscribers. Mainly useful in tests."""
        with self._lock:
            self._subscribers.clear()


# The single global event bus. Import `bus` directly anywhere you need it.
# Don't create additional EventBus instances — use this one.
bus = EventBus()
