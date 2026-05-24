"""Basic import test for Linux notification module."""

from __future__ import annotations


def test_linux_toast_importable():
    from neurabreak.notifications.platforms import linux

    assert linux.send_toast is not None
