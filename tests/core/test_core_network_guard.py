"""The local zero-socket helper refuses INET and allows unix-domain sockets."""

from __future__ import annotations

import socket

import pytest

from which_ai_model.core.network_guard import NetworkAttempted, NetworkGuard


def test_inet_socket_is_refused_and_recorded() -> None:
    with NetworkGuard() as guard, pytest.raises(NetworkAttempted):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    assert guard.attempts


def test_unix_socket_is_allowed() -> None:
    with NetworkGuard() as guard:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.close()
    assert guard.attempts == []


def test_guard_restores_the_previous_socket_class() -> None:
    before = socket.socket
    with NetworkGuard():
        assert socket.socket is not before
    assert socket.socket is before
