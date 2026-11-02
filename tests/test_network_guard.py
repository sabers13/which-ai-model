"""The test-suite network guard must actually be armed (PLAN.md §2.4)."""

from __future__ import annotations

import socket

import pytest
import pytest_socket


@pytest.mark.filterwarnings("ignore:A test tried to use socket.socket")
def test_inet_sockets_are_blocked() -> None:
    with pytest.raises(pytest_socket.SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_unix_sockets_are_allowed() -> None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.close()
