"""Explicit zero-socket guard for tests.

``pytest-socket`` already blocks INET sockets for every test (``pyproject.toml``). This
helper makes the assertion local and readable where it matters — replay, record/replay
and trace tests — and records the addresses a test tried to reach so a failure says
what leaked::

    with NetworkGuard() as guard:
        ...  # replay call
    assert guard.attempts == []

Unix-domain sockets stay allowed: Postgres over a unix socket and subprocess pipes are
local IPC, not network (``docs/INTERFACES.md`` §0).
"""

from __future__ import annotations

import _socket
import socket
from types import TracebackType
from typing import Any, NoReturn

__all__ = ["NetworkAttempted", "NetworkGuard"]

_INET_FAMILIES = {socket.AF_INET, getattr(socket, "AF_INET6", socket.AF_INET)}


class NetworkAttempted(OSError):  # noqa: N818
    """Raised when guarded code opens an INET socket or connects to a network address."""

    def __init__(self, address: Any) -> None:
        self.address = address
        super().__init__(f"network access attempted: {address!r}")


class NetworkGuard:
    """Context manager that refuses INET sockets and records every attempt."""

    def __init__(self) -> None:
        self.attempts: list[Any] = []
        self._real_socket: type[socket.socket] | None = None

    def _refuse(self, address: Any) -> NoReturn:
        self.attempts.append(address)
        raise NetworkAttempted(address)

    def __enter__(self) -> NetworkGuard:
        # ``socket.socket`` may already be a guard (pytest-socket). Subclass the real
        # C socket so this helper's refusal is the one that fires, then restore the
        # previous class on exit.
        real_socket = _socket.socket
        self._real_socket = socket.socket
        guard = self

        def guarded_init(sock: Any, family: int = -1, *args: Any, **kwargs: Any) -> None:
            if family in _INET_FAMILIES:
                guard._refuse(("socket", _family_name(family)))
            real_socket.__init__(sock, family, *args, **kwargs)

        def guarded_connect(sock: Any, address: Any) -> None:
            del sock
            guard._refuse(address)

        def guarded_connect_ex(sock: Any, address: Any) -> int:
            del sock
            guard._refuse(address)

        def guarded_sendto(sock: Any, data: Any, *args: Any, **kwargs: Any) -> int:
            del sock, data, kwargs
            guard._refuse(("sendto", args))

        guarded = type(
            "_GuardedSocket",
            (real_socket,),
            {
                "__init__": guarded_init,
                "connect": guarded_connect,
                "connect_ex": guarded_connect_ex,
                "sendto": guarded_sendto,
            },
        )
        setattr(socket, "socket", guarded)  # noqa: B010 - typeshed types it as a class
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._real_socket is not None:
            setattr(socket, "socket", self._real_socket)  # noqa: B010


def _family_name(family: int) -> str:
    return "AF_INET6" if family == getattr(socket, "AF_INET6", -1) else "AF_INET"
