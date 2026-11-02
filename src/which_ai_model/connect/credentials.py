"""Credential handles: resolve ``env:NAME`` at call time, never expose the value.

Only this module reads the process environment or a ``.env`` file for a secret.
The resolver returns a :class:`Credential` value object whose ``repr`` and ``str``
are ``***``; call sites that need the bytes call :meth:`Credential.reveal`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from which_ai_model.contracts import Connection

__all__ = [
    "Credential",
    "read_env_file",
    "resolve_credential",
    "resolve_credential_value",
]

#: Overrides the ``.env`` path (tests point this at a temporary file).
ENV_FILE_VAR = "WAM_ENV_FILE"


@dataclass(frozen=True)
class Credential:
    """A resolved secret. It is deliberately unpleasant to print."""

    _value: str = field(repr=False)
    env_name: str

    def reveal(self) -> str:
        """The raw value. Call this only where an HTTP client needs it."""
        return self._value

    def __repr__(self) -> str:
        return f"Credential(env={self.env_name}, value=***)"

    def __str__(self) -> str:
        return "***"


def _env_file() -> Path:
    return Path(os.environ.get(ENV_FILE_VAR, ".env"))


def read_env_file(path: Path | None = None) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines. Absent files are empty; malformed lines are ignored."""
    file = path or _env_file()
    try:
        text = file.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return {}
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if key:
            values[key] = value.strip().strip("\"'")
    return values


def resolve_credential(connection: Connection) -> Credential | None:
    """Resolve the connection's key, or ``None`` for a keyless connection.

    Precedence: the process environment, then the ``.env`` file. An empty value counts
    as missing. Never raises with the value in the message.
    """
    name = connection.credential_env
    if name is None:
        return None
    value = os.environ.get(name) or read_env_file().get(name)
    if not value:
        return None
    return Credential(_value=value, env_name=name)


def resolve_credential_value(connection: Connection) -> str | None:
    """The raw value for callers that cannot hold a :class:`Credential`.

    Prefer :func:`resolve_credential`; this exists for the interface contract in
    ``docs/INTERFACES.md`` and is used only inside this package.
    """
    credential = resolve_credential(connection)
    return credential.reveal() if credential is not None else None
