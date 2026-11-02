"""Read and write ``config/providers.toml`` — connections only, never values.

A saved file holds ``id``, ``provider``, ``protocol``, ``base_url``, ``credential_ref``
and ``preset``. Any key that looks like a credential, or any value that looks like a
key, is refused rather than written.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from which_ai_model.contracts import Connection

__all__ = [
    "FIELDS",
    "StoreError",
    "connections_to_toml",
    "load_connections",
    "save_connections",
]

FIELDS: tuple[str, ...] = (
    "id",
    "provider",
    "protocol",
    "base_url",
    "credential_ref",
    "preset",
)

_TOP_LEVEL = "connection"
_SUSPECT_KEYS = frozenset(
    {
        "api-key",
        "api_key",
        "apikey",
        "authorization",
        "key",
        "password",
        "secret",
        "token",
    }
)
_SECRET_VALUE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{8,}|sk-ant-|AIza[0-9A-Za-z_-]{10,}|lsv2_|"
    r"Bearer\s+[A-Za-z0-9._-]{8,}|[A-Za-z0-9_-]{48,})"
)
_HEADER = "# Connections only. Never put a key value here; use credential_ref = env:NAME.\n"


class StoreError(ValueError):
    """A providers file that cannot be used. Messages never echo values."""


def _refuse_secret(key: str, value: object) -> None:
    if key.lower().replace("_", "-") in {"api-key", "apikey", "authorization"}:
        raise StoreError(f"field {key!r} looks like a credential; use credential_ref=env:NAME")
    if isinstance(value, str) and _SECRET_VALUE.search(value):
        raise StoreError(f"field {key!r} contains a credential-looking value")


def connections_to_toml(connections: Iterable[Connection]) -> str:
    """Serialise connections deterministically (stable field and row order)."""
    rows = list(connections)
    seen: set[str] = set()
    parts = [_HEADER]
    for connection in rows:
        if connection.id in seen:
            raise StoreError(f"duplicate connection id {connection.id!r}")
        seen.add(connection.id)
        parts.append(f"\n[[{_TOP_LEVEL}]]\n")
        for name in FIELDS:
            value = getattr(connection, name)
            if value is None:
                continue
            rendered = value.value if hasattr(value, "value") else value
            parts.append(f"{name} = {json.dumps(rendered, ensure_ascii=False)}\n")
    return "".join(parts)


def save_connections(connections: Iterable[Connection], path: Path | None = None) -> Path:
    """Write a providers file, creating parent directories."""
    target = path or Path("config/providers.toml")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(connections_to_toml(connections), encoding="utf-8")
    return target


def _rows(document: dict[str, Any], path: Path) -> list[Any]:
    unknown = sorted(set(document) - {_TOP_LEVEL})
    if unknown:
        raise StoreError(f"{path}: unexpected top-level keys {unknown}")
    rows = document.get(_TOP_LEVEL, [])
    if not isinstance(rows, list):
        raise StoreError(f"{path}: [[{_TOP_LEVEL}]] must be an array of tables")
    return rows


def load_connections(path: Path | None = None) -> list[Connection]:
    """Load and validate a providers file. Any problem is a :class:`StoreError`."""
    target = path or Path("config/providers.toml")
    try:
        document = tomllib.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StoreError(f"providers file not found: {target}") from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise StoreError(f"{target}: {type(exc).__name__}") from exc

    connections: list[Connection] = []
    seen: set[str] = set()
    for index, row in enumerate(_rows(document, target)):
        where = f"{target}: [[{_TOP_LEVEL}]] #{index + 1}"
        if not isinstance(row, dict):
            raise StoreError(f"{where} is not a table")
        for key, value in row.items():
            if key.lower().replace("_", "-") in _SUSPECT_KEYS:
                raise StoreError(f"{where}: refuse field {key!r}; use credential_ref=env:NAME")
            _refuse_secret(str(key), value)
        unknown = sorted(set(row) - set(FIELDS))
        if unknown:
            raise StoreError(f"{where}: unexpected fields {unknown}")
        try:
            connection = Connection(**row)
        except ValidationError as exc:
            names = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
            raise StoreError(f"{where}: invalid fields {names}") from None
        if connection.id in seen:
            raise StoreError(f"{where}: duplicate id {connection.id!r}")
        seen.add(connection.id)
        connections.append(connection)
    if not connections:
        raise StoreError(f"{target}: no [[{_TOP_LEVEL}]] rows")
    return connections
