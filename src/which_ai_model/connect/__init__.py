"""BYOK connections for ``which_ai_model``.

Public surface (``docs/INTERFACES.md`` §2): connections, credentials, probe, discovery,
``doctor``. ``chat_model`` is lazy so importing this package stays cheap for the CLI.
"""

from __future__ import annotations

from typing import Any

from which_ai_model.connect.credentials import (
    Credential,
    resolve_credential,
    resolve_credential_value,
)
from which_ai_model.connect.discovery import (
    Catalog,
    check_discovery,
    discover_models,
    load_catalog,
)
from which_ai_model.connect.doctor import (
    DoctorReport,
    collect_report,
    doctor,
    format_doctor,
    report_to_dict,
)
from which_ai_model.connect.presets import (
    BENCHMARK_MODELS,
    GATEWAY_MODELS_URL,
    USER_AGENT,
    Preset,
    all_presets,
    connection_headers,
    default_connections,
    preset_connections,
)
from which_ai_model.connect.probe import ProbeOutcome, check_connection, probe_connection
from which_ai_model.connect.store import (
    StoreError,
    connections_to_toml,
    load_connections,
    save_connections,
)

__all__ = [
    "BENCHMARK_MODELS",
    "GATEWAY_MODELS_URL",
    "USER_AGENT",
    "Catalog",
    "Credential",
    "DoctorReport",
    "Preset",
    "ProbeOutcome",
    "StoreError",
    "all_presets",
    "chat_model",
    "check_connection",
    "check_discovery",
    "collect_report",
    "connection_headers",
    "connections_to_toml",
    "default_connections",
    "discover_models",
    "doctor",
    "format_doctor",
    "load_catalog",
    "load_connections",
    "preset_connections",
    "probe_connection",
    "report_to_dict",
    "resolve_credential",
    "resolve_credential_value",
    "save_connections",
]


def __getattr__(name: str) -> Any:
    if name == "chat_model":
        from which_ai_model.connect.to_langchain import chat_model

        return chat_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
