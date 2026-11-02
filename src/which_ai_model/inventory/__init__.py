"""Static inventory of LLM call sites (T06).

Public entry points follow ``docs/INTERFACES.md`` §7: :func:`scan`,
:func:`write_json`, :func:`write_markdown`, and the :class:`CallSite` /
:class:`Inventory` types.
"""

from __future__ import annotations

from which_ai_model.inventory.scan import (
    CallSite,
    Inventory,
    scan,
    write_json,
    write_markdown,
)

__all__ = ["CallSite", "Inventory", "scan", "write_json", "write_markdown"]
