"""Execute the replay example from ``docs/INTERFACES.md`` verbatim.

The documentation promises the block runs against the committed fixture; running it here
keeps that promise true. The fixture recording holds one structured answer and no
network call is possible (``pytest-socket`` + ``NetworkGuard``).
"""

from __future__ import annotations

import re
from pathlib import Path

from which_ai_model.core.network_guard import NetworkGuard


def test_interfaces_replay_example_runs(repo_root: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    document = (repo_root / "docs" / "INTERFACES.md").read_text(encoding="utf-8")
    marker = "### Replay example"
    start = document.index(marker)
    match = re.search(r"```python\n(.*?)```", document[start:], re.DOTALL)
    assert match is not None
    block = match.group(1)

    monkeypatch.chdir(repo_root)
    with NetworkGuard() as guard:
        exec(compile(block, "docs/INTERFACES.md", "exec"), {})
    assert guard.attempts == []
