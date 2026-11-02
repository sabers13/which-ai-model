#!/usr/bin/env python3
"""Fail when something in the repository looks like a live credential.

Scans every tracked text file plus anything under ``recordings/``, ``traces/`` and
``results/`` (those are committed or produced by the live run). Prints the file and
line, never the matching value.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

MAX_FILE_BYTES = 8 * 1024 * 1024
SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
EXTRA_DIRS = ("recordings", "traces", "results")

#: Pattern rules. Each is (rule name, compiled pattern); group 0 is the whole match.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai-style key", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_-]{20,}")),
    ("anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("google api key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}")),
    ("langsmith key", re.compile(r"\blsv2_[A-Za-z0-9_-]{20,}")),
    ("private key block", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
    (
        "secret assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|authorization)\b"
            r"\s*[:=]\s*[\"'](?P<value>[^\"'\s]{16,})[\"']"
        ),
    ),
    (
        "hex secret assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password)\b"
            r"\s*[:=]\s*[\"'](?P<value>[0-9a-f]{32,})[\"']"
        ),
    ),
)

#: Values that are obviously placeholders or indirections, not credentials.
SAFE_PREFIXES = (
    "env:",
    "<",
    "your_",
    "your-",
    "xxx",
    "redacted",
    "placeholder",
    "example",
    "changeme",
    "test-",
    "dummy",
)
SAFE_WORDS = ("...",)


def is_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered.startswith(SAFE_PREFIXES):
        return True
    if any(word in lowered for word in SAFE_WORDS):
        return True
    return lowered in {"none", "null", "true", "false"}


def scan_text(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, rule_name)`` for every suspicious line."""
    findings: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for rule, pattern in PATTERNS:
            match = pattern.search(line)
            if match is None:
                continue
            value = match.groupdict().get("value")
            if value is not None and is_placeholder(value):
                continue
            findings.append((lineno, rule))
            break
    return findings


def decode_text(path: Path) -> str | None:
    """Read a text file; return ``None`` for binaries or oversized files."""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        msg = f"{root} is not a git repository (git ls-files failed)"
        raise RuntimeError(msg)
    names = [n for n in result.stdout.split(b"\x00") if n]
    return [root / n.decode("utf-8", errors="replace") for n in names]


def candidate_files(root: Path) -> Iterator[Path]:
    """Tracked files plus everything in the run-output directories."""
    seen: set[Path] = set()
    for path in tracked_files(root):
        if path.is_file() and path not in seen:
            seen.add(path)
            yield path
    for name in EXTRA_DIRS:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def scan(root: Path, *, files: Iterable[Path] | None = None) -> list[str]:
    """Return human-readable findings; empty means clean."""
    findings: list[str] = []
    paths = candidate_files(root) if files is None else files
    for path in paths:
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        text = decode_text(path)
        if text is None:
            continue
        for lineno, rule in scan_text(text):
            findings.append(f"{relative}:{lineno}: possible {rule} (value not shown)")
    return findings


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root: Path = args.root.resolve()
    try:
        findings = scan(root)
    except RuntimeError as exc:
        print(f"scan-secrets: {exc}", file=sys.stderr)
        return 2

    if findings:
        for finding in findings:
            print(finding)
        print(f"scan-secrets: {len(findings)} suspicious line(s)", file=sys.stderr)
        return 1
    print("scan-secrets: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
