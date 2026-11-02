#!/usr/bin/env python3
"""Fail when a forbidden name appears in tracked text or the last N commit messages.

The forbidden list lives in ``config/forbidden.sha256``: one lowercase token per line,
hashed with sha256 so the plain tokens are never in the repository. We tokenise text
into lowercase words and digit/dot runs, hash each candidate and compare hashes.

A failure prints the file and line (or commit and line) and a hash prefix for
diagnosis. It never prints the token itself.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._+-]*")
SPLIT_RE = re.compile(r"[._+-]+")
MAX_FILE_BYTES = 8 * 1024 * 1024
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}


def load_hashes(path: Path) -> set[str]:
    """Read one sha256 per line; blank lines and ``#`` comments are ignored."""
    hashes: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip().lower()
        if not line or line.startswith("#"):
            continue
        if not HASH_RE.match(line):
            msg = f"{path}:{lineno}: not a sha256 hex digest"
            raise ValueError(msg)
        hashes.add(line)
    return hashes


def token_hashes(text: str) -> Iterator[tuple[str, str]]:
    """Yield ``(token, sha256)`` for every candidate token in ``text``.

    Full dotted/dashed runs are checked, plus their parts, so a forbidden ``jane``
    is caught inside ``jane.doe`` and ``jane-doe``.
    """
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip("._+-")
        if not token:
            continue
        yield token, hashlib.sha256(token.encode("utf-8")).hexdigest()
        for part in SPLIT_RE.split(token):
            if part and part != token:
                yield part, hashlib.sha256(part.encode("utf-8")).hexdigest()


def find_forbidden_lines(text: str, hashes: set[str]) -> list[tuple[int, str]]:
    """Return ``(line_number, hash_prefix)`` for each line containing a match."""
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for _token, digest in token_hashes(line):
            if digest in hashes:
                hits.append((lineno, digest[:12]))
                break
    return hits


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
    """Files tracked by git, relative to ``root``. Fail loudly outside a repo."""
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


def commit_messages(root: Path, limit: int) -> Iterator[tuple[str, str]]:
    """Yield ``(commit_sha, message)`` for the newest ``limit`` commits."""
    result = subprocess.run(
        ["git", "log", f"-{limit}", "--pretty=format:%H%x00%B%x1e"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode != 0:
        return
    for record in result.stdout.split("\x1e"):
        if not record.strip():
            continue
        sha, _, message = record.partition("\x00")
        yield sha.strip(), message


def scan(
    root: Path,
    hashes: set[str],
    *,
    commits: int = 200,
    files: Sequence[Path] | None = None,
) -> list[str]:
    """Return human-readable findings; empty means clean."""
    findings: list[str] = []
    for path in files if files is not None else tracked_files(root):
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        text = decode_text(path)
        if text is None:
            continue
        for lineno, prefix in find_forbidden_lines(text, hashes):
            findings.append(f"{relative}:{lineno}: forbidden token (sha256 {prefix}…)")
    if commits > 0:
        for sha, message in commit_messages(root, commits):
            for lineno, prefix in find_forbidden_lines(message, hashes):
                findings.append(f"commit {sha[:12]}:{lineno}: forbidden token (sha256 {prefix}…)")
    return findings


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument(
        "--hash-file",
        type=Path,
        default=Path("config/forbidden.sha256"),
        help="sha256 list, relative to --root unless absolute",
    )
    parser.add_argument("--commits", type=int, default=200, help="commit messages to scan")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root: Path = args.root.resolve()
    hash_path = args.hash_file if args.hash_file.is_absolute() else root / args.hash_file
    try:
        hashes = load_hashes(hash_path)
    except (OSError, ValueError) as exc:
        print(f"scan-forbidden: cannot read {hash_path}: {exc}", file=sys.stderr)
        return 2
    if not hashes:
        print(f"scan-forbidden: {hash_path} is empty; refusing a vacuous pass", file=sys.stderr)
        return 2

    try:
        findings = scan(root, hashes, commits=args.commits)
    except RuntimeError as exc:
        print(f"scan-forbidden: {exc}", file=sys.stderr)
        return 2

    if findings:
        for finding in findings:
            print(finding)
        print(f"scan-forbidden: {len(findings)} forbidden token(s) found", file=sys.stderr)
        return 1
    print(f"scan-forbidden: clean ({len(hashes)} hashes, {args.commits} commits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
