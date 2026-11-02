#!/usr/bin/env python3
"""Resolve the pinned NHTSA ODI files and, only on request, rebuild ``data/defect/``.

The raw files come from the scout's snapshot folder when every pinned sha256 matches;
otherwise they are downloaded into the gitignored ``data/raw/nhtsa/``. A republished
file that no longer matches is a **warning, not a stop** (``PLAN-v1.1`` §A2.4): the
committed extracts under ``data/defect/`` are the source of truth, so replacing them is
an explicit act and needs ``--rebuild``.

Run it with ``uv run python scripts/fetch_nhtsa.py``; ``--rebuild`` reads the resolved
files and rewrites the extracts, statistics and checksums.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

from which_ai_model.bench.nhtsa import (
    DEFECT_DIR,
    GROUPS_PATH,
    RAW_DIR,
    SPEC_PATH,
    NhtsaDataError,
    build_dataset,
    ensure_source,
    load_groups,
    load_spec,
)

EXIT_OK: Final = 0
EXIT_ERROR: Final = 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="snapshot folder to prefer (default: the spec's snapshot_dir)",
    )
    parser.add_argument("--dest", type=Path, default=RAW_DIR, help="download directory")
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--groups", type=Path, default=GROUPS_PATH)
    parser.add_argument("--out", type=Path, default=DEFECT_DIR)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rewrite data/defect/ from the resolved raw files",
    )
    args = parser.parse_args(argv)

    try:
        spec = load_spec(args.spec)
        source = ensure_source(spec, snapshot_dir=args.snapshot, dest=args.dest)
    except NhtsaDataError as exc:
        print(f"fetch-nhtsa: {exc}", file=sys.stderr)
        return EXIT_ERROR

    for warning in source.warnings:
        print(f"fetch-nhtsa: warning: {warning}", file=sys.stderr)
    origin = "snapshot" if source.from_snapshot else "download"
    print(f"fetch-nhtsa: {len(spec.files)} pinned files from the {origin} at {source.directory}")
    for entry in spec.files:
        print(f"  {entry.name} ({entry.role})")

    if not args.rebuild:
        print("fetch-nhtsa: committed extracts in data/defect/ left untouched (pass --rebuild)")
        return EXIT_OK

    try:
        stats = build_dataset(
            spec=spec,
            groups=load_groups(args.groups),
            source_dir=source.directory,
            out_dir=args.out,
            force=True,
        )
    except NhtsaDataError as exc:
        print(f"fetch-nhtsa: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(
        f"fetch-nhtsa: rebuilt {args.out}: sample {stats.sample_total:,} of a "
        f"{stats.wam_total:,}-item pool, train {stats.train_item_total:,}"
    )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
