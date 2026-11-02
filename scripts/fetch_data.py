#!/usr/bin/env python3
"""Download the four `PLAN.md` §6 parquet files and verify their checksums.

This is the only networked step in the data lane. Run it with ``make fetch-data``; it
writes the gitignored ``data/raw/`` and the committed ``data/CHECKSUMS.txt``. A later
mismatch writes ``reports/STOP.md`` (``PLAN.md`` §10.1) and exits non-zero, leaving the
known-good file untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

from which_ai_model.bench.data import (
    CHECKSUMS_PATH,
    DATASET_FILES,
    RAW_DIR,
    REPORTS_DIR,
    ChecksumMismatchError,
    fetch_data,
)

EXIT_CHECKSUM_MISMATCH: Final = 1
EXIT_OK: Final = 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=RAW_DIR, help="download directory")
    parser.add_argument("--checksums", type=Path, default=CHECKSUMS_PATH)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)

    print(f"fetch-data: {len(DATASET_FILES)} files into {args.dest}")
    for spec in DATASET_FILES:
        print(f"  {spec.name} <- {spec.url}")
    try:
        records = fetch_data(
            args.dest,
            checksums=args.checksums,
            reports_dir=args.reports,
        )
    except ChecksumMismatchError as exc:
        print(f"fetch-data: STOP: {exc}", file=sys.stderr)
        print(
            f"fetch-data: wrote {args.reports / 'STOP.md'}; refusing to continue (§10.1)",
            file=sys.stderr,
        )
        return EXIT_CHECKSUM_MISMATCH

    for record in records:
        print(f"  ok {record.name} {record.sha256} ({record.size} bytes)")
    print(f"fetch-data: checksums {args.checksums}")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
