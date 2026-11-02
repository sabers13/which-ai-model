"""The fixed benchmark samples (``PLAN.md`` §7).

Samples are committed as **id lists only** under ``data/samples/``; dataset text stays in
the gitignored ``data/raw/``. The seed is fixed for the whole project, so the ids in the
files are reproducible from the same parquet files and are covered by
``data/CHECKSUMS.txt``.

* ``route.txt`` — T-ROUTE: 8 per Banking77 intent (77 x 8 = 616).
* ``scope.txt`` — T-SCOPE: 150 ``oos`` + 150 in-scope CLINC150 items (300).
* ``reply.txt`` — T-REPLY: a seeded 150-item subset of ``route.txt``.

``sample_hash`` hashes the three id files in a fixed order; ``RunManifest`` records it so
the gate can refuse runs scored against different samples.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from which_ai_model.bench.data import (
    BANKING77_LABELS,
    CLINC150_LABELS,
    CLINC_OOS_LABEL,
    RAW_DIR,
    load_banking77,
    load_clinc150,
)

__all__ = [
    "SAMPLE_DIR",
    "SAMPLE_FILES",
    "SAMPLE_SEED",
    "build_samples",
    "load_sample_ids",
    "main",
    "sample_hash",
    "sample_reply",
    "sample_route",
    "sample_scope",
    "write_sample_ids",
]

SAMPLE_SEED = 20261028
SAMPLE_DIR = Path("data/samples")
SAMPLE_FILES: tuple[str, ...] = ("route.txt", "scope.txt", "reply.txt")

ROUTE_PER_INTENT = 8
SCOPE_PER_GROUP = 150
REPLY_ITEMS = 150


def _rng(seed: int, purpose: str) -> random.Random:
    """A private RNG per sample, so adding a task cannot move another task's ids."""
    return random.Random(f"{seed}:{purpose}")


def _require_columns(frame: pd.DataFrame, path: str) -> None:
    missing = {"item_id", "intent"} - set(frame.columns)
    if missing:
        msg = f"{path}: missing column(s) {sorted(missing)}"
        raise ValueError(msg)


def _select(frame: pd.DataFrame, ids: Sequence[str]) -> pd.DataFrame:
    """Return rows in the given id order; every id must exist exactly once."""
    indexed = frame.set_index("item_id", drop=False)
    missing = [item_id for item_id in ids if item_id not in indexed.index]
    if missing:
        msg = f"item id(s) not in the frame: {missing[:5]}"
        raise ValueError(msg)
    return indexed.loc[list(ids)].reset_index(drop=True)


def sample_route(
    frame: pd.DataFrame,
    *,
    per_intent: int = ROUTE_PER_INTENT,
    seed: int = SAMPLE_SEED,
) -> pd.DataFrame:
    """T-ROUTE: a fixed number of Banking77 test items per intent (stratified)."""
    _require_columns(frame, "sample_route")
    intents = sorted(str(value) for value in frame["intent"].unique())
    if len(intents) != len(BANKING77_LABELS):
        msg = f"expected {len(BANKING77_LABELS)} Banking77 intents, found {len(intents)}"
        raise ValueError(msg)
    rng = _rng(seed, "route")
    picked: list[str] = []
    for intent in intents:
        ids = sorted(frame.loc[frame["intent"] == intent, "item_id"].astype(str))
        if len(ids) < per_intent:
            msg = f"intent {intent!r} has {len(ids)} items, need {per_intent}"
            raise ValueError(msg)
        picked.extend(rng.sample(ids, per_intent))
    return _select(frame, picked)


def sample_scope(
    frame: pd.DataFrame,
    *,
    per_class: int = SCOPE_PER_GROUP,
    seed: int = SAMPLE_SEED,
) -> pd.DataFrame:
    """T-SCOPE: ``per_class`` out-of-scope plus ``per_class`` in-scope items.

    In-scope items are spread round-robin over the 150 in-scope intents, so the group is
    stratified even though CLINC150's test split is not balanced.
    """
    _require_columns(frame, "sample_scope")
    intents = sorted(str(value) for value in frame["intent"].unique())
    in_scope_intents = [intent for intent in intents if intent != CLINC_OOS_LABEL]
    if len(in_scope_intents) != len(CLINC150_LABELS) - 1:
        msg = (
            f"expected {len(CLINC150_LABELS) - 1} CLINC150 in-scope intents, "
            f"found {len(in_scope_intents)}"
        )
        raise ValueError(msg)
    oos_ids = sorted(frame.loc[frame["intent"] == CLINC_OOS_LABEL, "item_id"].astype(str))
    if len(oos_ids) < per_class:
        msg = f"{len(oos_ids)} out-of-scope items, need {per_class}"
        raise ValueError(msg)

    rng = _rng(seed, "scope")
    remaining = {
        intent: sorted(frame.loc[frame["intent"] == intent, "item_id"].astype(str))
        for intent in in_scope_intents
    }
    in_scope_picked: list[str] = []
    while len(in_scope_picked) < per_class:
        for intent in in_scope_intents:
            pool = remaining[intent]
            if not pool:
                msg = f"intent {intent!r} has no items left for the scope sample"
                raise ValueError(msg)
            in_scope_picked.append(pool.pop(rng.randrange(len(pool))))
            if len(in_scope_picked) >= per_class:
                break

    picked = in_scope_picked + rng.sample(oos_ids, per_class)
    return _select(frame, picked)


def sample_reply(
    route_sample: pd.DataFrame,
    *,
    n: int = REPLY_ITEMS,
    seed: int = SAMPLE_SEED,
) -> pd.DataFrame:
    """T-REPLY: a seeded subset of the T-ROUTE sample."""
    _require_columns(route_sample, "sample_reply")
    ids = sorted(route_sample["item_id"].astype(str))
    if len(ids) < n:
        msg = f"route sample has {len(ids)} items, need {n} for T-REPLY"
        raise ValueError(msg)
    picked = _rng(seed, "reply").sample(ids, n)
    return _select(route_sample, picked)


def _write_ids(path: Path, ids: Sequence[str]) -> None:
    path.write_text("".join(f"{item_id}\n" for item_id in sorted(ids)), encoding="utf-8")


def write_sample_ids(
    samples: dict[str, pd.DataFrame],
    *,
    samples_dir: Path = SAMPLE_DIR,
) -> Path:
    """Write the committed id lists and their ``sha256sum``-style manifest."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    for name in SAMPLE_FILES:
        task = name.removesuffix(".txt")
        _write_ids(samples_dir / name, samples[task]["item_id"].astype(str).tolist())
    manifest = samples_dir / "SAMPLES.sha256"
    lines = []
    for name in SAMPLE_FILES:
        digest = hashlib.sha256((samples_dir / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def build_samples(
    seed: int = SAMPLE_SEED,
    *,
    data_root: Path = RAW_DIR,
    samples_dir: Path = SAMPLE_DIR,
) -> dict[str, pd.DataFrame]:
    """Build the three §7 samples and write their committed id lists."""
    banking_test = load_banking77("test", root=data_root)
    clinc_test = load_clinc150("test", root=data_root)
    route = sample_route(banking_test, seed=seed)
    scope = sample_scope(clinc_test, seed=seed)
    reply = sample_reply(route, seed=seed)
    samples = {"route": route, "scope": scope, "reply": reply}
    write_sample_ids(samples, samples_dir=samples_dir)
    return samples


def load_sample_ids(
    task: str,
    *,
    samples_dir: Path = SAMPLE_DIR,
) -> list[str]:
    """Read one committed id list (``route``, ``scope`` or ``reply``)."""
    path = samples_dir / f"{task}.txt"
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not ids or any(not item_id for item_id in ids):
        msg = f"{path}: empty id list or blank line"
        raise ValueError(msg)
    if len(ids) != len(set(ids)):
        msg = f"{path}: duplicate item ids"
        raise ValueError(msg)
    return ids


def sample_hash(samples_dir: Path = SAMPLE_DIR) -> str:
    """The single sample hash recorded in ``RunManifest``.

    Hashes ``route.txt``, ``scope.txt`` and ``reply.txt`` in that fixed order, with the
    file name mixed in so swapping two lists changes the hash.
    """
    digest = hashlib.sha256()
    for name in SAMPLE_FILES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((samples_dir / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=RAW_DIR)
    parser.add_argument("--samples", type=Path, default=SAMPLE_DIR)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args(argv)
    try:
        samples = build_samples(
            args.seed,
            data_root=args.data_root,
            samples_dir=args.samples,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"samples: {exc}\n")
        sys.stderr.write("samples: run `make fetch-data` first.\n")
        return 1
    for name in SAMPLE_FILES:
        task = name.removesuffix(".txt")
        sys.stdout.write(f"samples: {name} {len(samples[task])} ids\n")
    sys.stdout.write(f"samples: hash {sample_hash(args.samples)}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI
    raise SystemExit(main())
