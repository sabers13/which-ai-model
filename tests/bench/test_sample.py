"""Sampling determinism, sizes and the committed id-list format."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import pytest

from which_ai_model.bench import sample as sample_module
from which_ai_model.bench.data import BANKING77_LABELS, CLINC150_LABELS
from which_ai_model.bench.sample import (
    SAMPLE_FILES,
    SAMPLE_SEED,
    build_samples,
    load_sample_ids,
    sample_hash,
    sample_reply,
    sample_route,
    sample_scope,
    write_sample_ids,
)

B77_ID = re.compile(r"^b77-test-\d+$")
CLINC_ID = re.compile(r"^clinc-plus-test-\d+$")


def banking_test_frame(*, rows_per_intent: int = 10) -> pd.DataFrame:
    rows = []
    item = 0
    for intent_index in range(len(BANKING77_LABELS)):
        for _ in range(rows_per_intent):
            rows.append(
                {
                    "item_id": f"b77-test-{item}",
                    "text": f"banking text {item}",
                    "intent": BANKING77_LABELS[intent_index],
                }
            )
            item += 1
    return pd.DataFrame(rows)


def clinc_test_frame(*, rows_per_intent: int = 2, oos_rows: int = 200) -> pd.DataFrame:
    rows = []
    item = 0
    for intent in CLINC150_LABELS:
        count = oos_rows if intent == "oos" else rows_per_intent
        for _ in range(count):
            rows.append(
                {
                    "item_id": f"clinc-plus-test-{item}",
                    "text": f"clinc text {item}",
                    "intent": intent,
                }
            )
            item += 1
    return pd.DataFrame(rows)


def test_sample_route_is_deterministic_and_stratified() -> None:
    frame = banking_test_frame()
    first = sample_route(frame)
    second = sample_route(frame)

    assert first.equals(second)
    assert len(first) == len(BANKING77_LABELS) * 8
    assert first["item_id"].is_unique
    assert first["intent"].value_counts().eq(8).all()
    assert all(B77_ID.match(item_id) for item_id in first["item_id"])

    other = sample_route(frame, seed=SAMPLE_SEED + 1)
    assert not first["item_id"].equals(other["item_id"])


def test_sample_route_rejects_wrong_shape() -> None:
    frame = banking_test_frame()
    with pytest.raises(ValueError, match="Banking77 intents"):
        sample_route(frame[frame["intent"] != BANKING77_LABELS[0]])
    with pytest.raises(ValueError, match="need 8"):
        sample_route(banking_test_frame(rows_per_intent=2))


def test_sample_scope_is_150_oos_and_150_in_scope() -> None:
    frame = clinc_test_frame()
    first = sample_scope(frame)
    second = sample_scope(frame)

    assert first.equals(second)
    assert len(first) == 300
    assert first["item_id"].is_unique
    assert (first["intent"] == "oos").sum() == 150
    in_scope = first[first["intent"] != "oos"]
    assert len(in_scope) == 150
    assert in_scope["intent"].nunique() == len(CLINC150_LABELS) - 1
    assert in_scope["intent"].value_counts().eq(1).all()
    assert all(CLINC_ID.match(item_id) for item_id in first["item_id"])


def test_sample_scope_needs_enough_oos() -> None:
    with pytest.raises(ValueError, match="out-of-scope"):
        sample_scope(clinc_test_frame(oos_rows=10))


def test_sample_reply_is_a_seeded_subset_of_route() -> None:
    route = sample_route(banking_test_frame())
    reply = sample_reply(route)
    assert len(reply) == 150
    assert reply.equals(sample_reply(route))
    assert set(reply["item_id"]) <= set(route["item_id"])


def test_build_samples_writes_ids_only_and_a_stable_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sample_module,
        "load_banking77",
        lambda *args, **kwargs: banking_test_frame(),
    )
    monkeypatch.setattr(
        sample_module,
        "load_clinc150",
        lambda *args, **kwargs: clinc_test_frame(),
    )

    samples = build_samples(samples_dir=tmp_path)
    assert {name: len(frame) for name, frame in samples.items()} == {
        "route": 616,
        "scope": 300,
        "reply": 150,
    }
    assert all((tmp_path / name).exists() for name in SAMPLE_FILES)

    for name in SAMPLE_FILES:
        lines = (tmp_path / name).read_text(encoding="utf-8").splitlines()
        assert lines == sorted(lines)
        assert all(B77_ID.match(line) or CLINC_ID.match(line) for line in lines)
        assert not any("text" in line for line in lines)

    manifest = (tmp_path / "SAMPLES.sha256").read_text(encoding="utf-8").splitlines()
    assert len(manifest) == len(SAMPLE_FILES)
    for line, name in zip(manifest, SAMPLE_FILES, strict=True):
        digest, listed = line.split("  ")
        assert listed == name
        assert digest == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()

    first_hash = sample_hash(tmp_path)
    build_samples(samples_dir=tmp_path)
    assert sample_hash(tmp_path) == first_hash

    assert load_sample_ids("route", samples_dir=tmp_path) == sorted(
        (tmp_path / "route.txt").read_text(encoding="utf-8").splitlines()
    )

    (tmp_path / "reply.txt").write_text("b77-test-1\n", encoding="utf-8")
    assert sample_hash(tmp_path) != first_hash


def test_sample_hash_hashes_file_names_too(tmp_path: Path) -> None:
    for name in SAMPLE_FILES:
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    first = sample_hash(tmp_path)
    for name in SAMPLE_FILES:
        (tmp_path / name).write_text("y\n", encoding="utf-8")
    assert sample_hash(tmp_path) != first


def test_write_sample_ids_does_not_include_dataset_text(tmp_path: Path) -> None:
    route = pd.DataFrame(
        {
            "item_id": ["b77-test-3", "b77-test-1"],
            "text": ["secret text a", "secret text b"],
            "intent": ["a", "b"],
        }
    )
    write_sample_ids({"route": route, "scope": route, "reply": route}, samples_dir=tmp_path)
    written = (tmp_path / "route.txt").read_text(encoding="utf-8")
    assert written == "b77-test-1\nb77-test-3\n"
    assert "secret" not in written
