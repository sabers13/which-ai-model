"""Dataset loading, checksums and the fetch path — synthetic only, no network."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from which_ai_model.bench.data import (
    BANKING77_LABELS,
    CLINC150_LABELS,
    DATASET_FILES,
    ChecksumMismatchError,
    fetch_data,
    load_banking77,
    load_clinc150,
    read_checksums,
    verify_checksums,
    write_checksums,
)


def _banking_frame() -> pd.DataFrame:
    rows = [
        {"text": f"banking message {index}", "label": label_index}
        for label_index in range(len(BANKING77_LABELS))
        for index in range(2)
    ]
    return pd.DataFrame(rows)


def _clinc_frame() -> pd.DataFrame:
    rows = [
        {"text": f"clinc message {intent_index}", "intent": intent_index}
        for intent_index in range(len(CLINC150_LABELS))
    ]
    return pd.DataFrame(rows)


def test_load_banking77_uses_stable_ids_and_card_names(tmp_path: Path) -> None:
    _banking_frame().to_parquet(tmp_path / "banking77_test.parquet")
    frame = load_banking77("test", root=tmp_path)

    assert list(frame.columns) == ["item_id", "text", "intent"]
    assert len(frame) == len(BANKING77_LABELS) * 2
    assert frame.loc[0, "item_id"] == "b77-test-0"
    assert frame.loc[0, "intent"] == BANKING77_LABELS[0]
    assert frame.loc[2, "intent"] == BANKING77_LABELS[1]
    assert frame["item_id"].is_unique


def test_load_clinc150_uses_stable_ids_and_oos_name(tmp_path: Path) -> None:
    _clinc_frame().to_parquet(tmp_path / "clinc150_plus_test.parquet")
    frame = load_clinc150("test", root=tmp_path)

    assert frame.loc[0, "item_id"] == "clinc-plus-test-0"
    assert frame.loc[42, "intent"] == "oos"
    assert frame["intent"].tolist() == list(CLINC150_LABELS)
    assert load_clinc150("plus", root=tmp_path).equals(frame)


def test_load_unknown_split_and_label_index_are_loud(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        load_banking77("validation", root=tmp_path)

    bad = pd.DataFrame({"text": ["x"], "label": [len(BANKING77_LABELS)]})
    bad.to_parquet(tmp_path / "banking77_test.parquet")
    with pytest.raises(ValueError, match="outside the dataset card"):
        load_banking77("test", root=tmp_path)


def _fake_downloader(payload: bytes | None = None) -> tuple[list[str], object]:
    calls: list[str] = []
    mapping = {spec.url: payload or f"payload:{spec.name}".encode() for spec in DATASET_FILES}

    def download(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(mapping[url])

    return calls, download


def test_fetch_data_records_checksums_then_verifies_without_downloading(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    checksums = tmp_path / "data" / "CHECKSUMS.txt"
    calls, downloader = _fake_downloader()

    records = fetch_data(
        raw,
        checksums=checksums,
        reports_dir=tmp_path / "reports",
        downloader=downloader,  # type: ignore[arg-type]
        retrieved="2026-10-28",
    )

    assert len(calls) == len(DATASET_FILES)
    assert len(records) == len(DATASET_FILES)
    assert checksums.exists()
    stored = read_checksums(checksums)
    assert set(stored) == {spec.name for spec in DATASET_FILES}
    assert all(record.retrieved == "2026-10-28" for record in stored.values())
    assert all(record.url.startswith("https://huggingface.co/") for record in stored.values())
    verify_checksums(checksums, dest=raw)

    def unexpected(url: str, dest: Path) -> None:  # pragma: no cover - must not run
        raise AssertionError(f"downloaded again: {url}")

    fetch_data(raw, checksums=checksums, reports_dir=tmp_path / "reports", downloader=unexpected)
    assert not (tmp_path / "reports" / "STOP.md").exists()


def test_fetch_data_missing_file_redownloads_and_checks_known_digest(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    checksums = tmp_path / "data" / "CHECKSUMS.txt"
    _, downloader = _fake_downloader()
    fetch_data(raw, checksums=checksums, reports_dir=tmp_path / "reports", downloader=downloader)  # type: ignore[arg-type]
    before = read_checksums(checksums)

    first = DATASET_FILES[0]
    (raw / first.name).unlink()

    _, good = _fake_downloader()
    fetch_data(raw, checksums=checksums, reports_dir=tmp_path / "reports", downloader=good)  # type: ignore[arg-type]
    assert read_checksums(checksums) == before
    verify_checksums(checksums, dest=raw)


def test_fetch_data_wrong_redownload_writes_stop_and_keeps_raw(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    checksums = tmp_path / "data" / "CHECKSUMS.txt"
    reports = tmp_path / "reports"
    _, downloader = _fake_downloader()
    fetch_data(raw, checksums=checksums, reports_dir=reports, downloader=downloader)  # type: ignore[arg-type]

    first = DATASET_FILES[0]
    (raw / first.name).unlink()

    def wrong(url: str, dest: Path) -> None:
        dest.write_bytes(b"not the same bytes")

    with pytest.raises(ChecksumMismatchError):
        fetch_data(raw, checksums=checksums, reports_dir=reports, downloader=wrong)

    assert not (raw / first.name).exists()
    stop = (reports / "STOP.md").read_text(encoding="utf-8")
    assert "10.1" in stop
    assert "checksum mismatch" in stop.lower()


def test_fetch_data_tampered_existing_file_writes_stop(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    checksums = tmp_path / "data" / "CHECKSUMS.txt"
    reports = tmp_path / "reports"
    _, downloader = _fake_downloader()
    fetch_data(raw, checksums=checksums, reports_dir=reports, downloader=downloader)  # type: ignore[arg-type]

    target = raw / DATASET_FILES[0].name
    target.write_bytes(b"tampered")

    with pytest.raises(ChecksumMismatchError):
        fetch_data(raw, checksums=checksums, reports_dir=reports, downloader=downloader)  # type: ignore[arg-type]

    assert target.read_bytes() == b"tampered"
    assert (reports / "STOP.md").exists()


def test_checksum_file_roundtrip_and_malformed_lines(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "one.parquet").write_bytes(b"one")
    records = fetch_data(
        raw,
        checksums=tmp_path / "CHECKSUMS.txt",
        reports_dir=tmp_path / "reports",
        downloader=_only_first,
    )
    assert len(records) == len(DATASET_FILES)

    broken = tmp_path / "broken.txt"
    broken.write_text("# header\nnot-a-valid-row\n", encoding="utf-8")
    with pytest.raises(ValueError, match="5 tab-separated"):
        read_checksums(broken)

    empty = tmp_path / "empty.txt"
    write_checksums(empty, [])
    assert read_checksums(empty) == {}
    with pytest.raises(FileNotFoundError):
        verify_checksums(tmp_path / "missing.txt", dest=raw)


def _only_first(url: str, dest: Path) -> None:
    dest.write_bytes(b"payload")
