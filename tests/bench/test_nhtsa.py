"""The NHTSA reader: parsing, the column drop, the rules, the splits and the sample.

Every test runs on small synthetic tab-delimited files written into ``tmp_path``. The
committed extracts under ``data/defect/`` are read only where the test is about them,
and nothing here reads the scout's snapshot folder.
"""

from __future__ import annotations

import gzip
import json
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from which_ai_model.bench.nhtsa import (
    COMPLAINT_COLUMNS,
    ITEM_FIELDS,
    DefectItem,
    NhtsaDataError,
    NhtsaSpec,
    SpecFile,
    allocate,
    build_dataset,
    compile_pii,
    draw_sample,
    draw_stratified,
    ensure_source,
    file_sha256,
    filter_complaints,
    halving_half,
    load_groups,
    load_spec,
    match_group,
    normalise_component,
    order_key,
    parse_row,
    pii_hits,
    read_complaints,
    scan_complaints,
    split_period,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NARRATIVE = "The airbag warning lamp stays on after the recall repair was completed by the dealer."

BASE_ROW: dict[str, str] = {
    "CMPLID": "1",
    "ODINO": "10000001",
    "MFR_NAME": "Test Motors",
    "MAKETXT": "test",
    "MODELTXT": "model a",
    "YEARTXT": "2021",
    "CRASH": "N",
    "FAILDATE": "20240601",
    "FIRE": "N",
    "INJURED": "0",
    "DEATHS": "0",
    "COMPDESC": "AIR BAGS:FRONTAL",
    "CITY": "Springfield",
    "STATE": "IL",
    "VIN": "1HGCM82633A",
    "DATEA": "20250101",
    "LDATE": "20250101",
    "MILES": "",
    "OCCURENCES": "",
    "CDESCR": NARRATIVE,
    "CMPL_TYPE": "IVOQ",
    "POLICE_RPT_YN": "N",
    "PURCH_DT": "",
    "ORIG_OWNER_YN": "N",
    "ANTI_BRAKES_YN": "N",
    "CRUISE_CONT_YN": "N",
    "NUM_CYLS": "",
    "DRIVE_TRAIN": "",
    "FUEL_SYS": "",
    "FUEL_TYPE": "",
    "TRANS_TYPE": "",
    "VEH_SPEED": "",
    "DOT": "",
    "TIRE_SIZE": "",
    "LOC_OF_TIRE": "",
    "TIRE_FAIL_TYPE": "",
    "ORIG_EQUIP_YN": "",
    "MANUF_DT": "",
    "SEAT_TYPE": "",
    "RESTRAINT_TYPE": "",
    "DEALER_NAME": "Dealer Of Springfield",
    "DEALER_TEL": "555-0100",
    "DEALER_CITY": "Springfield",
    "DEALER_STATE": "IL",
    "DEALER_ZIP": "62701",
    "PROD_TYPE": "V",
    "REPAIRED_YN": "",
    "MEDICAL_ATTN": "",
    "VEHICLES_TOWED_YN": "",
    "STATE_OF_INCIDENT": "IL",
    "VEHICLE_OPERATOR": "Test Operator",
}


def make_row(**overrides: str) -> str:
    """One 51-field complaint row in file order, with a sane default per column."""
    values = dict(BASE_ROW)
    values.update(overrides)
    return "\t".join(values[name] for name in COMPLAINT_COLUMNS)


def write_source(directory: Path, rows: list[str], *, name: str = "complaints.txt") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def small_spec(**overrides: object) -> NhtsaSpec:
    """The committed spec, pointed at one plain-text file and small enough for a test."""
    spec = load_spec(REPO_ROOT / "data/defect/nhtsa-spec.toml")
    files = (SpecFile(name="complaints.txt", url="", sha256="", size_bytes=0, role="complaints"),)
    defaults: dict[str, object] = {
        "files": files,
        "archive_members": {},
        "wam_n": 10,
        "wam_min_per_group": 2,
        "wam_min_safety": 3,
        "train_max_items": 6,
    }
    defaults.update(overrides)
    return replace(spec, **defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- parsing


def test_parse_row_needs_every_field_and_strips_crlf() -> None:
    fields = parse_row(make_row() + "\r\n")
    assert len(fields) == 51
    assert fields[1] == "10000001"
    with pytest.raises(NhtsaDataError, match="expected 51"):
        parse_row("one\ttwo")


def test_row_keeps_the_specs_columns_and_drops_the_rest() -> None:
    spec = load_spec(REPO_ROOT / "data/defect/nhtsa-spec.toml")
    kept = set(spec.kept_columns)
    identity = set(spec.identity_columns)
    assert kept.isdisjoint(identity)
    assert len(kept | identity | set(spec.dropped_columns)) == len(COMPLAINT_COLUMNS)
    assert {"CITY", "STATE", "VIN", "VEHICLE_OPERATOR", "DEALER_TEL"} <= identity


def test_read_and_scan_never_carry_identity_values(tmp_path: Path) -> None:
    spec = small_spec()
    source = write_source(
        tmp_path,
        [make_row(ODINO="10000001", CITY="Springfield", VIN="1HGCM82633A004352")],
    )
    scan = scan_complaints(read_complaints(source.parent, spec), load_groups())
    assert len(scan.complaints) == 1
    complaint = scan.complaints[0]
    assert complaint.make == "TEST"
    assert complaint.model == "MODEL A"
    assert complaint.year == "2021"
    assert complaint.mileage is None
    payload = json.dumps(
        {
            "id": complaint.odino,
            "make": complaint.make,
            "model": complaint.model,
            "narrative": complaint.narrative,
        }
    )
    for secret in ("Springfield", "1HGCM82633A004352", "Test Operator", "Dealer Of Springfield"):
        assert secret not in payload


def test_unknown_model_year_stays_a_string(tmp_path: Path) -> None:
    spec = small_spec()
    source = write_source(tmp_path, [make_row(YEARTXT="9999")])
    scan = scan_complaints(read_complaints(source.parent, spec), load_groups())
    assert scan.complaints[0].year == "9999"


# ------------------------------------------------------------------------ components


def test_normalise_and_longest_prefix_wins() -> None:
    groups = load_groups()
    assert normalise_component("  air bags : frontal  ") == "AIR BAGS:FRONTAL"
    assert match_group("AIR BAGS:FRONTAL", groups) == "airbags"
    assert match_group("electrical system:wiring", groups) == "battery_charging_12v"
    assert (
        match_group("ELECTRICAL SYSTEM:WIRING:SWITCHES/KNOBS/BUTTONS", groups)
        == "power_windows_locks_switches"
    )
    assert match_group("ELECTRICAL SYSTEM", groups) is None
    assert match_group("ENGINE", groups) is None


def test_single_group_rule_keeps_only_one_group_per_complaint(tmp_path: Path) -> None:
    spec = small_spec()
    rows = [
        make_row(ODINO="10000001", COMPDESC="AIR BAGS:FRONTAL"),
        make_row(ODINO="10000001", COMPDESC="AIR BAGS:SIDE"),
        make_row(ODINO="10000002", COMPDESC="AIR BAGS:FRONTAL"),
        make_row(ODINO="10000002", COMPDESC="VISIBILITY/WIPER"),
        make_row(ODINO="10000003", COMPDESC="AIR BAGS:FRONTAL"),
        make_row(ODINO="10000003", COMPDESC="ENGINE"),
        make_row(ODINO="10000004", COMPDESC="AIR BAGS:FRONTAL"),
        make_row(ODINO="10000004", COMPDESC="VISIBILITY/WIPER", PROD_TYPE="T"),
    ]
    scan = scan_complaints(
        read_complaints(write_source(tmp_path, rows).parent, spec), load_groups()
    )
    assert sorted(complaint.odino for complaint in scan.complaints) == ["10000001", "10000004"]
    assert scan.excluded == 2
    assert scan.rows == 8
    assert scan.vehicle_rows == 7
    assert scan.vehicle_complaints == 4


def test_safety_label_rule(tmp_path: Path) -> None:
    spec = small_spec()
    rows = [
        make_row(ODINO="1", CRASH="Y"),
        make_row(ODINO="2", FIRE="Y"),
        make_row(ODINO="3", INJURED="2"),
        make_row(ODINO="4", DEATHS="1"),
        make_row(ODINO="5"),
    ]
    scan = scan_complaints(
        read_complaints(write_source(tmp_path, rows).parent, spec), load_groups()
    )
    flagged = {complaint.odino for complaint in scan.complaints if complaint.safety_relevant}
    assert flagged == {"1", "2", "3", "4"}


# --------------------------------------------------------------------- narrative filters


def test_length_filter_counts_characters_after_trimming(tmp_path: Path) -> None:
    spec = small_spec()
    rows = [
        make_row(ODINO="1", CDESCR="  " + "a" * 79 + "  "),
        make_row(ODINO="2", CDESCR="  " + "a" * 80 + "  "),
    ]
    scan = scan_complaints(
        read_complaints(write_source(tmp_path, rows).parent, spec), load_groups()
    )
    kept, stats = filter_complaints(scan.complaints, spec)
    assert stats.too_short == 1
    assert [complaint.odino for complaint in kept] == ["2"]


PII_TEXTS = {
    "vin17": "The vehicle identification number is 1HGCM82633A004352 and the warning lamp stays on.",
    "email": "Please contact the vehicle owner at owner.reports@example.com about the warning lamp.",
    "phone_us": "The dealer can be reached on (555) 123-4567 about the warning lamp that stays on.",
    "street": "The repair was done at 123 MAIN STREET and the warning lamp still stays on after.",
}
CLEAN_TEXT = (
    "The warning lamp stays on after the repair and the dealer says the part is on back order."
)


def test_pii_screen_drops_each_pattern_and_counts_it(tmp_path: Path) -> None:
    spec = small_spec()
    rows = [
        make_row(ODINO=str(index), CDESCR=text)
        for index, text in enumerate([*PII_TEXTS.values(), CLEAN_TEXT], start=1)
    ]
    scan = scan_complaints(
        read_complaints(write_source(tmp_path, rows).parent, spec), load_groups()
    )
    kept, stats = filter_complaints(scan.complaints, spec)
    assert [complaint.odino for complaint in kept] == ["5"]
    assert stats.pii_dropped == len(PII_TEXTS)
    assert set(stats.pii_by_pattern) == {name for name, _ in spec.pii_patterns}


def test_pii_patterns_match_only_their_own_text() -> None:
    spec = small_spec()
    compiled = compile_pii(spec.pii_patterns)
    for name, text in PII_TEXTS.items():
        assert pii_hits(text, compiled) == (name,), name
    assert pii_hits(CLEAN_TEXT, compiled) == ()
    assert (
        pii_hits("the firefly lamp is dim and the wipers chatter on a dry screen", compiled) == ()
    )


def test_a_pii_narrative_is_never_redacted_in_place(tmp_path: Path) -> None:
    spec = small_spec()
    source = write_source(tmp_path, [make_row(CDESCR=PII_TEXTS["phone_us"])])
    scan = scan_complaints(read_complaints(source.parent, spec), load_groups())
    kept, _ = filter_complaints(scan.complaints, spec)
    assert kept == []


# ------------------------------------------------------------------ halves and seeds


def test_halving_is_deterministic_and_disjoint() -> None:
    ids = [str(11_000_000 + index) for index in range(400)]
    halves = {value: halving_half(value, "20261031:") for value in ids}
    assert set(halves.values()) == {"wam", "frt"}
    wam = {value for value, half in halves.items() if half == "wam"}
    frt = {value for value, half in halves.items() if half == "frt"}
    assert wam & frt == set()
    assert wam | frt == set(ids)
    again = {value: halving_half(value, "20261031:") for value in ids}
    assert halves == again
    other = {value: halving_half(value, "19990101:") for value in ids}
    assert other != halves


def test_order_key_is_the_full_hash_and_depends_on_the_seed() -> None:
    key = order_key("11633475", 20261031)
    assert len(key) == 64
    assert key == order_key("11633475", 20261031)
    assert key != order_key("11633475", 20261030)
    assert int(key, 16) % 2 == (0 if halving_half("11633475", "20261031:") == "wam" else 1)


def test_split_period_uses_the_received_date() -> None:
    spec = small_spec()
    assert split_period("20200101", spec) == "train"
    assert split_period("20241231", spec) == "train"
    assert split_period("20250101", spec) == "test"
    assert split_period("20261031", spec) == "test"
    assert split_period("20191231", spec) == "outside"


# ------------------------------------------------------------------------- sampling


def test_allocate_respects_the_floor_and_caps_at_the_pool() -> None:
    quotas = allocate({"a": 5, "b": 100, "c": 200}, 50, 3)
    assert sum(quotas.values()) == 50
    assert 3 <= quotas["a"] <= 5
    assert quotas["b"] >= 3
    assert quotas["c"] >= 3
    assert allocate({"a": 5, "b": 100, "c": 200}, 50, 3) == quotas

    capped = allocate({"a": 3, "b": 100}, 50, 3)
    assert capped["a"] == 3
    assert sum(capped.values()) == 50

    with pytest.raises(NhtsaDataError, match="floor"):
        allocate({"a": 3, "b": 3}, 4, 3)


def _items(per_group: int, groups: list[str]) -> list[DefectItem]:
    items: list[DefectItem] = []
    for group in groups:
        for index in range(per_group):
            number = len(items)
            items.append(
                DefectItem(
                    id=f"{20_000_000 + number}",
                    text="x" * 80,
                    make="TEST",
                    model="MODEL",
                    year="2025",
                    received="20250101",
                    component_group=group,
                    safety_relevant=index % 3 == 0,
                )
            )
    return items


def test_draw_sample_meets_the_floors_and_is_deterministic() -> None:
    groups = ["airbags", "wipers_washers", "exterior_lighting", "driver_assist_sensors"]
    items = _items(20, groups)
    first = draw_sample(items, n=40, min_per_group=3, min_safety=15, seed=20261031)
    second = draw_sample(items, n=40, min_per_group=3, min_safety=15, seed=20261031)
    assert [item.id for item in first] == [item.id for item in second]
    assert len(first) == 40
    per_group = {
        group: sum(1 for item in first if item.component_group == group) for group in groups
    }
    assert all(count >= 3 for count in per_group.values()), per_group
    assert sum(1 for item in first if item.safety_relevant) >= 15
    other = draw_sample(items, n=40, min_per_group=3, min_safety=15, seed=7)
    assert [item.id for item in other] != [item.id for item in first]


def test_draw_sample_fails_loudly_when_a_group_is_too_small() -> None:
    items = _items(20, ["airbags"]) + _items(1, ["wipers_washers"])
    with pytest.raises(NhtsaDataError, match="fewer than"):
        draw_sample(items, n=10, min_per_group=3, min_safety=0, seed=1)


def test_draw_stratified_takes_at_most_the_maximum() -> None:
    items = _items(20, ["airbags", "wipers_washers"])
    chosen = draw_stratified(items, n=12, seed=20261031)
    assert len(chosen) == 12
    assert {item.component_group for item in chosen} == {"airbags", "wipers_washers"}
    assert [item.id for item in chosen] == [
        item.id for item in draw_stratified(items, n=12, seed=20261031)
    ]


# ------------------------------------------------------------------------ the build


def _build_source(directory: Path) -> Path:
    rows: list[str] = []
    odino = 30_000_000
    for component in ("AIR BAGS:FRONTAL", "VISIBILITY/WIPER"):
        for index in range(40):
            odino += 1
            rows.append(
                make_row(
                    ODINO=str(odino),
                    COMPDESC=component,
                    LDATE="20250601",
                    CRASH="Y" if index % 5 == 0 else "N",
                )
            )
    for index in range(10):
        odino += 1
        rows.append(
            make_row(
                ODINO=str(odino),
                COMPDESC="AIR BAGS:FRONTAL",
                LDATE="20220601",
                INJURED="1" if index % 4 == 0 else "0",
            )
        )
    return write_source(directory, rows)


def test_build_dataset_writes_byte_identical_extracts(tmp_path: Path) -> None:
    source = _build_source(tmp_path / "raw")
    spec = small_spec()
    groups = load_groups()
    first = tmp_path / "out1"
    second = tmp_path / "out2"
    stats = build_dataset(spec=spec, groups=groups, source_dir=source.parent, out_dir=first)
    build_dataset(spec=spec, groups=groups, source_dir=source.parent, out_dir=second)

    for name in ("test.jsonl", "train.jsonl.gz", "CHECKSUMS.txt", "STATS.md"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name

    items = [json.loads(line) for line in (first / "test.jsonl").read_text().splitlines()]
    assert len(items) == spec.wam_n
    assert all(set(item) == set(ITEM_FIELDS) for item in items)
    assert all(halving_half(item["id"], spec.halving_prefix) == "wam" for item in items)
    assert stats.sample_safety_total >= spec.wam_min_safety
    assert stats.train_item_total == 6
    checksums = (first / "CHECKSUMS.txt").read_text().splitlines()
    assert checksums[0].startswith("# sha256")
    assert {line.split("\t")[-1] for line in checksums[1:]} == {"test.jsonl", "train.jsonl.gz"}


def test_build_refuses_to_replace_committed_extracts(tmp_path: Path) -> None:
    source = _build_source(tmp_path / "raw")
    out = tmp_path / "out"
    build_dataset(spec=small_spec(), groups=load_groups(), source_dir=source.parent, out_dir=out)
    with pytest.raises(NhtsaDataError, match="--rebuild"):
        build_dataset(
            spec=small_spec(), groups=load_groups(), source_dir=source.parent, out_dir=out
        )


# ------------------------------------------------------------------ source resolution


def test_ensure_source_prefers_a_matching_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "one.txt").write_text("pinned", encoding="utf-8")
    digest = file_sha256(snapshot / "one.txt")
    spec = replace(
        load_spec(REPO_ROOT / "data/defect/nhtsa-spec.toml"),
        files=(SpecFile("one.txt", "https://example.invalid/one.txt", digest, 6, "complaints"),),
        archive_members={},
    )
    calls: list[str] = []

    def downloader(url: str, dest: Path) -> None:  # pragma: no cover - must not run
        calls.append(url)

    source = ensure_source(
        spec, snapshot_dir=snapshot, dest=tmp_path / "raw", downloader=downloader
    )
    assert source.from_snapshot is True
    assert source.warnings == ()
    assert calls == []


def test_ensure_source_downloads_and_warns_but_does_not_stop(tmp_path: Path) -> None:
    spec = replace(
        load_spec(REPO_ROOT / "data/defect/nhtsa-spec.toml"),
        files=(
            SpecFile(
                "one.txt",
                "https://example.invalid/one.txt",
                "0" * 64,
                6,
                "complaints",
            ),
        ),
        archive_members={},
    )
    downloaded: list[Path] = []

    def downloader(url: str, dest: Path) -> None:
        downloaded.append(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("other!", encoding="utf-8")

    source = ensure_source(
        spec, snapshot_dir=tmp_path / "missing", dest=tmp_path / "raw", downloader=downloader
    )
    assert source.from_snapshot is False
    assert source.directory == tmp_path / "raw"
    assert downloaded == [tmp_path / "raw" / "one.txt"]
    assert any("continuing" in warning for warning in source.warnings)


# --------------------------------------------------------------- the committed files


def test_committed_extracts_match_the_spec() -> None:
    spec = load_spec(REPO_ROOT / "data/defect/nhtsa-spec.toml")
    group_ids = {group.id for group in load_groups(REPO_ROOT / "data/defect/groups.toml")}

    test_items = [
        json.loads(line)
        for line in (REPO_ROOT / "data/defect/test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(test_items) == spec.wam_n
    assert len({item["id"] for item in test_items}) == spec.wam_n
    assert all(set(item) == set(ITEM_FIELDS) for item in test_items)
    assert all(item["component_group"] in group_ids for item in test_items)
    assert all(halving_half(item["id"], spec.halving_prefix) == spec.half_0 for item in test_items)
    counts: dict[str, int] = {}
    for item in test_items:
        counts[item["component_group"]] = counts.get(item["component_group"], 0) + 1
    assert set(counts) == group_ids
    assert min(counts.values()) >= spec.wam_min_per_group
    assert sum(1 for item in test_items if item["safety_relevant"]) >= spec.wam_min_safety
    assert all(isinstance(item["safety_relevant"], bool) for item in test_items)

    with gzip.open(REPO_ROOT / "data/defect/train.jsonl.gz", "rt", encoding="utf-8") as handle:
        train_items = [json.loads(line) for line in handle if line.strip()]
    assert 0 < len(train_items) <= spec.train_max_items
    assert all(set(item) == set(ITEM_FIELDS) for item in train_items)
    assert all(item["component_group"] in group_ids for item in train_items)
    assert all(spec.train_start <= item["received"] <= spec.train_end for item in train_items)


def test_committed_checksums_verify() -> None:
    directory = REPO_ROOT / "data/defect"
    for line in (directory / "CHECKSUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, size, name = line.split("\t")
        path = directory / name
        assert path.is_file(), name
        assert file_sha256(path) == digest, name
        assert path.stat().st_size == int(size), name


def test_the_committed_spec_documents_are_the_scout_snapshot() -> None:
    spec_document = tomllib.loads(
        (REPO_ROOT / "data/defect/nhtsa-spec.toml").read_text(encoding="utf-8")
    )
    assert spec_document["snapshot"] == "2026-10-31"
    groups_document = tomllib.loads(
        (REPO_ROOT / "data/defect/groups.toml").read_text(encoding="utf-8")
    )
    assert len(groups_document["group"]) == spec_document["groups"]["count"] == 8


def test_the_task_spec_labels_come_from_the_committed_groups() -> None:
    from which_ai_model.task.spec import load_labels, validate_spec
    from which_ai_model.task.spec import load_spec as load_task_spec

    task = load_task_spec(REPO_ROOT / "tasks/defect.toml")
    labels = load_labels(task, root=REPO_ROOT)
    assert len(labels) == 8
    assert "airbags" in labels
    assert all(description for description in labels.values())
    assert validate_spec(task, root=REPO_ROOT) == []
