"""NHTSA ODI complaint data for T-DEFECT (``PLAN-v1.1`` §A4).

The binding specification is ``data/defect/nhtsa-spec.toml``, copied byte for byte from
the scout snapshot ``2026-10-31``, with its ``groups.toml`` beside it. Nothing in this
module decides a rule the spec does not state: it parses the complaint flat files, keeps
only the columns the spec keeps, applies the single-group rule, the 80-character minimum
and the PII screen, assigns the two labels and the test halves, and draws the committed
sample and the training set with the spec's deterministic procedure.

Two things are deliberately separate:

* the **raw files** stay outside the repository (a local snapshot folder or the
  gitignored ``data/raw/nhtsa/``) and are only read by :func:`ensure_source` and a build;
* the **derived files** under ``data/defect/`` are committed and are the source of truth.
  A rebuild needs ``--rebuild``/``force=True`` because NHTSA republishes its files and a
  fresh download must never silently replace a committed extract.

No test reads the snapshot folder: the suite drives the same functions on small
synthetic tab-delimited files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import re
import sys
import tomllib
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

__all__ = [
    "CHECKSUMS_HEADER",
    "CHECKSUMS_PATH",
    "COMPLAINT_COLUMNS",
    "DEFECT_DIR",
    "GROUPS_PATH",
    "RAW_DIR",
    "SPEC_PATH",
    "STATS_PATH",
    "TEST_PATH",
    "TRAIN_PATH",
    "BuildStats",
    "Complaint",
    "ComplaintRow",
    "ComponentGroup",
    "DefectItem",
    "NhtsaDataError",
    "NhtsaSpec",
    "ScanResult",
    "SpecFile",
    "allocate",
    "build_dataset",
    "draw_sample",
    "draw_stratified",
    "ensure_source",
    "file_sha256",
    "filter_complaints",
    "halving_half",
    "load_groups",
    "load_spec",
    "main",
    "match_group",
    "normalise_component",
    "order_key",
    "parse_row",
    "pii_hits",
    "read_complaints",
    "scan_complaints",
    "snapshot_problems",
    "split_period",
]

DEFECT_DIR: Final = Path("data/defect")
SPEC_PATH: Final = DEFECT_DIR / "nhtsa-spec.toml"
GROUPS_PATH: Final = DEFECT_DIR / "groups.toml"
TEST_PATH: Final = DEFECT_DIR / "test.jsonl"
TRAIN_PATH: Final = DEFECT_DIR / "train.jsonl.gz"
CHECKSUMS_PATH: Final = DEFECT_DIR / "CHECKSUMS.txt"
STATS_PATH: Final = DEFECT_DIR / "STATS.md"
RAW_DIR: Final = Path("data/raw/nhtsa")

CHECKSUMS_HEADER: Final = "# sha256\tbytes\tfile"
"""Committed checksum file for the derived extracts: one tab-separated row per file."""

#: The complaint flat file's 51 fields, in file order (``CMPL.txt``, 2026-04-30 layout).
#: The spec's kept, dropped and identity lists must partition exactly this tuple; that is
#: checked when the spec is loaded, so a layout change cannot silently shift a column.
COMPLAINT_COLUMNS: Final[tuple[str, ...]] = (
    "CMPLID",
    "ODINO",
    "MFR_NAME",
    "MAKETXT",
    "MODELTXT",
    "YEARTXT",
    "CRASH",
    "FAILDATE",
    "FIRE",
    "INJURED",
    "DEATHS",
    "COMPDESC",
    "CITY",
    "STATE",
    "VIN",
    "DATEA",
    "LDATE",
    "MILES",
    "OCCURENCES",
    "CDESCR",
    "CMPL_TYPE",
    "POLICE_RPT_YN",
    "PURCH_DT",
    "ORIG_OWNER_YN",
    "ANTI_BRAKES_YN",
    "CRUISE_CONT_YN",
    "NUM_CYLS",
    "DRIVE_TRAIN",
    "FUEL_SYS",
    "FUEL_TYPE",
    "TRANS_TYPE",
    "VEH_SPEED",
    "DOT",
    "TIRE_SIZE",
    "LOC_OF_TIRE",
    "TIRE_FAIL_TYPE",
    "ORIG_EQUIP_YN",
    "MANUF_DT",
    "SEAT_TYPE",
    "RESTRAINT_TYPE",
    "DEALER_NAME",
    "DEALER_TEL",
    "DEALER_CITY",
    "DEALER_STATE",
    "DEALER_ZIP",
    "PROD_TYPE",
    "REPAIRED_YN",
    "MEDICAL_ATTN",
    "VEHICLES_TOWED_YN",
    "STATE_OF_INCIDENT",
    "VEHICLE_OPERATOR",
)

_COLUMN_INDEX: Final[dict[str, int]] = {
    name: position for position, name in enumerate(COMPLAINT_COLUMNS)
}

#: Fields the model-facing extracts carry, in this order (task brief M2).
ITEM_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "text",
    "make",
    "model",
    "year",
    "received",
    "component_group",
    "safety_relevant",
)


class NhtsaDataError(RuntimeError):
    """The source files, the spec or the sample cannot satisfy ``PLAN-v1.1`` §A4."""


# --------------------------------------------------------------------------- the spec


@dataclass(frozen=True)
class SpecFile:
    """One pinned snapshot file: local name, URL, sha256 and size."""

    name: str
    url: str
    sha256: str
    size_bytes: int
    role: str


@dataclass(frozen=True)
class NhtsaSpec:
    """The parts of ``nhtsa-spec.toml`` this lane acts on."""

    snapshot: str
    snapshot_dir: Path
    files: tuple[SpecFile, ...]
    archive_members: Mapping[str, str]
    kept_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    identity_columns: tuple[str, ...]
    encoding: str
    delimiter: str
    field_count: int
    product_type: str
    min_chars: int
    pii_patterns: tuple[tuple[str, str], ...]
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    halving_prefix: str
    half_0: str
    half_1: str
    seed: int
    wam_n: int
    wam_min_per_group: int
    wam_min_safety: int
    train_max_items: int

    @property
    def complaint_files(self) -> tuple[SpecFile, ...]:
        return tuple(entry for entry in self.files if entry.role == "complaints")

    @property
    def kept_columns_in_file_order(self) -> tuple[str, ...]:
        """The kept columns in flat-file order, which is the order they are read in."""
        return tuple(name for name in COMPLAINT_COLUMNS if name in self.kept_columns)


def _require(document: Mapping[str, Any], key: str, where: str) -> Any:
    value = document.get(key)
    if value is None:
        msg = f"the NHTSA spec has no {where}.{key}"
        raise NhtsaDataError(msg)
    return value


def _yyyymmdd(value: str, where: str) -> str:
    """``2020-01-01`` as ``20200101``: the spec stores dates as YYYYMMDD strings."""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        msg = f"{where} must be YYYY-MM-DD, got {value!r}"
        raise NhtsaDataError(msg)
    return value.replace("-", "")


def load_spec(path: Path = SPEC_PATH) -> NhtsaSpec:
    """Read the binding spec and check that its column lists cover the flat file."""
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"cannot read the NHTSA spec {path}: {exc}"
        raise NhtsaDataError(msg) from exc
    except tomllib.TOMLDecodeError as exc:
        msg = f"{path} is not valid TOML: {exc}"
        raise NhtsaDataError(msg) from exc

    parse = _require(document, "parse", "spec")
    columns = _require(document, "columns", "spec")
    filters = _require(document, "filters", "spec")
    split = _require(document, "split", "spec")
    sampling = _require(document, "sampling", "spec")

    kept = tuple(str(name) for name in _require(columns, "complaint_kept", "columns"))
    dropped = tuple(str(name) for name in _require(columns, "complaint_dropped", "columns"))
    identity = tuple(
        str(name) for name in _require(columns, "identity_location_dropped", "columns")
    )
    covered = [*kept, *dropped, *identity]
    missing = sorted(set(COMPLAINT_COLUMNS) - set(covered))
    extra = sorted(set(covered) - set(COMPLAINT_COLUMNS))
    duplicated = sorted({name for name in covered if covered.count(name) > 1})
    if missing or extra or duplicated:
        msg = (
            f"{path}: the column lists do not partition the {len(COMPLAINT_COLUMNS)}-field "
            f"complaint layout (missing={missing}, unknown={extra}, duplicated={duplicated})"
        )
        raise NhtsaDataError(msg)

    files = tuple(
        SpecFile(
            name=str(_require(entry, "name", "files.entry")),
            url=str(_require(entry, "url", "files.entry")),
            sha256=str(_require(entry, "sha256", "files.entry")),
            size_bytes=int(_require(entry, "size_bytes", "files.entry")),
            role=str(_require(entry, "role", "files.entry")),
        )
        for entry in document.get("files", {}).get("entry", [])
    )
    archives = {
        str(_require(entry, "file", "parse.archive")): str(
            _require(entry, "member", "parse.archive")
        )
        for entry in parse.get("archive", [])
    }
    pii = _require(document, "pii", "spec")
    patterns = tuple(
        (str(name), str(pattern)) for name, pattern in _require(pii, "patterns", "pii").items()
    )
    wam = _require(sampling, "wam", "sampling")
    train = _require(sampling, "train", "sampling")

    return NhtsaSpec(
        snapshot=str(_require(document, "snapshot", "spec")),
        snapshot_dir=Path(str(_require(document, "snapshot_dir", "spec"))),
        files=files,
        archive_members=archives,
        kept_columns=kept,
        dropped_columns=dropped,
        identity_columns=identity,
        encoding=str(parse.get("encoding", "utf-8")),
        delimiter=str(parse.get("delimiter", "\t")),
        field_count=int(_require(parse, "complaint_field_count", "parse")),
        product_type=str(_require(filters, "product_type", "filters")),
        min_chars=int(_require(filters, "minimum_narrative_chars", "filters")),
        pii_patterns=patterns,
        train_start=_yyyymmdd(str(_require(split, "train_start", "split")), "split.train_start"),
        train_end=_yyyymmdd(str(_require(split, "train_end", "split")), "split.train_end"),
        test_start=_yyyymmdd(str(_require(split, "test_start", "split")), "split.test_start"),
        test_end=_yyyymmdd(str(_require(split, "test_end", "split")), "split.test_end"),
        halving_prefix=str(_require(split, "halving_key_prefix", "split")),
        half_0=str(_require(split, "half_0", "split")),
        half_1=str(_require(split, "half_1", "split")),
        seed=int(_require(sampling, "seed", "sampling")),
        wam_n=int(_require(wam, "n", "sampling.wam")),
        wam_min_per_group=int(_require(wam, "min_per_group", "sampling.wam")),
        wam_min_safety=int(_require(wam, "min_safety", "sampling.wam")),
        train_max_items=int(_require(train, "max_items", "sampling.train")),
    )


@dataclass(frozen=True)
class ComponentGroup:
    """One component group: its id, its one-line description and its component prefixes."""

    id: str
    description: str
    prefixes: tuple[str, ...]


def load_groups(path: Path = GROUPS_PATH) -> tuple[ComponentGroup, ...]:
    """Read ``groups.toml``: the eight groups and the prefixes that map to them."""
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"cannot read the component groups {path}: {exc}"
        raise NhtsaDataError(msg) from exc
    except tomllib.TOMLDecodeError as exc:
        msg = f"{path} is not valid TOML: {exc}"
        raise NhtsaDataError(msg) from exc

    groups: list[ComponentGroup] = []
    for entry in document.get("group", []):
        group_id = str(entry.get("id", "")).strip()
        description = str(entry.get("description", "")).strip()
        prefixes = tuple(str(prefix) for prefix in entry.get("prefixes", []))
        if not group_id or not description or not prefixes:
            msg = f"{path}: every [[group]] needs an id, a description and prefixes"
            raise NhtsaDataError(msg)
        groups.append(ComponentGroup(id=group_id, description=description, prefixes=prefixes))
    if not groups:
        msg = f"{path} lists no [[group]] entries"
        raise NhtsaDataError(msg)
    ids = [group.id for group in groups]
    if len(set(ids)) != len(ids):
        msg = f"{path}: a group id appears more than once ({ids})"
        raise NhtsaDataError(msg)
    return tuple(groups)


# ------------------------------------------------------------------------- components


def normalise_component(value: str) -> str:
    """Trim, uppercase, collapse whitespace and remove spaces around ``:`` (§groups.toml)."""
    collapsed = " ".join(value.split()).upper()
    return re.sub(r"\s*:\s*", ":", collapsed)


def match_group(value: str, groups: Sequence[ComponentGroup]) -> str | None:
    """The group whose prefix matches ``value`` longest, or ``None`` when unmapped."""
    normalised = normalise_component(value)
    best: str | None = None
    best_length = -1
    for group in groups:
        for prefix in group.prefixes:
            candidate = normalise_component(prefix)
            matches = normalised == candidate or normalised.startswith(candidate + ":")
            if matches and len(candidate) > best_length:
                best = group.id
                best_length = len(candidate)
    return best


# --------------------------------------------------------------------------- parsing


def parse_row(
    line: str,
    *,
    field_count: int = len(COMPLAINT_COLUMNS),
    delimiter: str = "\t",
) -> list[str]:
    """Split one flat-file line, tolerating the recall file's CRLF line endings."""
    fields = line.rstrip("\r\n").split(delimiter)
    if len(fields) != field_count:
        msg = f"expected {field_count} tab-separated fields, got {len(fields)}: {line[:120]!r}"
        raise NhtsaDataError(msg)
    return fields


def _count(value: str) -> int:
    text = value.strip()
    if not text:
        return 0
    try:
        return int(text)
    except ValueError:
        return 0


def _flag(value: str) -> bool:
    return value.strip().upper() == "Y"


@dataclass(frozen=True)
class ComplaintRow:
    """One complaint row (one component of one complaint) with the spec's kept columns."""

    odino: str
    make: str
    model: str
    year: str
    received: str
    failure_date: str
    mileage: int | None
    crash: bool
    fire: bool
    injured: int
    deaths: int
    component_description: str
    narrative: str
    product_type: str

    @property
    def safety_relevant(self) -> bool:
        """``crash == 'Y' or fire == 'Y' or injured > 0 or deaths > 0`` (§safety)."""
        return self.crash or self.fire or self.injured > 0 or self.deaths > 0


def complaint_row(fields: Sequence[str]) -> ComplaintRow:
    """Map one 51-field row onto :class:`ComplaintRow`; every other column is dropped."""
    if len(fields) != len(COMPLAINT_COLUMNS):
        msg = f"expected {len(COMPLAINT_COLUMNS)} fields, got {len(fields)}"
        raise NhtsaDataError(msg)
    mileage_text = fields[_COLUMN_INDEX["MILES"]].strip()
    return ComplaintRow(
        odino=fields[_COLUMN_INDEX["ODINO"]].strip(),
        make=fields[_COLUMN_INDEX["MAKETXT"]].strip(),
        model=fields[_COLUMN_INDEX["MODELTXT"]].strip(),
        year=fields[_COLUMN_INDEX["YEARTXT"]].strip(),
        received=fields[_COLUMN_INDEX["LDATE"]].strip(),
        failure_date=fields[_COLUMN_INDEX["FAILDATE"]].strip(),
        mileage=int(mileage_text) if mileage_text else None,
        crash=_flag(fields[_COLUMN_INDEX["CRASH"]]),
        fire=_flag(fields[_COLUMN_INDEX["FIRE"]]),
        injured=_count(fields[_COLUMN_INDEX["INJURED"]]),
        deaths=_count(fields[_COLUMN_INDEX["DEATHS"]]),
        component_description=fields[_COLUMN_INDEX["COMPDESC"]].strip(),
        narrative=fields[_COLUMN_INDEX["CDESCR"]].strip(),
        product_type=fields[_COLUMN_INDEX["PROD_TYPE"]].strip(),
    )


def _open_member(path: Path, member: str | None, encoding: str) -> Iterator[str]:
    """Yield the lines of a flat file, read straight out of a zip when it is one."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            name = member or archive.namelist()[0]
            with (
                archive.open(name) as raw,
                io.TextIOWrapper(raw, encoding=encoding, newline="") as text,
            ):
                yield from text
        return
    with path.open(encoding=encoding, newline="") as handle:
        yield from handle


def read_complaints(source_dir: Path, spec: NhtsaSpec) -> Iterator[ComplaintRow]:
    """Every complaint row in the pinned complaint files, in file order."""
    entries = spec.complaint_files
    if not entries:
        msg = f"{SPEC_PATH} pins no complaint files (role = 'complaints')"
        raise NhtsaDataError(msg)
    for entry in entries:
        path = source_dir / entry.name
        if not path.is_file():
            msg = f"missing complaint file {path}; run scripts/fetch_nhtsa.py first"
            raise NhtsaDataError(msg)
        member = spec.archive_members.get(entry.name)
        for line in _open_member(path, member, spec.encoding):
            if not line.strip():
                continue
            yield complaint_row(
                parse_row(line, field_count=spec.field_count, delimiter=spec.delimiter)
            )


# --------------------------------------------------------------- complaints and labels


@dataclass(frozen=True)
class Complaint:
    """A complaint whose component rows all map to one group, with both labels."""

    odino: str
    make: str
    model: str
    year: str
    received: str
    failure_date: str
    mileage: int | None
    crash: bool
    fire: bool
    injured: int
    deaths: int
    component_description: str
    component_group: str
    narrative: str
    safety_relevant: bool


@dataclass
class _Accumulator:
    """The vehicle rows of one ``ODINO``, reduced to what the rules need."""

    first: ComplaintRow
    groups: set[str] = field(default_factory=set)
    unmapped: bool = False

    def add(self, row: ComplaintRow, group: str | None) -> None:
        if group is None:
            self.unmapped = True
        else:
            self.groups.add(group)

    @property
    def single_group(self) -> str | None:
        if self.unmapped or len(self.groups) != 1:
            return None
        return next(iter(self.groups))


@dataclass(frozen=True)
class ScanResult:
    """What reading the raw files produced, before the narrative filters."""

    complaints: tuple[Complaint, ...]
    rows: int
    vehicle_rows: int
    vehicle_complaints: int
    excluded: int


def scan_complaints(
    rows: Iterable[ComplaintRow],
    groups: Sequence[ComponentGroup],
    *,
    product_type: str = "V",
) -> ScanResult:
    """Apply the product-type filter and the single-group rule to every complaint.

    A complaint counts only when every one of its vehicle component rows maps to one and
    the same group; an unmapped row or a second group excludes it whole.
    """
    total = 0
    vehicle_rows = 0
    accumulators: dict[str, _Accumulator] = {}
    for row in rows:
        total += 1
        if row.product_type != product_type:
            continue
        vehicle_rows += 1
        accumulator = accumulators.get(row.odino)
        if accumulator is None:
            accumulator = _Accumulator(first=row)
            accumulators[row.odino] = accumulator
        accumulator.add(row, match_group(row.component_description, groups))

    complaints: list[Complaint] = []
    excluded = 0
    for accumulator in accumulators.values():
        group = accumulator.single_group
        if group is None:
            excluded += 1
            continue
        first = accumulator.first
        complaints.append(
            Complaint(
                odino=first.odino,
                make=first.make.upper(),
                model=first.model.upper(),
                year=first.year,
                received=first.received,
                failure_date=first.failure_date,
                mileage=first.mileage,
                crash=first.crash,
                fire=first.fire,
                injured=first.injured,
                deaths=first.deaths,
                component_description=first.component_description,
                component_group=group,
                narrative=first.narrative,
                safety_relevant=first.safety_relevant,
            )
        )
    return ScanResult(
        complaints=tuple(complaints),
        rows=total,
        vehicle_rows=vehicle_rows,
        vehicle_complaints=len(accumulators),
        excluded=excluded,
    )


# ---------------------------------------------------------------- narrative filters


def compile_pii(patterns: Sequence[tuple[str, str]]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Compile the spec's PII regexes once; an invalid pattern is a loud error."""
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for name, pattern in patterns:
        try:
            compiled.append((name, re.compile(pattern)))
        except re.error as exc:  # pragma: no cover - the pinned spec compiles
            msg = f"the PII pattern {name!r} does not compile: {exc}"
            raise NhtsaDataError(msg) from exc
    return tuple(compiled)


def pii_hits(
    text: str,
    compiled: Sequence[tuple[str, re.Pattern[str]]],
) -> tuple[str, ...]:
    """Names of every PII pattern the narrative matches (empty means it passes)."""
    return tuple(name for name, pattern in compiled if pattern.search(text))


@dataclass(frozen=True)
class FilterStats:
    """Counts for the two narrative filters, in the order they are applied."""

    single_group: int
    too_short: int
    pii_by_pattern: Mapping[str, int]
    pii_dropped: int
    kept: int


def filter_complaints(
    complaints: Sequence[Complaint],
    spec: NhtsaSpec,
    *,
    compiled: Sequence[tuple[str, re.Pattern[str]]] | None = None,
) -> tuple[list[Complaint], FilterStats]:
    """Drop short narratives, then every narrative that matches a PII pattern.

    The length is measured after trimming; a narrative that matches any pattern is
    dropped whole and never redacted in place (§filters.pii_action).
    """
    patterns = compile_pii(spec.pii_patterns) if compiled is None else compiled
    kept: list[Complaint] = []
    too_short = 0
    pii_dropped = 0
    by_pattern: Counter[str] = Counter()
    for complaint in complaints:
        if len(complaint.narrative) < spec.min_chars:
            too_short += 1
            continue
        hits = pii_hits(complaint.narrative, patterns)
        if hits:
            pii_dropped += 1
            by_pattern.update(hits)
            continue
        kept.append(complaint)
    stats = FilterStats(
        single_group=len(complaints),
        too_short=too_short,
        pii_by_pattern=dict(sorted(by_pattern.items())),
        pii_dropped=pii_dropped,
        kept=len(kept),
    )
    return kept, stats


# ------------------------------------------------------------------- splits and seeds


def file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(odino: str, prefix: str) -> str:
    return hashlib.sha256(f"{prefix}{odino}".encode()).hexdigest()


def halving_half(odino: str, prefix: str, *, half_0: str = "wam", half_1: str = "frt") -> str:
    """``int(sha256(prefix + ODINO), 16) % 2``: even is the first half, odd the second."""
    return half_0 if int(_hash(odino, prefix), 16) % 2 == 0 else half_1


def order_key(odino: str, seed: int) -> str:
    """The full lowercase hex digest used to order a pool for the deterministic draw."""
    return _hash(odino, f"{seed}:")


def split_period(received: str, spec: NhtsaSpec) -> str:
    """``train``, ``test`` or ``outside`` for one YYYYMMDD received date."""
    if spec.train_start <= received <= spec.train_end:
        return "train"
    if spec.test_start <= received <= spec.test_end:
        return "test"
    return "outside"


# ------------------------------------------------------------------------- sampling


def _largest_remainder(
    pool_sizes: Mapping[str, int],
    capacity: Mapping[str, int],
    remaining: int,
) -> dict[str, int]:
    """Share ``remaining`` in proportion to pool size, capped at each group's capacity."""
    added = dict.fromkeys(pool_sizes, 0)
    left = remaining
    while left > 0:
        open_groups = sorted(group for group in pool_sizes if capacity[group] > added[group])
        if not open_groups:
            break
        total_pool = sum(pool_sizes[group] for group in open_groups)
        exact = {group: left * pool_sizes[group] / total_pool for group in open_groups}
        for group in open_groups:
            room = capacity[group] - added[group]
            step = min(math.floor(exact[group]), room)
            added[group] += step
            left -= step
        order = sorted(
            open_groups,
            key=lambda group: (-(exact[group] - math.floor(exact[group])), group),
        )
        for group in order:
            if left == 0:
                break
            if added[group] < capacity[group]:
                added[group] += 1
                left -= 1
    return added


def allocate(
    pool_sizes: Mapping[str, int],
    n: int,
    min_per_group: int,
) -> dict[str, int]:
    """Per-group quotas: ``min(min_per_group, pool)`` each, then largest remainder."""
    quotas = {group: min(min_per_group, size) for group, size in pool_sizes.items()}
    base = sum(quotas.values())
    if base > n:
        msg = f"the per-group floor needs {base} items but the sample is {n}"
        raise NhtsaDataError(msg)
    capacity = {group: size - quotas[group] for group, size in pool_sizes.items()}
    added = _largest_remainder(pool_sizes, capacity, n - base)
    return {group: quotas[group] + added[group] for group in pool_sizes}


@dataclass(frozen=True)
class DefectItem:
    """One model-facing T-DEFECT item (the eight committed fields)."""

    id: str
    text: str
    make: str
    model: str
    year: str
    received: str
    component_group: str
    safety_relevant: bool

    def as_json(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in ITEM_FIELDS}


def _by_group(items: Iterable[DefectItem]) -> dict[str, list[DefectItem]]:
    grouped: dict[str, list[DefectItem]] = {}
    for item in items:
        grouped.setdefault(item.component_group, []).append(item)
    return grouped


def draw_stratified(
    items: Sequence[DefectItem],
    *,
    n: int,
    seed: int,
) -> list[DefectItem]:
    """At most ``n`` items, shared between groups in proportion to pool size."""
    grouped = _by_group(items)
    sizes = {group: len(pool) for group, pool in sorted(grouped.items())}
    quotas = allocate(sizes, min(n, len(items)), 0)
    chosen: list[DefectItem] = []
    for group, pool in sorted(grouped.items()):
        ordered = sorted(pool, key=lambda item: order_key(item.id, seed))
        chosen.extend(ordered[: quotas[group]])
    return sorted(chosen, key=lambda item: (item.component_group, order_key(item.id, seed)))


def draw_sample(
    items: Sequence[DefectItem],
    *,
    n: int,
    min_per_group: int,
    min_safety: int,
    seed: int,
) -> list[DefectItem]:
    """The spec's stratified draw with the safety top-up (§sampling).

    Every group first takes ``min(min_per_group, pool)`` items ordered by the full hash,
    the rest is shared by largest remainder, and safety-relevant items are swapped in
    from the group with the most of them left until ``min_safety`` is met.
    """
    grouped = _by_group(items)
    sizes = {group: len(pool) for group, pool in sorted(grouped.items())}
    short = sorted(group for group, size in sizes.items() if size < min_per_group)
    if short:
        msg = (
            f"group(s) {short} have fewer than the {min_per_group} items the sample needs; "
            "the filters or the snapshot changed"
        )
        raise NhtsaDataError(msg)
    quotas = allocate(sizes, n, min_per_group)

    selected: dict[str, list[DefectItem]] = {}
    unselected: dict[str, list[DefectItem]] = {}
    for group, pool in sorted(grouped.items()):
        ordered = sorted(pool, key=lambda item: order_key(item.id, seed))
        selected[group] = ordered[: quotas[group]]
        unselected[group] = ordered[quotas[group] :]

    safety = sum(1 for pool in selected.values() for item in pool if item.safety_relevant)
    while safety < min_safety:
        candidates = sorted(
            (
                group
                for group in selected
                if any(item.safety_relevant for item in unselected[group])
            ),
            key=lambda group: (
                -sum(1 for item in unselected[group] if item.safety_relevant),
                group,
            ),
        )
        swapped = False
        for group in candidates:
            incoming = min(
                (item for item in unselected[group] if item.safety_relevant),
                key=lambda item: order_key(item.id, seed),
            )
            outgoing = [item for item in selected[group] if not item.safety_relevant]
            if not outgoing:
                continue
            victim = max(outgoing, key=lambda item: order_key(item.id, seed))
            selected[group].remove(victim)
            selected[group].append(incoming)
            unselected[group].remove(incoming)
            unselected[group].append(victim)
            safety += 1
            swapped = True
            break
        if not swapped:
            break

    if safety < min_safety:
        msg = (
            f"the sample holds {safety} safety-relevant items but the spec needs "
            f"{min_safety}; the pool is too small"
        )
        raise NhtsaDataError(msg)

    chosen = [item for pool in selected.values() for item in pool]
    return sorted(chosen, key=lambda item: (item.component_group, order_key(item.id, seed)))


# ------------------------------------------------------------------- source handling


def snapshot_problems(source_dir: Path, spec: NhtsaSpec) -> list[str]:
    """Every pinned file that is missing or does not match its sha256 and size."""
    problems: list[str] = []
    for entry in spec.files:
        path = source_dir / entry.name
        if not path.is_file():
            problems.append(f"{entry.name}: missing from {source_dir}")
            continue
        size = path.stat().st_size
        if size != entry.size_bytes:
            problems.append(f"{entry.name}: {size} bytes, expected {entry.size_bytes}")
            continue
        actual = file_sha256(path)
        if actual != entry.sha256:
            problems.append(f"{entry.name}: sha256 {actual}, expected {entry.sha256}")
    return problems


Downloader = Callable[[str, Path], None]


def _download(url: str, dest: Path) -> None:
    """Stream one URL to ``dest`` (used only by :func:`ensure_source`)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)


@dataclass(frozen=True)
class Source:
    """Where a build reads its raw files, and any checksum warning worth printing."""

    directory: Path
    from_snapshot: bool
    warnings: tuple[str, ...]


def ensure_source(
    spec: NhtsaSpec,
    *,
    snapshot_dir: Path | None = None,
    dest: Path = RAW_DIR,
    downloader: Downloader = _download,
) -> Source:
    """The snapshot folder when every checksum matches, else a fresh download.

    A download that does not match the pinned sha256 is a **warning**, never a stop: the
    committed derived files stay the source of truth and a rebuild is an explicit act.
    """
    snapshot = spec.snapshot_dir if snapshot_dir is None else snapshot_dir
    problems = snapshot_problems(snapshot, spec)
    if not problems:
        return Source(directory=snapshot, from_snapshot=True, warnings=())

    warnings = [f"snapshot {snapshot}: {problem}" for problem in problems]
    warnings.append(f"downloading the pinned files into {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    for entry in spec.files:
        target = dest / entry.name
        if target.is_file() and snapshot_problems(dest, replace(spec, files=(entry,))) == []:
            continue
        downloader(entry.url, target)
        actual = file_sha256(target)
        size = target.stat().st_size
        if actual != entry.sha256 or size != entry.size_bytes:
            warnings.append(
                f"{entry.name}: downloaded sha256 {actual} ({size} bytes), "
                f"the spec pins {entry.sha256} ({entry.size_bytes} bytes); "
                "continuing, because a republished file is a warning, not a stop"
            )
    return Source(directory=dest, from_snapshot=False, warnings=tuple(warnings))


# ------------------------------------------------------------------------ the build


@dataclass(frozen=True)
class BuildStats:
    """Every count the build reports, computed from the data it read."""

    snapshot: str
    rows: int
    vehicle_rows: int
    vehicle_complaints: int
    single_group: int
    excluded: int
    too_short: int
    pii_by_pattern: Mapping[str, int]
    pii_dropped: int
    kept: int
    outside_window: int
    train_pool: Mapping[str, int]
    test_pool: Mapping[str, int]
    wam_pool: Mapping[str, int]
    frt_pool: Mapping[str, int]
    natural_safety: Mapping[str, int]
    sample_items: Mapping[str, int]
    sample_safety: Mapping[str, int]
    train_items: Mapping[str, int]
    seed: int
    sample_hash: str

    @property
    def train_total(self) -> int:
        return sum(self.train_pool.values())

    @property
    def test_total(self) -> int:
        return sum(self.test_pool.values())

    @property
    def wam_total(self) -> int:
        return sum(self.wam_pool.values())

    @property
    def frt_total(self) -> int:
        return sum(self.frt_pool.values())

    @property
    def sample_total(self) -> int:
        return sum(self.sample_items.values())

    @property
    def sample_safety_total(self) -> int:
        return sum(self.sample_safety.values())

    @property
    def natural_safety_total(self) -> int:
        return sum(self.natural_safety.values())

    @property
    def train_item_total(self) -> int:
        return sum(self.train_items.values())


def _item(complaint: Complaint) -> DefectItem:
    return DefectItem(
        id=complaint.odino,
        text=complaint.narrative,
        make=complaint.make,
        model=complaint.model,
        year=complaint.year,
        received=complaint.received,
        component_group=complaint.component_group,
        safety_relevant=complaint.safety_relevant,
    )


def _write_jsonl(path: Path, items: Sequence[DefectItem]) -> None:
    lines = [json.dumps(item.as_json(), ensure_ascii=False) for item in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_jsonl_gz(path: Path, items: Sequence[DefectItem]) -> None:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        for item in items:
            line = json.dumps(item.as_json(), ensure_ascii=False) + "\n"
            handle.write(line.encode("utf-8"))
    path.write_bytes(buffer.getvalue())


def _write_checksums(path: Path, files: Sequence[Path]) -> None:
    lines = [CHECKSUMS_HEADER]
    for file in sorted(files, key=lambda item: item.name):
        lines.append(f"{file_sha256(file)}\t{file.stat().st_size}\t{file.name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rate(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}%" if denominator else "n/a"


def _stats_markdown(
    stats: BuildStats,
    groups: Sequence[ComponentGroup],
    out_dir: Path,
) -> str:
    lines = [
        "# T-DEFECT data statistics",
        "",
        "Generated by `python -m which_ai_model.bench.nhtsa build` from the NHTSA ODI",
        f"snapshot **{stats.snapshot}** (`data/defect/nhtsa-spec.toml` and",
        "`data/defect/groups.toml`, copied byte for byte from the scout). The committed",
        "files below are the source of truth; NHTSA republishes its flat files, so a fresh",
        "download with a different sha256 is a warning and never rebuilds them by itself.",
        "",
        "## Counts",
        "",
        "| Step | Items |",
        "|---|---:|",
        f"| complaint rows read | {stats.rows:,} |",
        f"| vehicle rows (`PROD_TYPE = V`) | {stats.vehicle_rows:,} |",
        f"| vehicle complaints | {stats.vehicle_complaints:,} |",
        f"| excluded by the single-group rule | {stats.excluded:,} |",
        f"| single-group complaints | {stats.single_group:,} |",
        f"| dropped: narrative shorter than the minimum | {stats.too_short:,} |",
        f"| dropped: PII screen | {stats.pii_dropped:,} |",
        f"| kept | {stats.kept:,} |",
        f"| dropped: received date outside both windows | {stats.outside_window:,} |",
        f"| train pool (2020-01-01 to 2024-12-31) | {stats.train_total:,} |",
        f"| test pool (2025-01-01 to 2026-10-31) | {stats.test_total:,} |",
        f"| test half `wam` | {stats.wam_total:,} |",
        f"| test half `frt` | {stats.frt_total:,} |",
        f"| `data/defect/test.jsonl` (the sample) | {stats.sample_total:,} |",
        f"| `data/defect/train.jsonl.gz` | {stats.train_item_total:,} |",
        "",
        "## PII screen",
        "",
        "Counted on the narratives that already passed the length filter. One narrative can",
        "match several patterns, so the rows below add up to more than the total dropped.",
        "",
        "| Pattern | Narratives dropped |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | {count:,} |" for name, count in sorted(stats.pii_by_pattern.items())
    )
    lines.extend(
        [
            f"| **any pattern** | **{stats.pii_dropped:,}** |",
            "",
            "## Per group",
            "",
            "| Group | Train pool | Test pool | `wam` half | `frt` half | Sample | Sample safety "
            "| Natural `wam` safety |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in groups:
        natural = stats.natural_safety.get(group.id, 0)
        wam_pool = stats.wam_pool.get(group.id, 0)
        lines.append(
            f"| `{group.id}` | {stats.train_pool.get(group.id, 0):,} "
            f"| {stats.test_pool.get(group.id, 0):,} | {wam_pool:,} "
            f"| {stats.frt_pool.get(group.id, 0):,} "
            f"| {stats.sample_items.get(group.id, 0):,} "
            f"| {stats.sample_safety.get(group.id, 0):,} "
            f"| {_rate(natural, wam_pool)} |"
        )
    lines.extend(
        [
            f"| **Total** | **{stats.train_total:,}** | **{stats.test_total:,}** "
            f"| **{stats.wam_total:,}** | **{stats.frt_total:,}** "
            f"| **{stats.sample_total:,}** | **{stats.sample_safety_total:,}** "
            f"| **{_rate(stats.natural_safety_total, stats.wam_total)}** |",
            "",
            "## Safety label",
            "",
            "`safety_relevant = crash == 'Y' or fire == 'Y' or injured > 0 or deaths > 0`.",
            "",
            f"- Natural rate in the `wam` test pool: "
            f"{stats.natural_safety_total:,} / {stats.wam_total:,} = "
            f"{_rate(stats.natural_safety_total, stats.wam_total)}.",
            f"- Sampled rate: {stats.sample_safety_total:,} / {stats.sample_total:,} = "
            f"{_rate(stats.sample_safety_total, stats.sample_total)}. The sample is",
            f"  safety-oversampled to the spec's floor of {stats.sample_safety_total:,}",
            "  safety-relevant items, so the sampled rate is **not** the natural rate and",
            "  must always be reported beside it.",
            "",
            "## Method notes",
            "",
            "- The single-group rule counts only vehicle rows: an unmapped vehicle row, or",
            "  two different mapped groups, excludes the complaint whole.",
            "- The narrative is kept exactly as published apart from trimming: no case",
            "  change, no redaction in place.",
            '- Test halves use `int(sha256("20261031:" + ODINO), 16) % 2`; the train window',
            "  is shared, so only test items are halved.",
            f"- The sample is drawn with seed {stats.seed} by the spec's order key",
            "  (the full hash of the same string) and the largest-remainder allocation.",
            "- The committed files carry the eight model-facing fields the task brief names",
            "  (`id`, `text`, `make`, `model`, `year`, `received`, `component_group`,",
            "  `safety_relevant`). Identity and location columns, and the audit-only",
            "  columns, are not written anywhere.",
            "",
            "## Files",
            "",
            "| File | Items | sha256 |",
            "|---|---:|---|",
        ]
    )
    for name, count in (
        (TEST_PATH.name, stats.sample_total),
        (TRAIN_PATH.name, stats.train_item_total),
    ):
        path = out_dir / name
        if path.is_file():
            lines.append(f"| `{name}` | {count:,} | `{file_sha256(path)}` |")
    lines.extend(
        [
            "",
            "`data/defect/CHECKSUMS.txt` records the sha256 and size of every derived file",
            "and of the two copied spec files.",
            "",
        ]
    )
    return "\n".join(lines)


def build_dataset(
    *,
    spec: NhtsaSpec,
    groups: Sequence[ComponentGroup],
    source_dir: Path,
    out_dir: Path = DEFECT_DIR,
    force: bool = False,
) -> BuildStats:
    """Read the raw files and write the committed extracts, statistics and checksums."""
    targets = (out_dir / TEST_PATH.name, out_dir / TRAIN_PATH.name)
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        msg = (
            f"{names} already exist; the committed extracts are the source of truth. "
            "Pass --rebuild to replace them."
        )
        raise NhtsaDataError(msg)

    scan = scan_complaints(
        read_complaints(source_dir, spec), groups, product_type=spec.product_type
    )
    kept, filters = filter_complaints(scan.complaints, spec)

    train_pool: list[DefectItem] = []
    test_pool: list[DefectItem] = []
    outside = 0
    for complaint in kept:
        period = split_period(complaint.received, spec)
        if period == "train":
            train_pool.append(_item(complaint))
        elif period == "test":
            test_pool.append(_item(complaint))
        else:
            outside += 1

    wam = [
        item
        for item in test_pool
        if halving_half(item.id, spec.halving_prefix, half_0=spec.half_0, half_1=spec.half_1)
        == spec.half_0
    ]
    frt = [
        item
        for item in test_pool
        if halving_half(item.id, spec.halving_prefix, half_0=spec.half_0, half_1=spec.half_1)
        == spec.half_1
    ]
    if {item.id for item in wam} & {item.id for item in frt}:
        msg = "the two test halves share an item; the halving rule is broken"
        raise NhtsaDataError(msg)

    sample = draw_sample(
        wam,
        n=spec.wam_n,
        min_per_group=spec.wam_min_per_group,
        min_safety=spec.wam_min_safety,
        seed=spec.seed,
    )
    train = draw_stratified(train_pool, n=spec.train_max_items, seed=spec.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / TEST_PATH.name, sample)
    _write_jsonl_gz(out_dir / TRAIN_PATH.name, train)
    copied = [out_dir / SPEC_PATH.name, out_dir / GROUPS_PATH.name]
    _write_checksums(
        out_dir / CHECKSUMS_PATH.name,
        [path for path in copied if path.is_file()]
        + [out_dir / TEST_PATH.name, out_dir / TRAIN_PATH.name],
    )

    stats = BuildStats(
        snapshot=spec.snapshot,
        rows=scan.rows,
        vehicle_rows=scan.vehicle_rows,
        vehicle_complaints=scan.vehicle_complaints,
        single_group=filters.single_group,
        excluded=scan.excluded,
        too_short=filters.too_short,
        pii_by_pattern=filters.pii_by_pattern,
        pii_dropped=filters.pii_dropped,
        kept=filters.kept,
        outside_window=outside,
        train_pool=_group_counts(train_pool),
        test_pool=_group_counts(test_pool),
        wam_pool=_group_counts(wam),
        frt_pool=_group_counts(frt),
        natural_safety=_group_safety(wam),
        sample_items=_group_counts(sample),
        sample_safety=_group_safety(sample),
        train_items=_group_counts(train),
        seed=spec.seed,
        sample_hash=file_sha256(out_dir / TEST_PATH.name),
    )
    (out_dir / STATS_PATH.name).write_text(
        _stats_markdown(stats, groups, out_dir), encoding="utf-8"
    )
    return stats


def _group_counts(items: Iterable[DefectItem]) -> dict[str, int]:
    return dict(sorted(Counter(item.component_group for item in items).items()))


def _group_safety(items: Iterable[DefectItem]) -> dict[str, int]:
    return dict(
        sorted(Counter(item.component_group for item in items if item.safety_relevant).items())
    )


# ----------------------------------------------------------------------------- CLI


def _summary(stats: BuildStats) -> str:
    return (
        f"nhtsa: {stats.rows:,} rows, {stats.vehicle_complaints:,} vehicle complaints, "
        f"{stats.single_group:,} single-group, {stats.kept:,} after the narrative filters\n"
        f"nhtsa: train {stats.train_item_total:,}, test pool {stats.test_total:,}, "
        f"sample {stats.sample_total:,} "
        f"({stats.sample_safety_total:,} safety-relevant, natural "
        f"{_rate(stats.natural_safety_total, stats.wam_total)})\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m which_ai_model.bench.nhtsa build --source <dir> [--rebuild]``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="read the raw files and write data/defect/")
    build.add_argument("--source", type=Path, required=True, help="folder with the raw files")
    build.add_argument("--out", type=Path, default=DEFECT_DIR)
    build.add_argument("--spec", type=Path, default=SPEC_PATH)
    build.add_argument("--groups", type=Path, default=GROUPS_PATH)
    build.add_argument(
        "--rebuild",
        action="store_true",
        help="replace the committed extracts (otherwise an existing one is an error)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        stats = build_dataset(
            spec=load_spec(args.spec),
            groups=load_groups(args.groups),
            source_dir=args.source,
            out_dir=args.out,
            force=args.rebuild,
        )
    except NhtsaDataError as exc:
        sys.stderr.write(f"nhtsa: {exc}\n")
        return 1
    sys.stdout.write(_summary(stats))
    sys.stdout.write(f"nhtsa: wrote {args.out / TEST_PATH.name} and {args.out / STATS_PATH.name}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI
    raise SystemExit(main())
