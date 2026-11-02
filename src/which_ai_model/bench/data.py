"""Dataset download, checksum verification and loading (``PLAN.md`` §6).

Only two datasets are used, both from the Hugging Face dataset cards named in the plan:

* **Banking77** (PolyAI, CC-BY-4.0) — routing and structured replies.
* **CLINC150** ``plus`` configuration (CC-BY-3.0) — out-of-scope detection.

``fetch_data`` is the one networked function in this lane. It never runs in the test
suite: ``make test`` blocks sockets and tests use synthetic frames. The files land in
``data/raw/`` (gitignored); ``data/CHECKSUMS.txt`` is committed so a later run refuses a
changed file (``PLAN.md`` §10.1). Only ids are written to ``data/samples/`` — never data
text.

Rows get stable ids from their position in the downloaded parquet file, for example
``b77-test-12`` or ``clinc-plus-test-40``. The committed sample id lists therefore point
at exactly one row, and a checksum change is caught before it can silently move them.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

__all__ = [
    "BANKING77_LABELS",
    "CHECKSUMS_PATH",
    "CLINC150_LABELS",
    "DATASET_FILES",
    "RAW_DIR",
    "ChecksumMismatchError",
    "ChecksumRecord",
    "DatasetFile",
    "fetch_data",
    "load_banking77",
    "load_clinc150",
    "read_checksums",
    "sha256_file",
    "verify_checksums",
    "write_checksums",
]

RAW_DIR = Path("data/raw")
CHECKSUMS_PATH = Path("data/CHECKSUMS.txt")
REPORTS_DIR = Path("reports")

CHECKSUMS_HEADER = "# sha256\tbytes\turl\tretrieved\tfile"
"""Committed checksum file: one tab-separated row per dataset file."""

# Intent names transcribed from the Hugging Face dataset cards on 2026-10-28. The card
# order is the class index order used by the parquet files, so a shifted dataset is a
# loud error instead of a wrong label.
BANKING77_LABELS: tuple[str, ...] = (
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
)

CLINC150_LABELS: tuple[str, ...] = (
    "restaurant_reviews",
    "nutrition_info",
    "account_blocked",
    "oil_change_how",
    "time",
    "weather",
    "redeem_rewards",
    "interest_rate",
    "gas_type",
    "accept_reservations",
    "smart_home",
    "user_name",
    "report_lost_card",
    "repeat",
    "whisper_mode",
    "what_are_your_hobbies",
    "order",
    "jump_start",
    "schedule_meeting",
    "meeting_schedule",
    "freeze_account",
    "what_song",
    "meaning_of_life",
    "restaurant_reservation",
    "traffic",
    "make_call",
    "text",
    "bill_balance",
    "improve_credit_score",
    "change_language",
    "no",
    "measurement_conversion",
    "timer",
    "flip_coin",
    "do_you_have_pets",
    "balance",
    "tell_joke",
    "last_maintenance",
    "exchange_rate",
    "uber",
    "car_rental",
    "credit_limit",
    "oos",
    "shopping_list",
    "expiration_date",
    "routing",
    "meal_suggestion",
    "tire_change",
    "todo_list",
    "card_declined",
    "rewards_balance",
    "change_accent",
    "vaccines",
    "reminder_update",
    "food_last",
    "change_ai_name",
    "bill_due",
    "who_do_you_work_for",
    "share_location",
    "international_visa",
    "calendar",
    "translate",
    "carry_on",
    "book_flight",
    "insurance_change",
    "todo_list_update",
    "timezone",
    "cancel_reservation",
    "transactions",
    "credit_score",
    "report_fraud",
    "spending_history",
    "directions",
    "spelling",
    "insurance",
    "what_is_your_name",
    "reminder",
    "where_are_you_from",
    "distance",
    "payday",
    "flight_status",
    "find_phone",
    "greeting",
    "alarm",
    "order_status",
    "confirm_reservation",
    "cook_time",
    "damaged_card",
    "reset_settings",
    "pin_change",
    "replacement_card_duration",
    "new_card",
    "roll_dice",
    "income",
    "taxes",
    "date",
    "who_made_you",
    "pto_request",
    "tire_pressure",
    "how_old_are_you",
    "rollover_401k",
    "pto_request_status",
    "how_busy",
    "application_status",
    "recipe",
    "calendar_update",
    "play_music",
    "yes",
    "direct_deposit",
    "credit_limit_change",
    "gas",
    "pay_bill",
    "ingredients_list",
    "lost_luggage",
    "goodbye",
    "what_can_i_ask_you",
    "book_hotel",
    "are_you_a_bot",
    "next_song",
    "change_speed",
    "plug_type",
    "maybe",
    "w2",
    "oil_change_when",
    "thank_you",
    "shopping_list_update",
    "pto_balance",
    "order_checks",
    "travel_alert",
    "fun_fact",
    "sync_device",
    "schedule_maintenance",
    "apr",
    "transfer",
    "ingredient_substitution",
    "calories",
    "current_location",
    "international_fees",
    "calculator",
    "definition",
    "next_holiday",
    "update_playlist",
    "mpg",
    "min_payment",
    "change_user_name",
    "restaurant_suggestion",
    "travel_notification",
    "cancel",
    "pto_used",
    "travel_suggestion",
    "change_volume",
)

CLINC_OOS_LABEL = "oos"


@dataclass(frozen=True)
class DatasetFile:
    """One parquet file: local name and the exact URL from ``PLAN.md`` §6."""

    name: str
    url: str


DATASET_FILES: tuple[DatasetFile, ...] = (
    DatasetFile(
        "banking77_train.parquet",
        "https://huggingface.co/api/datasets/PolyAI/banking77/parquet/default/train/0.parquet",
    ),
    DatasetFile(
        "banking77_test.parquet",
        "https://huggingface.co/api/datasets/PolyAI/banking77/parquet/default/test/0.parquet",
    ),
    DatasetFile(
        "clinc150_plus_train.parquet",
        "https://huggingface.co/api/datasets/clinc/clinc_oos/parquet/plus/train/0.parquet",
    ),
    DatasetFile(
        "clinc150_plus_test.parquet",
        "https://huggingface.co/api/datasets/clinc/clinc_oos/parquet/plus/test/0.parquet",
    ),
)


class ChecksumMismatchError(RuntimeError):
    """A downloaded or existing file no longer matches ``data/CHECKSUMS.txt`` (§10.1)."""


@dataclass(frozen=True)
class ChecksumRecord:
    """One row of ``data/CHECKSUMS.txt``."""

    name: str
    sha256: str
    size: int
    url: str
    retrieved: str

    def format_line(self) -> str:
        return f"{self.sha256}\t{self.size}\t{self.url}\t{self.retrieved}\t{self.name}"


Downloader = Callable[[str, Path], None]


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums(path: Path = CHECKSUMS_PATH) -> dict[str, ChecksumRecord]:
    """Parse the committed checksum file, keyed by local file name."""
    records: dict[str, ChecksumRecord] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 5:
            msg = f"{path}:{lineno}: expected 5 tab-separated fields, got {len(fields)}"
            raise ValueError(msg)
        sha, size_text, url, retrieved, name = fields
        if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
            msg = f"{path}:{lineno}: not a sha256 hex digest"
            raise ValueError(msg)
        records[name] = ChecksumRecord(
            name=name,
            sha256=sha,
            size=int(size_text),
            url=url,
            retrieved=retrieved,
        )
    return records


def write_checksums(path: Path, records: list[ChecksumRecord]) -> None:
    """Write the checksum file in a fixed order, with a header comment."""
    lines = [CHECKSUMS_HEADER]
    lines.extend(record.format_line() for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_checksums(
    checksums: Path = CHECKSUMS_PATH,
    *,
    dest: Path = RAW_DIR,
) -> None:
    """Check every recorded file in ``dest``; raise ``ChecksumMismatchError`` on a change."""
    if not checksums.exists():
        msg = f"{checksums} does not exist; run `make fetch-data` first"
        raise FileNotFoundError(msg)
    records = read_checksums(checksums)
    missing = [name for name in records if not (dest / name).exists()]
    if missing:
        msg = f"missing dataset file(s) in {dest}: {', '.join(sorted(missing))}"
        raise ChecksumMismatchError(msg)
    for name, record in records.items():
        path = dest / name
        size = path.stat().st_size
        actual = sha256_file(path)
        if actual != record.sha256 or size != record.size:
            msg = (
                f"checksum mismatch for {name}: expected {record.sha256} "
                f"({record.size} bytes), got {actual} ({size} bytes)"
            )
            raise ChecksumMismatchError(msg)


def _download(url: str, dest: Path) -> None:
    """Stream one URL to ``dest`` (used only by ``fetch_data``)."""
    with (
        urllib.request.urlopen(url, timeout=120) as response,
        dest.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle)


def _write_stop(reports_dir: Path, detail: str) -> None:
    """Append a stop-rule entry to ``reports/STOP.md`` (``PLAN.md`` §10.1)."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "STOP.md"
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ")
    lines = [
        f"# STOP - dataset checksum mismatch ({stamp})",
        "",
        "`PLAN.md` §10.1: a dataset checksum differs from `data/CHECKSUMS.txt`.",
        "No dataset text was committed and no sample was rebuilt.",
        "",
        f"Detail: {detail}",
        "",
    ]
    prefix = "\n" if path.exists() else ""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prefix + "\n".join(lines))


def _check_existing(path: Path, record: ChecksumRecord, reports_dir: Path) -> None:
    size = path.stat().st_size
    actual = sha256_file(path)
    if actual != record.sha256 or size != record.size:
        detail = (
            f"{path} expected {record.sha256} ({record.size} bytes), got {actual} ({size} bytes)"
        )
        _write_stop(reports_dir, detail)
        raise ChecksumMismatchError(detail)


def fetch_data(
    dest: Path = RAW_DIR,
    *,
    checksums: Path = CHECKSUMS_PATH,
    reports_dir: Path = REPORTS_DIR,
    downloader: Downloader = _download,
    retrieved: str | None = None,
) -> list[ChecksumRecord]:
    """Download the four §6 parquet files and record their checksums.

    On the first run this writes ``checksums``. On later runs every existing or newly
    downloaded file must match it; a difference writes ``reports/STOP.md`` and raises
    ``ChecksumMismatchError`` without replacing the known-good file.
    """
    dest.mkdir(parents=True, exist_ok=True)
    known = read_checksums(checksums) if checksums.exists() else {}
    retrieved_date = retrieved or datetime.now(UTC).strftime("%Y-%m-%d")
    records: list[ChecksumRecord] = []

    for spec in DATASET_FILES:
        target = dest / spec.name
        expected = known.get(spec.name)

        if expected is not None and target.exists():
            _check_existing(target, expected, reports_dir)
            records.append(expected)
            continue

        with tempfile.NamedTemporaryFile(
            dir=dest, prefix=f".{spec.name}.", suffix=".part", delete=False
        ) as handle:
            partial = Path(handle.name)
        size = 0
        actual = ""
        try:
            downloader(spec.url, partial)
            size = partial.stat().st_size
            actual = sha256_file(partial)
            if expected is not None and (actual != expected.sha256 or size != expected.size):
                detail = (
                    f"{spec.url} expected {expected.sha256} ({expected.size} bytes), "
                    f"downloaded {actual} ({size} bytes)"
                )
                _write_stop(reports_dir, detail)
                raise ChecksumMismatchError(detail)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)

        records.append(
            expected
            if expected is not None
            else ChecksumRecord(
                name=spec.name,
                sha256=actual,
                size=size,
                url=spec.url,
                retrieved=retrieved_date,
            )
        )

    if {record.name: record for record in records} != known or not checksums.exists():
        write_checksums(checksums, records)
    return records


def load_banking77(
    split: str = "test",
    *,
    root: Path = RAW_DIR,
) -> pd.DataFrame:
    """Load Banking77 with stable ids and intent names from the dataset card.

    Columns: ``item_id`` (``b77-<split>-<row>``), ``text``, ``intent``.
    """
    if split not in {"train", "test"}:
        msg = f"Banking77 split must be 'train' or 'test', got {split!r}"
        raise ValueError(msg)
    path = root / f"banking77_{split}.parquet"
    frame = pd.read_parquet(path)
    labels = _map_labels(frame, "label", BANKING77_LABELS, path)
    return pd.DataFrame(
        {
            "item_id": [f"b77-{split}-{row}" for row in range(len(frame))],
            "text": frame["text"].astype("string").tolist(),
            "intent": labels,
        }
    )


def load_clinc150(
    split: str = "test",
    *,
    root: Path = RAW_DIR,
) -> pd.DataFrame:
    """Load CLINC150 ``plus`` with stable ids and intent names.

    Columns: ``item_id`` (``clinc-plus-<split>-<row>``), ``text``, ``intent``; the
    out-of-scope class keeps the dataset card's name ``oos``. ``split="plus"`` is
    accepted as an alias for ``test`` because the config is called ``plus``.
    """
    if split == "plus":
        split = "test"
    if split not in {"train", "test"}:
        msg = f"CLINC150 split must be 'train' or 'test', got {split!r}"
        raise ValueError(msg)
    path = root / f"clinc150_plus_{split}.parquet"
    frame = pd.read_parquet(path)
    labels = _map_labels(frame, "intent", CLINC150_LABELS, path)
    return pd.DataFrame(
        {
            "item_id": [f"clinc-plus-{split}-{row}" for row in range(len(frame))],
            "text": frame["text"].astype("string").tolist(),
            "intent": labels,
        }
    )


def _map_labels(
    frame: pd.DataFrame,
    column: str,
    names: tuple[str, ...],
    path: Path,
) -> list[str]:
    if column not in frame.columns:
        msg = f"{path}: expected a {column!r} column, found {list(frame.columns)}"
        raise ValueError(msg)
    values = [int(value) for value in frame[column].tolist()]
    unknown = sorted({value for value in values if value < 0 or value >= len(names)})
    if unknown:
        msg = f"{path}: label index {unknown} outside the dataset card's {len(names)} names"
        raise ValueError(msg)
    return [names[value] for value in values]
