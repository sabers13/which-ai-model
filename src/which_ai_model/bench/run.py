"""Benchmark runner: replay or record every model x task x item (``PLAN.md`` §7-§8).

The runner owns the orchestration only. A task predictor is any object implementing the
:class:`Predictor` protocol: it builds the messages and parses the answer. T07a/T09 wire
their real predictors through the registry in T10; tests use fake predictors and a fake
model invoker, so this module never imports a tool and never opens a socket in replay.

Guarantees:

* **One prompt per task for every model.** Prompts come from
  :mod:`which_ai_model.bench.prompts`; a predictor may add task context (T-DOCS passes
  retrieved passages), but never a model-specific tweak.
* **Resumable.** A recording is looked up by
  ``sha256(model_id + prompt_version + request_digest)`` before a call. In ``record``
  mode an existing key is served without a new call; ``replay`` never calls a model.
* **Paced and budgeted.** Calls take a per-model concurrency/requests-per-minute slot
  from ``config/budget.toml`` and DeepSeek V4.1 Flash waits out its peak windows. There
  is **no hard USD cap** (captain deviation, ``reports/DEVIATIONS.md``); spend is
  recorded per model and reported. Per-model 5-hour window ceilings still apply: the
  runner reduces T-ROUTE to 5 per intent first, logs that in ``reports/DEVIATIONS.md``,
  waits for the next window when it can, and writes ``reports/STOP.md`` only when the
  cell cannot fit any window.
* **Stop rule §10.7.** A 403, an abuse/policy/suspension body or an account warning
  stops every live call, writes ``reports/STOP.md`` and is never retried.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from datetime import time as wall_time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ValidationError

from which_ai_model.bench.data import CLINC_OOS_LABEL, load_banking77, load_clinc150
from which_ai_model.bench.prompts import PROMPT_VERSION, prompt_for
from which_ai_model.bench.sample import load_sample_ids, sample_hash
from which_ai_model.bench.score import (
    SCORER_VERSION,
    SpecCellResult,
    baseline_candidates,
    build_recommendations_with_specs,
    evaluate_cell,
    evaluate_spec_cell,
    write_cells,
    write_recommendations,
)
from which_ai_model.contracts import (
    SPEC_ID_PATTERN,
    CallRecord,
    CellResult,
    DocsOut,
    ModelRef,
    Prediction,
    RunManifest,
    RunSettings,
    TaskName,
    TaskRef,
    TaskSpec,
    builtin_task,
    task_key,
)
from which_ai_model.core import (
    RateLimiter,
    Recorder,
    RecordingMode,
    ReplayMiss,
    cost_usd,
    extract_usage,
    load_prices,
    recording_key,
    request_digest,
    response_from_message,
)
from which_ai_model.core.ratelimit import Limits, load_limits
from which_ai_model.core.record import (
    DEFAULT_FRESH_RECORDINGS_ROOT,
    fresh_recording_root,
    model_slug,
)

__all__ = [
    "DEFAULT_FRESH_RECORDINGS_ROOT",
    "DEFAULT_RUNS_DIR",
    "GOOD_ENOUGH_POINTS",
    "SCORER_VERSION",
    "BenchStopError",
    "Budget",
    "BudgetConfig",
    "Clock",
    "DocsPredictor",
    "ModelAnswer",
    "ModelSpec",
    "PolicyStopError",
    "Predictor",
    "PromptPredictor",
    "ReplayIncompleteError",
    "SpecCallRecord",
    "default_registry",
    "estimate_cell_cost",
    "estimate_run_cost",
    "estimate_spec_cell_cost",
    "load_budget_config",
    "load_items",
    "load_models",
    "load_spec_items",
    "load_specs",
    "manifest_settings",
    "rag_registry",
    "reduce_route_items",
    "run",
    "run_settings_for",
    "spec_predictor",
    "spec_registry",
    "write_manifest_settings",
]

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime import cycle
    from which_ai_model.task import SpecPredictor

DEFAULT_RUNS_DIR = Path("results/runs")
DEFAULT_RECORDINGS_DIR = Path("recordings")
DEFAULT_SAMPLES_DIR = Path("data/samples")
DEFAULT_DATA_ROOT = Path("data/raw")
DEFAULT_DOCS_GOLD = Path("data/rag/gold.jsonl")
GOOD_ENOUGH_POINTS = 3.0

type Message = dict[str, str]

_TASK_SLUGS: dict[TaskName, str] = {
    TaskName.ROUTE: "route",
    TaskName.SCOPE: "scope",
    TaskName.REPLY: "reply",
    TaskName.DOCS: "docs",
}
_TASK_OUTPUT_TOKENS: dict[TaskName, int] = {
    TaskName.ROUTE: 16,
    TaskName.SCOPE: 8,
    TaskName.REPLY: 256,
    TaskName.DOCS: 512,
}
#: A spec prompt asks for one small JSON object; §A4 fixes at least 1,024 output tokens.
SPEC_OUTPUT_TOKENS = 1024
_SECRET_TEXT = re.compile(
    r"(sk-[A-Za-z0-9_-]{8,}|Bearer\s+\S+|AIza[0-9A-Za-z_-]{10,}|lsv2_[A-Za-z0-9_-]+)"
)
_POLICY_WORDS = (
    "abuse",
    "policy",
    "suspension",
    "suspended",
    "account warning",
    "violat",
)


class BenchStopError(RuntimeError):
    """The runner cannot proceed; ``reports/STOP.md`` says why."""


class PolicyStopError(BenchStopError):
    """A 403/abuse/policy response. Never retried (``PLAN.md`` §10.7)."""


class ReplayIncompleteError(RuntimeError):
    """A replay run has no recording for a key. It must never fall back to a call."""


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class ModelSpec:
    """One model in ``config/models.lock.toml``."""

    ref: ModelRef
    reasoning: str = "minimal"
    max_tokens: int = 1024
    #: ``None`` means the lock says nothing and the run's temperature applies. A number
    #: is sent as-is; ``"provider-default"`` sends no parameter at all (the API rejects
    #: it for that model, ``PLAN.md`` §7 "temperature 0 where the API allows").
    temperature: float | Literal["provider-default"] | None = None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return default
    return value


def _as_temperature(value: Any) -> float | Literal["provider-default"] | None:
    """The lock's optional per-model temperature (a number or ``provider-default``)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and 0.0 <= float(value) <= 2.0:
        return float(value)
    if isinstance(value, str) and value.strip() == "provider-default":
        return "provider-default"
    return None


def load_models(
    path: Path,
    *,
    model_ids: Sequence[str] | None = None,
) -> list[ModelSpec]:
    """Read the T10 lock file. Accepts ``[[model]]`` (and ``[[models]]`` as a fallback)."""
    if not path.is_file():
        msg = f"model lock file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    rows = data.get("model", data.get("models", []))
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        msg = f"{path}: [[model]] must be an array of tables"
        raise ValueError(msg)
    wanted = set(model_ids) if model_ids else None
    specs: list[ModelSpec] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        table = _as_mapping(row)
        connection_id = _as_str(table.get("connection_id"))
        model_id = _as_str(table.get("model_id"))
        if not connection_id or not model_id:
            continue
        if wanted is not None and model_id not in wanted:
            continue
        key = (connection_id, model_id)
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            ModelSpec(
                ref=ModelRef(connection_id=connection_id, model_id=model_id),
                reasoning=_as_str(table.get("reasoning"), "minimal"),
                max_tokens=_as_positive_int(table.get("max_tokens"), 1024),
                temperature=_as_temperature(table.get("temperature")),
            )
        )
    if not specs:
        msg = f"{path}: no models selected"
        raise ValueError(msg)
    return specs


# ----------------------------------------------------------------------- run settings


def _recorded_reasoning(reasoning: str) -> str:
    """The lock's reasoning setting as recorded; empty means no parameter is sent."""
    return reasoning.strip() or "not-sent"


def run_settings_for(spec: ModelSpec, *, temperature: float = 0.0) -> RunSettings:
    """The sampling settings a run records for one model (``PLAN-v1.1`` §A5).

    A lock entry that says ``temperature = "provider-default"`` sends no parameter and
    is recorded that way, so the gate can refuse a run whose sampling settings differ.
    """
    recorded = temperature if spec.temperature is None else spec.temperature
    return RunSettings(
        temperature=recorded,
        reasoning=_recorded_reasoning(spec.reasoning),
        max_output_tokens=spec.max_tokens,
    )


def manifest_settings(settings: Mapping[str, RunSettings]) -> dict[str, dict[str, object]]:
    """The JSON body of a manifest's ``settings`` block, keyed by model id."""
    return {
        model_id: dict(settings[model_id].model_dump(mode="json")) for model_id in sorted(settings)
    }


def write_manifest_settings(
    path: Path,
    settings: Mapping[str, RunSettings],
    *,
    lock_sha256: str | None = None,
) -> None:
    """Add ``settings`` and ``lock_sha256`` to an existing manifest.

    Every other key keeps its value, so a v1 manifest is only ever read, and a new
    manifest still carries the exact v1 fields the gate already compares.
    """
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        msg = f"{path}: manifest must be a JSON object"
        raise ValueError(msg)
    document["settings"] = manifest_settings(settings)
    if lock_sha256 is not None:
        document["lock_sha256"] = lock_sha256
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _lock_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------- budget config


@dataclass(frozen=True)
class PeakWindow:
    """A UTC peak window; DeepSeek is only called outside these on weekdays."""

    start: wall_time
    end: wall_time


@dataclass(frozen=True)
class BudgetConfig:
    """The parts of ``config/budget.toml`` the runner needs."""

    cap_usd: float | None = None
    window_hours: float = 5.0
    window_ceilings: Mapping[str, float] = field(default_factory=dict)
    limits: Mapping[str, Limits] = field(default_factory=dict)
    peak_windows: Mapping[str, tuple[PeakWindow, ...]] = field(default_factory=dict)


def _opt_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _parse_peak_windows(rows: Any) -> tuple[PeakWindow, ...]:
    windows: list[PeakWindow] = []
    if not isinstance(rows, list):
        return ()
    for row in rows:
        if not isinstance(row, str) or "-" not in row:
            continue
        start_text, end_text = row.split("-", 1)
        try:
            start = wall_time.fromisoformat(start_text.strip())
            end = wall_time.fromisoformat(end_text.strip())
        except ValueError:
            continue
        windows.append(PeakWindow(start=start, end=end))
    return tuple(windows)


def load_budget_config(path: Path) -> BudgetConfig:
    """Read the budget file. Missing file = no cap, 5-hour windows, default pacing."""
    if not path.is_file():
        return BudgetConfig(
            limits={"opencode-go": Limits(max_concurrency=2, requests_per_minute=30)}
        )
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    budget_table = _as_mapping(data.get("budget"))
    window_table = _as_mapping(data.get("window"))
    cap = _opt_float(budget_table.get("cap_usd", data.get("cap_usd")))
    window_hours = _opt_float(budget_table.get("window_hours", window_table.get("hours"))) or 5.0
    ceilings_raw = _as_mapping(window_table.get("ceilings", data.get("window_ceilings")))
    ceilings = {
        str(model_id): float(value)
        for model_id, value in ceilings_raw.items()
        if _opt_float(value) is not None
    }
    schedule_raw = _as_mapping(data.get("schedule"))
    peaks = {
        str(model_id): _parse_peak_windows(_as_mapping(row).get("peak_windows_utc"))
        for model_id, row in schedule_raw.items()
    }
    return BudgetConfig(
        cap_usd=cap,
        window_hours=window_hours,
        window_ceilings=ceilings,
        limits=load_limits(path),
        peak_windows={model_id: windows for model_id, windows in peaks.items() if windows},
    )


# ---------------------------------------------------------------------------- clock


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Clock:
    """Injectable time for tests: monotonic for limits, wall clock for scheduling."""

    monotonic: Callable[[], float] = time.monotonic
    now: Callable[[], datetime] = _utcnow
    sleep: Callable[[float], None] = time.sleep


def _aware(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


# --------------------------------------------------------------------------- budget


class Budget:
    """Tracked spend with an optional hard cap and optional per-model window ceilings.

    The committed configuration has **no** ``cap_usd``: the USD 8 cap was removed by
    captain order. Passing ``cap_usd=None`` therefore means "track and report, never
    block on the total". A ceiling is a pacing guard, not a cap: when it is the blocker
    the runner waits for the next window; a cell that cannot fit even a fresh window is
    a :class:`BenchStopError`.
    """

    def __init__(
        self,
        cap_usd: float | None = None,
        *,
        ceilings: Mapping[str, float] | None = None,
        window_s: float = 5 * 3600,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cap_usd = cap_usd
        self._ceilings = dict(ceilings or {})
        self._window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._spent_total = 0.0
        self._spent_model: dict[str, float] = {}
        self._window_start: dict[str, float] = {}
        self._window_spent: dict[str, float] = {}
        self._unknown_calls = 0

    @property
    def unknown_calls(self) -> int:
        return self._unknown_calls

    def ceiling(self, model_id: str | None = None) -> float | None:
        return self._ceilings.get(model_id) if model_id is not None else None

    def total_spent(self) -> float:
        with self._lock:
            return self._spent_total

    def spent_by_model(self) -> dict[str, float]:
        with self._lock:
            return dict(self._spent_model)

    def _reset_if_due(self, model_id: str) -> None:
        start = self._window_start.get(model_id)
        if start is not None and self._clock() - start >= self._window_s:
            self._window_start[model_id] = self._clock()
            self._window_spent[model_id] = 0.0

    def window_spent(self, model_id: str) -> float:
        with self._lock:
            self._reset_if_due(model_id)
            return self._window_spent.get(model_id, 0.0)

    def remaining_total(self) -> float | None:
        if self.cap_usd is None:
            return None
        return self.cap_usd - self.total_spent()

    def window_remaining(self, model_id: str) -> float | None:
        ceiling = self._ceilings.get(model_id)
        if ceiling is None:
            return None
        return ceiling - self.window_spent(model_id)

    def can_spend(self, estimate_usd: float | None, *, model_id: str | None = None) -> bool:
        if estimate_usd is None:
            return True
        with self._lock:
            if self.cap_usd is not None and self._spent_total + estimate_usd > self.cap_usd + 1e-9:
                return False
            if model_id is not None and model_id in self._ceilings:
                self._reset_if_due(model_id)
                ceiling = self._ceilings[model_id]
                spent = self._window_spent.get(model_id, 0.0)
                return spent + estimate_usd <= ceiling + 1e-9
        return True

    def spend(self, usd: float | None, *, model_id: str | None = None) -> None:
        with self._lock:
            if usd is None:
                self._unknown_calls += 1
                return
            self._spent_total += usd
            if model_id is not None:
                self._spent_model[model_id] = self._spent_model.get(model_id, 0.0) + usd
                self._reset_if_due(model_id)
                if model_id not in self._window_start:
                    self._window_start[model_id] = self._clock()
                self._window_spent[model_id] = self._window_spent.get(model_id, 0.0) + usd

    def wait_for_window(self, model_id: str) -> float:
        """Sleep until the model's window resets; return the seconds waited."""
        with self._lock:
            start = self._window_start.get(model_id)
            elapsed = self._clock() - start if start is not None else self._window_s
            wait = max(0.0, self._window_s - elapsed)
            if start is None:
                wait = 0.0
        self._sleep(wait)
        with self._lock:
            self._window_start[model_id] = self._clock()
            self._window_spent[model_id] = 0.0
        return wait


# ------------------------------------------------------------------------ predictors


@dataclass(frozen=True)
class ModelAnswer:
    """One recorded (or freshly made) model response plus its recording key."""

    key: str
    response: dict[str, Any]
    latency_ms: int
    error: str | None = None


type Call = Callable[[Sequence[Message]], ModelAnswer]


class Predictor(Protocol):
    """What the runner needs from a task predictor (T07a/T09 implement this in T10)."""

    task: TaskName
    prompt_version: str

    def predict(self, item: Mapping[str, Any], model: ModelRef, call: Call) -> Prediction: ...


class SpecPredictorLike(Protocol):
    """A spec-task predictor (M1's ``SpecPredictor``) as the runner sees it.

    ``task`` is the spec id, never a ``TaskName``: the spec rows are written beside the
    v1 rows (``PLAN-v1.1`` §A2.1). The predictor builds the messages, calls the closure
    the runner hands it and parses its own schema; the runner owns everything else.
    """

    @property
    def task(self) -> str: ...
    @property
    def prompt_version(self) -> str: ...
    def predict(self, item: Mapping[str, Any], model: ModelRef, call: Call) -> Any: ...


def parse_response(
    schema: type[BaseModel], response: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Parse structured output from a recorded response.

    A tool call's ``args`` wins (LangChain structured output); a JSON object in
    ``content`` is the fallback. Anything else is schema-invalid — never guessed and
    never retried to fish for a better answer.
    """
    tool_calls = response.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            args = call.get("args")
            if isinstance(args, dict):
                try:
                    return schema.model_validate(args).model_dump(mode="json"), True
                except ValidationError:
                    continue
    content = response.get("content")
    if isinstance(content, dict):
        try:
            return schema.model_validate(content).model_dump(mode="json"), True
        except ValidationError:
            return {}, False
    if isinstance(content, str) and content.strip():
        try:
            return schema.model_validate_json(content).model_dump(mode="json"), True
        except ValidationError:
            return {}, False
    return {}, False


class PromptPredictor:
    """The default predictor: one versioned prompt per task, parsed against its schema."""

    def __init__(self, task: TaskName) -> None:
        self.task = task
        self.prompt_version = PROMPT_VERSION
        self._prompt = prompt_for(task)

    def predict(self, item: Mapping[str, Any], model: ModelRef, call: Call) -> Prediction:
        answer = call(self._prompt.messages(item))
        output, _valid = parse_response(self._prompt.schema, answer.response)
        return Prediction(
            item_id=str(item["item_id"]),
            task=self.task,
            model=model,
            output=output,
            raw_key=answer.key,
        )


def default_registry() -> dict[TaskName, Predictor]:
    """A prompt predictor per task. T10 replaces entries with the real tool predictors."""
    return {task: PromptPredictor(task) for task in TaskName}


# ---------------------------------------------------------------------- spec tasks


def spec_predictor(spec: TaskSpec, *, root: Path = Path()) -> SpecPredictor:
    """M1's generic predictor for one spec task, served to the runner."""
    from which_ai_model.task import SpecPredictor

    return SpecPredictor(spec, root=root)


def spec_registry(specs: Sequence[TaskSpec], *, root: Path = Path()) -> dict[str, SpecPredictor]:
    """Spec id -> predictor, served to the runner beside :func:`default_registry`."""
    return {spec.id: spec_predictor(spec, root=root) for spec in specs}


def load_specs(
    task_ids: Sequence[str] | None = None,
    *,
    tasks_dir: Path = Path("tasks"),
) -> list[TaskSpec]:
    """Load the committed specs, or only the selected ids, in a stable order.

    A spec id that has no ``<tasks_dir>/<id>.toml`` raises :class:`FileNotFoundError`
    with the path a person has to create, instead of silently skipping the task.
    """
    from which_ai_model.task.spec import load_spec

    if task_ids is None:
        paths = sorted(tasks_dir.glob("*.toml"))
    else:
        paths = [tasks_dir / f"{task_id}.toml" for task_id in task_ids]
    specs: list[TaskSpec] = []
    for path in paths:
        if not path.is_file():
            msg = f"task spec not found: {path} (write the spec or pass specs= explicitly)"
            raise FileNotFoundError(msg)
        specs.append(load_spec(path))
    return specs


def load_spec_items(spec: TaskSpec, *, root: Path = Path()) -> list[dict[str, Any]]:
    """Read one spec's committed JSONL data file in file order.

    Every item gains ``item_id`` from the spec's ``id_field`` so the predictor, the
    scorer and the predictions file agree on one key. The truth fields stay on the row.
    """
    path = root / spec.data.file
    if not path.is_file():
        msg = f"task data file not found: {path}"
        raise FileNotFoundError(msg)
    items: list[dict[str, Any]] = []
    for row in _read_jsonl(path):
        item_id = row.get(spec.data.id_field)
        if item_id is None or str(item_id).strip() == "":
            msg = f"{path}: every item needs a {spec.data.id_field!r} value"
            raise ValueError(msg)
        items.append({**row, "item_id": str(item_id)})
    if not items:
        msg = f"task data file has no items: {path}"
        raise ValueError(msg)
    return items


def _estimate_spec_input_tokens(
    spec: TaskSpec, items: Sequence[Mapping[str, Any]], *, root: Path
) -> int:
    from which_ai_model.task import load_labels, render_prompt

    labels = load_labels(spec, root=root)
    total = 0
    for item in items:
        text = item.get(spec.data.text_field)
        if not isinstance(text, str):
            text = ""
        total += max(1, len(render_prompt(spec, labels, text, root=root)) // 4)
    return total


def estimate_spec_cell_cost(
    spec: TaskSpec,
    items: Sequence[Mapping[str, Any]],
    prices: Mapping[str, Any],
    *,
    model_id: str,
    max_tokens: int = 1024,
    root: Path = Path(),
) -> float | None:
    """Estimate one spec cell with the same guard as the v1 cells.

    Four characters per input token from the rendered prompt, and a fixed output
    allowance capped by the model's ``max_tokens``. It is a guard, not a bill.
    """
    price = prices.get(model_id)
    if price is None or getattr(price, "currency", "USD") != "USD":
        return None
    input_tokens = _estimate_spec_input_tokens(spec, items, root=root)
    output_tokens = min(max_tokens, SPEC_OUTPUT_TOKENS) * len(items)
    return (
        input_tokens * float(price.input_per_mtok) + output_tokens * float(price.output_per_mtok)
    ) / 1_000_000


class DocsPredictor:
    """T-DOCS on the real retrieval path, wrapped around the runner's call closure.

    The runner owns record/replay, pacing and cost; this predictor owns the task context.
    It embeds the question with the T05 embedder, retrieves the four best guide passages
    from the in-memory index built from ``docs/guide`` (the same corpus and embedding
    model as the pgvector index), appends them to the one committed T-DOCS prompt, then
    parses the answer as :class:`DocsOut`. Cited ids that were not retrieved are dropped,
    exactly as T09's ``cite`` node does, and ``retrieved_ids`` is added for the scorer's
    hit@4 metric.

    The store is injectable so tests and alternate stores never touch the real embedder.
    """

    task = TaskName.DOCS
    prompt_version = PROMPT_VERSION

    def __init__(self, store: Any = None, *, k: int = 4) -> None:
        self._store = store
        self._k = k
        self._prompt = prompt_for(TaskName.DOCS)

    def _vector_store(self) -> Any:
        if self._store is None:
            from which_ai_model.tools.docs import default_store

            self._store = default_store()
        return self._store

    def _messages(self, item: Mapping[str, Any], retrieved: Sequence[Any]) -> list[Message]:
        base = self._prompt.messages(item)
        passages = "\n\n".join(f"[{hit.chunk.id}] {hit.chunk.text}" for hit in retrieved)
        user = (
            f"{base[1]['content']}\n\n"
            f"Passages (cite their ids exactly):\n{passages}\n\n"
            "Use only these passages. If they do not answer the question, abstain."
        )
        return [base[0], {"role": "user", "content": user}]

    def predict(self, item: Mapping[str, Any], model: ModelRef, call: Call) -> Prediction:
        from which_ai_model.rag.embed import embed_texts

        question = str(item.get("question", item.get("text", "")))
        vector = embed_texts([question])[0]
        retrieved = self._vector_store().search(vector, k=self._k)
        answer = call(self._messages(item, retrieved))
        output, valid = parse_response(DocsOut, answer.response)
        retrieved_ids = [str(hit.chunk.id) for hit in retrieved]
        if valid:
            cited = [str(value) for value in output.get("cited_ids", [])]
            kept = [value for value in cited if value in set(retrieved_ids)]
            output["cited_ids"] = list(dict.fromkeys(kept))
            output["retrieved_ids"] = retrieved_ids
        else:
            # Retrieval still happened; keep hit@4 measurable without making an
            # invalid answer look schema-valid (the runner checks "answer" for docs).
            output = {"retrieved_ids": retrieved_ids}
        return Prediction(
            item_id=str(item.get("item_id", item.get("id", ""))),
            task=self.task,
            model=model,
            output=output,
            raw_key=answer.key,
        )


def rag_registry(store: Any = None) -> dict[TaskName, Predictor]:
    """The registry used by the live/replay benchmark: the real T-DOCS path."""
    registry: dict[TaskName, Predictor] = {
        task: PromptPredictor(task) for task in TaskName if task is not TaskName.DOCS
    }
    registry[TaskName.DOCS] = DocsPredictor(store)
    return registry


# ---------------------------------------------------------------------------- items


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _index_frame(frame: Any) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in frame.itertuples(index=False):
        index[str(row.item_id)] = {"text": str(row.text), "intent": str(row.intent)}
    return index


def load_items(
    task: TaskName,
    *,
    samples_dir: Path = DEFAULT_SAMPLES_DIR,
    data_root: Path = DEFAULT_DATA_ROOT,
    items_dir: Path | None = None,
    docs_gold: Path = DEFAULT_DOCS_GOLD,
) -> list[dict[str, Any]]:
    """Load a task's items in sample order.

    ``items_dir`` is the tiny fixture path used by ``make bench-replay``; without it the
    committed sample ids are joined to the downloaded datasets (T10 has them).
    """
    slug = _TASK_SLUGS[task]
    if items_dir is not None:
        path = items_dir / f"{slug}.jsonl"
        rows = _read_jsonl(path)
        return [
            {**row, "item_id": str(row.get("item_id", ""))}
            for row in rows
            if str(row.get("item_id", ""))
        ]
    if task is TaskName.DOCS:
        return [
            {
                "item_id": str(row["id"]),
                "question": str(row["question"]),
                "gold_ids": [str(value) for value in row.get("gold_ids", [])],
                "answerable": bool(row.get("answerable", True)),
            }
            for row in _read_jsonl(docs_gold)
        ]

    ids = load_sample_ids(slug, samples_dir=samples_dir)
    if task is TaskName.SCOPE:
        index = _index_frame(load_clinc150("test", root=data_root))
    else:
        index = _index_frame(load_banking77("test", root=data_root))
    missing = [item_id for item_id in ids if item_id not in index]
    if missing:
        msg = f"sample id(s) missing from the dataset: {missing[:5]}"
        raise ValueError(msg)
    items: list[dict[str, Any]] = []
    for item_id in ids:
        row = dict(index[item_id])
        row["item_id"] = item_id
        if task is TaskName.SCOPE:
            row["out_of_scope"] = row["intent"] == CLINC_OOS_LABEL
        items.append(row)
    return items


def reduce_route_items(
    items: Sequence[Mapping[str, Any]], *, per_intent: int = 5
) -> list[dict[str, Any]]:
    """PLAN.md §8 reduction: keep the first ``per_intent`` items of each intent."""
    kept: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for item in items:
        intent = str(item.get("intent", ""))
        if seen.get(intent, 0) >= per_intent:
            continue
        seen[intent] = seen.get(intent, 0) + 1
        kept.append(dict(item))
    return kept


# --------------------------------------------------------------------- cost estimate


def _estimate_input_tokens(messages: Sequence[Message]) -> int:
    characters = sum(len(message.get("content", "")) for message in messages)
    return max(1, characters // 4)


def estimate_cell_cost(
    task: TaskName,
    items: Sequence[Mapping[str, Any]],
    prices: Mapping[str, Any],
    *,
    model_id: str,
    max_tokens: int = 1024,
) -> float | None:
    """Estimate one cell from prompt characters and a task output allowance.

    The estimate is deliberately simple and deterministic: four characters per input
    token and a fixed output allowance per task, capped by the model's ``max_tokens``.
    It is a guard, not a bill; the recorded spend always uses measured usage.
    """
    price = prices.get(model_id)
    if price is None or getattr(price, "currency", "USD") != "USD":
        return None
    prompt = prompt_for(task)
    input_tokens = sum(_estimate_input_tokens(prompt.messages(item)) for item in items)
    output_tokens = min(max_tokens, _TASK_OUTPUT_TOKENS[task]) * len(items)
    return (
        input_tokens * float(price.input_per_mtok) + output_tokens * float(price.output_per_mtok)
    ) / 1_000_000


def estimate_run_cost(
    models: Sequence[ModelSpec | ModelRef],
    tasks: Sequence[TaskName],
    *,
    prices: Mapping[str, Any] | Path | None = None,
    samples_dir: Path = DEFAULT_SAMPLES_DIR,
    data_root: Path = DEFAULT_DATA_ROOT,
    items_dir: Path | None = None,
    docs_gold: Path = DEFAULT_DOCS_GOLD,
) -> float | None:
    """Estimate a whole run from the same per-cell guard; ``None`` if a price is unknown."""
    if prices is None:
        table: Mapping[str, Any] = load_prices()
    elif isinstance(prices, Mapping):
        table = prices
    else:
        table = load_prices(prices)
    total = 0.0
    for model in models:
        spec = model if isinstance(model, ModelSpec) else ModelSpec(ref=model)
        for task in tasks:
            items = load_items(
                task,
                samples_dir=samples_dir,
                data_root=data_root,
                items_dir=items_dir,
                docs_gold=docs_gold,
            )
            estimate = estimate_cell_cost(
                task,
                items,
                table,
                model_id=spec.ref.model_id,
                max_tokens=spec.max_tokens,
            )
            if estimate is None:
                return None
            total += estimate
    return total


# ------------------------------------------------------------------------ stop rule


def _scrub_text(value: str) -> str:
    return _SECRET_TEXT.sub("[redacted]", value)[:600]


def _status_code(exc: BaseException) -> int | None:
    direct = getattr(exc, "status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _policy_reason(exc: BaseException) -> str | None:
    """A 403, an abuse/policy/suspension body or an account warning (never a retry)."""
    status = _status_code(exc)
    text = _scrub_text(f"{type(exc).__name__}: {exc}").lower()
    if status == 403:
        return _scrub_text(f"HTTP 403: {exc}")
    if any(word in text for word in _POLICY_WORDS):
        return _scrub_text(f"{type(exc).__name__}: {exc}")
    return None


def _write_stop(reports_dir: Path, reason: str, run_id: str) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "STOP.md"
    path.write_text(
        "# STOP\n\n"
        f"- Run: `{run_id}`\n"
        f"- When: {datetime.now(UTC).isoformat().replace('+00:00', 'Z')}\n"
        f"- Reason: {_scrub_text(reason)}\n\n"
        "Every live call is stopped. PLAN.md §10: escalate; never retry around this.\n",
        encoding="utf-8",
    )
    return path


def _write_deviation(reports_dir: Path, note: str) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "DEVIATIONS.md"
    if not path.exists():
        path.write_text(
            "# Deviations from `PLAN.md`\n\n"
            "`PLAN.md` §1-§11 are frozen and are never edited here. This file records "
            "changes the captain ordered explicitly, with the reason, as the plan requires.\n",
            encoding="utf-8",
        )
    stamp = datetime.now(UTC).date().isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {stamp} — T08 budget guard (automatic)\n\n{note}\n")


# ----------------------------------------------------------------------- scheduling


def seconds_until_off_peak(now: datetime, windows: Sequence[PeakWindow]) -> float:
    """Seconds until ``now`` is outside every weekday peak window (UTC)."""
    moment = _aware(now)
    if moment.weekday() >= 5:
        return 0.0
    current = moment.timetz().replace(tzinfo=None)
    for window in windows:
        if window.start <= current < window.end:
            end = moment.replace(
                hour=window.end.hour,
                minute=window.end.minute,
                second=window.end.second,
                microsecond=0,
            )
            return max(0.0, (end - moment).total_seconds())
    return 0.0


# ------------------------------------------------------------------------ execution


@dataclass
class _RawCall:
    key: str
    latency_ms: int
    response: dict[str, Any]
    error: str | None
    cost_usd: float | None
    measured_cost_usd: float | None
    price_cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class _ItemResult:
    prediction: Any
    calls: list[_RawCall]


@dataclass
class _CellRun:
    predictions: list[Any] = field(default_factory=list)
    calls: list[tuple[str, _RawCall]] = field(default_factory=list)


def _failed_response() -> dict[str, Any]:
    return {"content": None, "tool_calls": [], "usage": {}}


def _retry_after(exc: BaseException) -> float:
    value = getattr(exc, "retry_after", None)
    if isinstance(value, (int, float)) and value >= 0:
        return float(value)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if isinstance(headers, Mapping):
        raw = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return float(raw) if raw is not None else 60.0
        except (TypeError, ValueError):
            return 60.0
    return 60.0


class _Runner:
    """State shared by the per-item call closures (one instance per run)."""

    def __init__(
        self,
        *,
        mode: RecordingMode,
        run_id: str,
        recorder: Recorder,
        budget: Budget,
        prices: Mapping[str, Any],
        config: BudgetConfig,
        invoker: Callable[[ModelRef, Sequence[Message]], tuple[dict[str, Any], int]] | None,
        limiter: RateLimiter,
        clock: Clock,
        reports_dir: Path,
        peak_wait: bool = True,
    ) -> None:
        self.mode = mode
        self.run_id = run_id
        self.recorder = recorder
        self.budget = budget
        self.prices = prices
        self.config = config
        self.invoker = invoker
        self.limiter = limiter
        self.clock = clock
        self.reports_dir = reports_dir
        self.peak_wait = peak_wait
        self.waits: list[str] = []
        self._schedule_lock = threading.Lock()

    def _schedule(self, model: ModelRef) -> None:
        if not self.peak_wait:
            return
        windows = self.config.peak_windows.get(model.model_id)
        if not windows:
            return
        with self._schedule_lock:
            wait = seconds_until_off_peak(self.clock.now(), windows)
            if wait > 0:
                self.clock.sleep(wait)
                self.waits.append(f"{model.model_id}: waited {wait:.0f}s for off-peak")

    def _invoke(
        self, model: ModelRef, messages: Sequence[Message]
    ) -> tuple[dict[str, Any], int, str | None]:
        if self.invoker is None:  # pragma: no cover - guarded before the run
            msg = "record mode needs a model invoker"
            raise BenchStopError(msg)
        self._schedule(model)
        key = f"{model.connection_id}::{model.model_id}"
        last_error: BaseException | None = None
        for attempt in range(4):
            try:
                with self.limiter.acquire(key):
                    response, latency_ms = self.invoker(model, messages)
                    return response, latency_ms, None
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                reason = _policy_reason(exc)
                if reason is not None:
                    _write_stop(self.reports_dir, reason, self.run_id)
                    raise PolicyStopError(reason) from exc
                last_error = exc
                status = _status_code(exc)
                if attempt < 3 and (status == 429 or "rate limit" in str(exc).lower()):
                    self.clock.sleep(_retry_after(exc))
                    continue
                if attempt < 3:
                    self.clock.sleep(1.0)
                    continue
                break
        assert last_error is not None
        if _status_code(last_error) == 429:
            reason = f"still rate limited after retries: {last_error}"
            _write_stop(self.reports_dir, reason, self.run_id)
            raise BenchStopError(reason) from last_error
        return _failed_response(), 0, _scrub_text(f"{type(last_error).__name__}: {last_error}")

    def make_call(
        self,
        *,
        task: str,
        item_id: str,
        model: ModelRef,
        prompt_version: str,
        collected: list[_RawCall],
    ) -> Call:
        def call(messages: Sequence[Message]) -> ModelAnswer:
            digest = request_digest(list(messages))
            key = recording_key(model.model_id, prompt_version, digest)
            if self.mode is RecordingMode.REPLAY:
                try:
                    response = self.recorder.lookup(key, task=task)
                except ReplayMiss as exc:
                    msg = (
                        f"replay miss for {task}/{model.model_id}/{item_id}; "
                        "record it first (replay never calls a model)"
                    )
                    raise ReplayIncompleteError(msg) from exc
                if response is None:  # pragma: no cover - replay lookup raises
                    msg = f"replay miss for {key}"
                    raise ReplayIncompleteError(msg)
                raw = _priced_raw_call(self._raw_call(key, response, 0, None), model, self.prices)
                collected.append(raw)
                self.budget.spend(raw.cost_usd, model_id=model.model_id)
                return ModelAnswer(key=key, response=response, latency_ms=0)

            existing = self.recorder.lookup(key, task=task)
            if existing is not None:
                raw = _priced_raw_call(self._raw_call(key, existing, 0, None), model, self.prices)
                collected.append(raw)
                self.budget.spend(raw.cost_usd, model_id=model.model_id)
                return ModelAnswer(key=key, response=existing, latency_ms=0)

            response, latency_ms, error = self._invoke(model, messages)
            raw = _priced_raw_call(
                self._raw_call(key, response, latency_ms, error), model, self.prices
            )
            collected.append(raw)
            self.budget.spend(raw.cost_usd, model_id=model.model_id)
            if raw.error is None:
                self.recorder.record(
                    key,
                    model.model_id,
                    prompt_version,
                    digest,
                    response,
                    task=task,
                )
            return ModelAnswer(key=key, response=response, latency_ms=latency_ms)

        return call

    def _raw_call(
        self,
        key: str,
        response: Mapping[str, Any],
        latency_ms: int,
        error: str | None,
    ) -> _RawCall:
        usage = extract_usage(response)
        return _RawCall(
            key=key,
            latency_ms=latency_ms,
            response=dict(response),
            error=error,
            cost_usd=None,
            measured_cost_usd=usage.measured_cost_usd,
            price_cost_usd=None,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )


def _priced_raw_call(raw: _RawCall, model: ModelRef, prices: Mapping[str, Any]) -> _RawCall:
    price_cost = cost_usd(model.model_id, raw.input_tokens, raw.output_tokens, prices)
    measured = raw.measured_cost_usd
    raw.price_cost_usd = price_cost
    raw.cost_usd = measured if measured is not None else price_cost
    return raw


def _execute_item(
    runner: _Runner,
    predictor: Predictor | SpecPredictorLike,
    item: Mapping[str, Any],
    model: ModelRef,
) -> _ItemResult:
    collected: list[_RawCall] = []
    call = runner.make_call(
        task=str(predictor.task),
        item_id=str(item["item_id"]),
        model=model,
        prompt_version=predictor.prompt_version,
        collected=collected,
    )
    prediction = predictor.predict(item, model, call)
    return _ItemResult(prediction=prediction, calls=collected)


def _run_cell(
    runner: _Runner,
    predictor: Predictor | SpecPredictorLike,
    items: Sequence[Mapping[str, Any]],
    model: ModelRef,
    *,
    workers: int,
) -> _CellRun:
    cell = _CellRun()
    if workers <= 1:
        for item in items:
            result = _execute_item(runner, predictor, item, model)
            cell.predictions.append(result.prediction)
            cell.calls.extend((result.prediction.item_id, raw) for raw in result.calls)
        return cell

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_execute_item, runner, predictor, item, model) for item in items]
        for future in futures:
            result = future.result()
            cell.predictions.append(result.prediction)
            cell.calls.extend((result.prediction.item_id, raw) for raw in result.calls)
    return cell


# ---------------------------------------------------------------------------- plan


@dataclass(frozen=True)
class _CellPlan:
    items: list[dict[str, Any]]
    estimate_usd: float | None
    reductions: tuple[str, ...] = ()


def _plan_cell(
    *,
    task: TaskName,
    items: Sequence[Mapping[str, Any]],
    spec: ModelSpec,
    budget: Budget,
    prices: Mapping[str, Any],
    reports_dir: Path,
    reduce_per_intent: int,
) -> _CellPlan:
    chosen = [dict(item) for item in items]
    estimate = estimate_cell_cost(
        task,
        chosen,
        prices,
        model_id=spec.ref.model_id,
        max_tokens=spec.max_tokens,
    )
    reductions: list[str] = []
    if estimate is not None and not budget.can_spend(estimate, model_id=spec.ref.model_id):
        if task is TaskName.ROUTE:
            reduced = reduce_route_items(chosen, per_intent=reduce_per_intent)
            if len(reduced) < len(chosen):
                reduced_estimate = estimate_cell_cost(
                    task,
                    reduced,
                    prices,
                    model_id=spec.ref.model_id,
                    max_tokens=spec.max_tokens,
                )
                note = (
                    f"T-ROUTE reduced to {reduce_per_intent} items per intent "
                    f"({len(reduced)} items) for `{spec.ref.model_id}`: estimated "
                    f"${reduced_estimate if reduced_estimate is not None else 0:.4f} "
                    f"exceeds the remaining allowance (${estimate:.4f} for the "
                    f"{len(chosen)}-item cell)."
                )
                _write_deviation(reports_dir, note)
                reductions.append(note)
                chosen = reduced
                estimate = reduced_estimate
        if estimate is not None and not budget.can_spend(estimate, model_id=spec.ref.model_id):
            ceiling = budget.ceiling(spec.ref.model_id)
            remaining_total = budget.remaining_total()
            if remaining_total is not None and estimate > remaining_total + 1e-9:
                reason = (
                    f"budget cap: `{spec.ref.model_id}` {task.value} cell still needs "
                    f"${estimate:.4f} after the reduction but only ${remaining_total:.4f} "
                    "remains"
                )
                _write_stop(reports_dir, reason, "budget-guard")
                raise BenchStopError(reason)
            if ceiling is None or estimate > ceiling + 1e-9:
                reason = (
                    f"budget window: `{spec.ref.model_id}` {task.value} cell needs "
                    f"${estimate:.4f}, which does not fit even a fresh ${ceiling or 0:.2f} window"
                )
                _write_stop(reports_dir, reason, "budget-guard")
                raise BenchStopError(reason)
            budget.wait_for_window(spec.ref.model_id)
            if not budget.can_spend(estimate, model_id=spec.ref.model_id):
                reason = (
                    f"budget window: `{spec.ref.model_id}` {task.value} still cannot fit "
                    f"${estimate:.4f} after waiting for a new window"
                )
                _write_stop(reports_dir, reason, "budget-guard")
                raise BenchStopError(reason)
    return _CellPlan(items=chosen, estimate_usd=estimate, reductions=tuple(reductions))


def _stratified_subset(
    items: Sequence[Mapping[str, Any]], *, n: int, label_field: str | None
) -> list[dict[str, Any]]:
    """A proportional subset that keeps every label's share (``PLAN-v1.1`` §A10).

    A spec sample is stratified by its label, and T-DEFECT's committed file is grouped
    by component group, so cutting the first ``n`` rows would drop whole groups and make
    macro-F1 incomparable with the full cells. Quotas use the largest remainder over the
    primary label's truth field, and the kept rows stay in file order.
    """
    chosen = [dict(item) for item in items]
    if n <= 0:
        return []
    if n >= len(chosen) or label_field is None:
        return chosen[:n]
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(chosen):
        groups.setdefault(str(item.get(label_field, "")), []).append(index)
    total = len(chosen)
    quotas = {key: (n * len(indexes)) // total for key, indexes in groups.items()}
    remainder = n - sum(quotas.values())
    ranked = sorted(groups.items(), key=lambda pair: (-((n * len(pair[1])) % total), pair[0]))
    for key, _indexes in ranked:
        if remainder <= 0:
            break
        quotas[key] += 1
        remainder -= 1
    keep: set[int] = set()
    for key, indexes in groups.items():
        keep.update(indexes[: quotas[key]])
    return [item for index, item in enumerate(chosen) if index in keep]


def _plan_spec_cell(
    *,
    spec: TaskSpec,
    items: Sequence[Mapping[str, Any]],
    model_spec: ModelSpec,
    budget: Budget,
    prices: Mapping[str, Any],
    reports_dir: Path,
    reduce_items: int,
    root: Path,
) -> _CellPlan:
    """The v1 budget guard for one spec cell (``PLAN-v1.1`` §A10).

    The reduction path cuts the cell to ``reduce_items`` items first (400 for T-DEFECT),
    logs that in ``reports/DEVIATIONS.md``, then waits for a window or stops with
    ``reports/STOP.md`` exactly as a built-in cell does.
    """
    chosen = [dict(item) for item in items]
    estimate = estimate_spec_cell_cost(
        spec,
        chosen,
        prices,
        model_id=model_spec.ref.model_id,
        max_tokens=model_spec.max_tokens,
        root=root,
    )
    reductions: list[str] = []
    if estimate is not None and not budget.can_spend(estimate, model_id=model_spec.ref.model_id):
        if 0 < reduce_items < len(chosen):
            primary_name = spec.metrics.primary.split(".", 1)[0]
            label_field = next(
                (output.truth_field for output in spec.outputs if output.name == primary_name),
                None,
            )
            reduced = _stratified_subset(chosen, n=reduce_items, label_field=label_field)
            reduced_estimate = estimate_spec_cell_cost(
                spec,
                reduced,
                prices,
                model_id=model_spec.ref.model_id,
                max_tokens=model_spec.max_tokens,
                root=root,
            )
            note = (
                f"{spec.id} reduced to {len(reduced)} items for `{model_spec.ref.model_id}`: "
                f"estimated ${reduced_estimate if reduced_estimate is not None else 0:.4f} "
                f"exceeds the remaining allowance (${estimate:.4f} for the "
                f"{len(chosen)}-item cell)."
            )
            _write_deviation(reports_dir, note)
            reductions.append(note)
            chosen = reduced
            estimate = reduced_estimate
        if estimate is not None and not budget.can_spend(
            estimate, model_id=model_spec.ref.model_id
        ):
            ceiling = budget.ceiling(model_spec.ref.model_id)
            remaining_total = budget.remaining_total()
            if remaining_total is not None and estimate > remaining_total + 1e-9:
                reason = (
                    f"budget cap: `{model_spec.ref.model_id}` {spec.id} cell still needs "
                    f"${estimate:.4f} after the reduction but only ${remaining_total:.4f} "
                    "remains"
                )
                _write_stop(reports_dir, reason, "budget-guard")
                raise BenchStopError(reason)
            if ceiling is None or estimate > ceiling + 1e-9:
                reason = (
                    f"budget window: `{model_spec.ref.model_id}` {spec.id} cell needs "
                    f"${estimate:.4f}, which does not fit even a fresh ${ceiling or 0:.2f} window"
                )
                _write_stop(reports_dir, reason, "budget-guard")
                raise BenchStopError(reason)
            budget.wait_for_window(model_spec.ref.model_id)
            if not budget.can_spend(estimate, model_id=model_spec.ref.model_id):
                reason = (
                    f"budget window: `{model_spec.ref.model_id}` {spec.id} still cannot fit "
                    f"${estimate:.4f} after waiting for a new window"
                )
                _write_stop(reports_dir, reason, "budget-guard")
                raise BenchStopError(reason)
    return _CellPlan(items=chosen, estimate_usd=estimate, reductions=tuple(reductions))


# ----------------------------------------------------------------------------- run


def _build_limiter(specs: Sequence[ModelSpec], config: BudgetConfig) -> RateLimiter:
    limits: dict[str, Limits] = {}
    for spec in specs:
        connection_id = spec.ref.connection_id
        limit = config.limits.get(connection_id)
        if limit is None and connection_id.startswith("opencode-go"):
            limit = config.limits.get("opencode-go")
        if limit is None:
            limit = Limits(max_concurrency=2, requests_per_minute=30)
        limits[f"{connection_id}::{spec.ref.model_id}"] = limit
    return RateLimiter(limits)


def _default_invoker(
    config_dir: Path,
    specs: Sequence[ModelSpec] = (),
    *,
    session_id: str | None = None,
) -> Callable[[ModelRef, Sequence[Message]], tuple[dict[str, Any], int]]:
    from which_ai_model.connect import load_connections

    connections = {
        connection.id: connection for connection in load_connections(config_dir / "providers.toml")
    }
    by_key = {(spec.ref.connection_id, spec.ref.model_id): spec for spec in specs}
    cache: dict[tuple[str, str], Any] = {}
    lock = threading.Lock()

    def invoke(model: ModelRef, messages: Sequence[Message]) -> tuple[dict[str, Any], int]:
        key = (model.connection_id, model.model_id)
        with lock:
            chat = cache.get(key)
            if chat is None:
                connection = connections[model.connection_id]
                from which_ai_model.core import make_chat_model

                spec = by_key.get(key)
                options: dict[str, Any] = {}
                if spec is not None:
                    options["reasoning"] = spec.reasoning
                    options["max_tokens"] = spec.max_tokens
                    # The lock's per-model temperature: a number is sent, and
                    # "provider-default" sends no parameter at all (PLAN.md §7).
                    if spec.temperature is not None:
                        options["temperature"] = (
                            None if spec.temperature == "provider-default" else spec.temperature
                        )
                if session_id is not None:
                    options["session_id"] = session_id
                chat = make_chat_model(connection, model.model_id, **options)
                cache[key] = chat
        started = time.perf_counter()
        message = chat.invoke(list(messages))
        latency_ms = int((time.perf_counter() - started) * 1000)
        return response_from_message(message), latency_ms

    return invoke


@dataclass(frozen=True)
class SpecCallRecord:
    """One spec-task model call, written beside the v1 ``CallRecord`` rows.

    ``CallRecord.task`` is a ``TaskName`` and stays built-in-only, so a spec row carries
    its spec id in the same field name as plain text. Every other field matches.
    """

    run_id: str
    task: str
    tool: str
    item_id: str
    model: ModelRef
    prompt_version: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    schema_valid: bool | None = None
    error: str | None = None
    response_key: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "tool": self.tool,
            "item_id": self.item_id,
            "model": self.model.model_dump(mode="json"),
            "prompt_version": self.prompt_version,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "schema_valid": self.schema_valid,
            "error": self.error,
            "response_key": self.response_key,
        }


def _write_calls(path: Path, rows: Sequence[CallRecord | SpecCallRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, SpecCallRecord):
                handle.write(json.dumps(row.as_json(), sort_keys=True) + "\n")
            else:
                handle.write(row.model_dump_json() + "\n")


def _spec_sample_hash(specs: Sequence[TaskSpec], root: Path) -> str:
    """A stable hash of the spec data files, so sample drift is refused like v1's."""
    digest = hashlib.sha256()
    for task_spec in sorted(specs, key=lambda item: item.id):
        digest.update(task_spec.id.encode("utf-8"))
        digest.update(b"\0")
        path = root / task_spec.data.file
        if path.is_file():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _selected_refs(tasks: Sequence[TaskRef] | None) -> tuple[str, ...]:
    """Normalise the requested task refs, keeping order and dropping duplicates."""
    raw = list(tasks) if tasks else [task.value for task in TaskName]
    refs: list[str] = []
    for value in raw:
        if builtin_task(value) is None and not re.fullmatch(SPEC_ID_PATTERN, value):
            msg = (
                f"task ref {value!r} is neither a built-in task nor a spec id "
                "(lower-case letters, digits and hyphens, e.g. 'defect')"
            )
            raise ValueError(msg)
        key = task_key(value)
        if key not in refs:
            refs.append(key)
    if not refs:
        msg = "no tasks selected"
        raise ValueError(msg)
    return tuple(refs)


def run(
    mode: str = "replay",
    *,
    config: Path = Path("config"),
    run_id: str | None = None,
    models_path: Path | None = None,
    tasks: Sequence[TaskRef] | None = None,
    specs: Sequence[TaskSpec] | None = None,
    spec_predictors: Mapping[str, SpecPredictorLike] | None = None,
    tasks_dir: Path = Path("tasks"),
    root: Path = Path(),
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
    samples_dir: Path = DEFAULT_SAMPLES_DIR,
    data_root: Path = DEFAULT_DATA_ROOT,
    items_dir: Path | None = None,
    docs_gold: Path = DEFAULT_DOCS_GOLD,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    reports_dir: Path = Path("reports"),
    recommendations_path: Path = Path("results/recommendations.json"),
    baseline_metrics: Path = Path("results/baseline/metrics.json"),
    registry: Mapping[TaskName, Predictor] | None = None,
    invoker: Callable[[ModelRef, Sequence[Message]], tuple[dict[str, Any], int]] | None = None,
    clock: Clock | None = None,
    workers: int | None = None,
    reduce_per_intent: int = 5,
    spec_reduce_items: int = 400,
    model_ids: Sequence[str] | None = None,
    fresh: bool = False,
    fresh_root: Path = DEFAULT_FRESH_RECORDINGS_ROOT,
    settings: Mapping[str, RunSettings] | None = None,
    lock_sha256: str | None = None,
    temperature: float = 0.0,
    peak_wait: bool = True,
    cap_usd: float | None = None,
) -> RunManifest:
    """Run every selected model x task x item in ``replay`` or ``record`` mode.

    ``tasks`` accepts built-in names and spec ids (``--tasks defect``). A spec task runs
    through M1's generic predictor and scorer, shares the same budget, pacing and stop
    paths as v1, and writes into ``results/runs/<run_id>/`` beside the built-in rows.
    ``specs`` / ``spec_predictors`` let a caller supply an already-loaded spec; otherwise
    ``<tasks_dir>/<id>.toml`` is read from ``root``.

    ``fresh=True`` writes to ``<fresh_root>/<run_id>/`` and never reads the saved
    recording root. A record run adds the per-model ``settings`` block and the lock
    file's sha256 to the manifest; replay leaves the v1 manifest body untouched.
    ``peak_wait=False`` skips the v1 off-peak wait (the v1.1 run calls DeepSeek whenever
    it reaches it, ``PLAN-v1.1`` §A10). ``cap_usd`` overrides the optional
    ``[budget].cap_usd`` from ``config/budget.toml`` for this run only (the v1.1 hard
    cap); ``None`` keeps the committed configuration.
    """
    try:
        recording_mode = RecordingMode(mode)
    except ValueError as exc:
        msg = f"unsupported mode {mode!r}; use replay or record"
        raise ValueError(msg) from exc
    if recording_mode not in (RecordingMode.REPLAY, RecordingMode.RECORD):
        msg = f"unsupported mode {mode!r}; use replay or record"
        raise ValueError(msg)

    resolved_clock = clock or Clock()
    started = _aware(resolved_clock.now())
    resolved_run_id = run_id or (f"{recording_mode.value}-{started.strftime('%Y%m%dT%H%M%SZ')}")
    resolved_models = models_path or (config / "models.lock.toml")
    model_specs = load_models(resolved_models, model_ids=model_ids)
    recording_root = (
        fresh_recording_root(resolved_run_id, root=fresh_root) if fresh else Path(recordings_dir)
    )
    recorded_settings: dict[str, RunSettings] = (
        dict(settings)
        if settings is not None
        else (
            {
                model_spec.ref.model_id: run_settings_for(model_spec, temperature=temperature)
                for model_spec in model_specs
            }
            if recording_mode is RecordingMode.RECORD
            else {}
        )
    )
    selected_refs = _selected_refs(tasks)
    selected_builtin = [TaskName(ref) for ref in selected_refs if builtin_task(ref) is not None]
    selected_spec_ids = [ref for ref in selected_refs if builtin_task(ref) is None]

    provided_specs = {spec.id: spec for spec in (specs or ())}
    resolved_specs: list[TaskSpec] = []
    for spec_id in selected_spec_ids:
        task_spec = provided_specs.get(spec_id)
        if task_spec is None:
            task_spec = load_specs([spec_id], tasks_dir=tasks_dir)[0]
        resolved_specs.append(task_spec)
    specs_by_id = {spec.id: spec for spec in resolved_specs}

    budget_config = load_budget_config(config / "budget.toml")
    prices = load_prices(config / "prices.toml")
    if not prices:
        prices = {}
    predictors: dict[str, Predictor | SpecPredictorLike] = {
        task.value: predictor for task, predictor in (registry or default_registry()).items()
    }
    missing_tasks = [task for task in selected_builtin if task.value not in predictors]
    if missing_tasks:
        msg = f"no predictor registered for: {', '.join(task.value for task in missing_tasks)}"
        raise ValueError(msg)
    predictor_versions = {predictors[task.value].prompt_version for task in selected_builtin}
    if predictor_versions not in (set(), {PROMPT_VERSION}):
        msg = (
            "predictors must report the committed prompt version "
            f"{PROMPT_VERSION!r}; got {sorted(predictor_versions)!r}"
        )
        raise ValueError(msg)
    provided_predictors = spec_predictors or {}
    for task_spec in resolved_specs:
        predictor = provided_predictors.get(task_spec.id)
        if predictor is None:
            predictor = spec_predictor(task_spec, root=root)
        if predictor.prompt_version != task_spec.prompt_version:
            msg = (
                f"the {task_spec.id!r} predictor reports prompt version "
                f"{predictor.prompt_version!r}; the spec says {task_spec.prompt_version!r}"
            )
            raise ValueError(msg)
        predictors[task_spec.id] = predictor

    resolved_invoker = invoker
    if resolved_invoker is None and recording_mode is not RecordingMode.REPLAY:
        resolved_invoker = _default_invoker(config, model_specs, session_id=resolved_run_id)

    budget = Budget(
        budget_config.cap_usd if cap_usd is None else cap_usd,
        ceilings=budget_config.window_ceilings,
        window_s=budget_config.window_hours * 3600,
        clock=resolved_clock.monotonic,
        sleep=resolved_clock.sleep,
    )
    recorder = Recorder(recording_root, mode=recording_mode, reuse=not fresh)
    limiter = _build_limiter(model_specs, budget_config)
    runner = _Runner(
        mode=recording_mode,
        run_id=resolved_run_id,
        recorder=recorder,
        budget=budget,
        prices=prices,
        config=budget_config,
        invoker=resolved_invoker,
        limiter=limiter,
        clock=resolved_clock,
        reports_dir=reports_dir,
        peak_wait=peak_wait,
    )

    run_dir = Path(runs_dir) / resolved_run_id
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    items_by_task: dict[str, list[dict[str, Any]]] = {}
    for builtin in selected_builtin:
        items_by_task[builtin.value] = load_items(
            builtin,
            samples_dir=samples_dir,
            data_root=data_root,
            items_dir=items_dir,
            docs_gold=docs_gold,
        )
    for task_spec in resolved_specs:
        items_by_task[task_spec.id] = load_spec_items(task_spec, root=root)

    cells: list[CellResult] = []
    spec_cells: list[SpecCellResult] = []
    all_calls: list[CallRecord | SpecCallRecord] = []
    estimated_by_model: dict[str, float] = {}
    measured_by_model: dict[str, float] = {}
    price_by_model: dict[str, float] = {}
    reductions: list[str] = []
    waits: list[str] = []
    overruns: list[str] = []

    for model_spec in model_specs:
        for ref in selected_refs:
            task = builtin_task(ref)
            cell_workers = workers
            if cell_workers is None:
                key = f"{model_spec.ref.connection_id}::{model_spec.ref.model_id}"
                cell_workers = limiter.limits_for(key).max_concurrency
            if task is not None:
                plan = _plan_cell(
                    task=task,
                    items=items_by_task[ref],
                    spec=model_spec,
                    budget=budget,
                    prices=prices,
                    reports_dir=reports_dir,
                    reduce_per_intent=reduce_per_intent,
                )
                if plan.estimate_usd is not None:
                    estimated_by_model[model_spec.ref.model_id] = (
                        estimated_by_model.get(model_spec.ref.model_id, 0.0) + plan.estimate_usd
                    )
                reductions.extend(plan.reductions)
                predictor = predictors[ref]
                cell = _run_cell(
                    runner,
                    predictor,
                    plan.items,
                    model_spec.ref,
                    workers=max(1, cell_workers),
                )
                valid_by_item = {
                    prediction.item_id: (
                        bool(prediction.output.get("answer") is not None)
                        and "cited_ids" in prediction.output
                        if task is TaskName.DOCS
                        else bool(prediction.output)
                    )
                    for prediction in cell.predictions
                }
                for item_id, raw in cell.calls:
                    usage = extract_usage(raw.response)
                    all_calls.append(
                        CallRecord(
                            run_id=resolved_run_id,
                            task=task,
                            tool="bench",
                            item_id=item_id,
                            model=model_spec.ref,
                            prompt_version=predictor.prompt_version,
                            latency_ms=raw.latency_ms,
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            cost_usd=raw.cost_usd,
                            schema_valid=valid_by_item.get(item_id),
                            error=raw.error,
                            response_key=raw.key,
                        )
                    )
                    _add_spend(raw, model_spec.ref.model_id, measured_by_model, price_by_model)
                _write_predictions(
                    predictions_dir / f"{task.value}-{model_slug(model_spec.ref.model_id)}.jsonl",
                    cell.predictions,
                )
                truth = {str(item["item_id"]): item for item in plan.items}
                cells.append(
                    evaluate_cell(
                        task,
                        model_spec.ref,
                        cell.predictions,
                        truth,
                        [row for _item_id, row in cell.calls],
                        schema_valid=valid_by_item,
                    )
                )
            else:
                task_spec = specs_by_id[ref]
                plan = _plan_spec_cell(
                    spec=task_spec,
                    items=items_by_task[ref],
                    model_spec=model_spec,
                    budget=budget,
                    prices=prices,
                    reports_dir=reports_dir,
                    reduce_items=spec_reduce_items,
                    root=root,
                )
                if plan.estimate_usd is not None:
                    estimated_by_model[model_spec.ref.model_id] = (
                        estimated_by_model.get(model_spec.ref.model_id, 0.0) + plan.estimate_usd
                    )
                reductions.extend(plan.reductions)
                predictor = predictors[ref]
                cell = _run_cell(
                    runner,
                    predictor,
                    plan.items,
                    model_spec.ref,
                    workers=max(1, cell_workers),
                )
                valid_by_item = {
                    prediction.item_id: bool(prediction.output) for prediction in cell.predictions
                }
                for item_id, raw in cell.calls:
                    usage = extract_usage(raw.response)
                    all_calls.append(
                        SpecCallRecord(
                            run_id=resolved_run_id,
                            task=task_spec.id,
                            tool="bench",
                            item_id=item_id,
                            model=model_spec.ref,
                            prompt_version=predictor.prompt_version,
                            latency_ms=raw.latency_ms,
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            cost_usd=raw.cost_usd,
                            schema_valid=valid_by_item.get(item_id),
                            error=raw.error,
                            response_key=raw.key,
                        )
                    )
                    _add_spend(raw, model_spec.ref.model_id, measured_by_model, price_by_model)
                _write_spec_predictions(
                    predictions_dir / f"{task_spec.id}-{model_slug(model_spec.ref.model_id)}.jsonl",
                    cell.predictions,
                )
                truth = {str(item["item_id"]): item for item in plan.items}
                spec_cells.append(
                    evaluate_spec_cell(
                        task_spec,
                        model_spec.ref,
                        cell.predictions,
                        truth,
                        [row for _item_id, row in cell.calls],
                        schema_valid=valid_by_item,
                    )
                )
            ceiling = budget.ceiling(model_spec.ref.model_id)
            if ceiling is not None:
                spent_in_window = budget.window_spent(model_spec.ref.model_id)
                if spent_in_window > ceiling + 1e-9:
                    overrun = (
                        f"`{model_spec.ref.model_id}` spent ${spent_in_window:.4f} in its "
                        f"5-hour window, over the ${ceiling:.2f} ceiling; the next cell "
                        "waits for a new window (the spend is recorded, not hidden)."
                    )
                    overruns.append(overrun)
                    _write_deviation(reports_dir, overrun)

    finished = _aware(resolved_clock.now())
    manifest_prompt_version = PROMPT_VERSION
    manifest_scorer_version = SCORER_VERSION
    if not selected_builtin and resolved_specs:
        prompt_versions = {task_spec.prompt_version for task_spec in resolved_specs}
        scorer_versions = {task_spec.scorer_version for task_spec in resolved_specs}
        if len(prompt_versions) == 1 and len(scorer_versions) == 1:
            manifest_prompt_version = prompt_versions.pop()
            manifest_scorer_version = scorer_versions.pop()
    manifest_sample_hash = sample_hash(samples_dir)
    if resolved_specs:
        manifest_sample_hash = hashlib.sha256(
            (manifest_sample_hash + _spec_sample_hash(resolved_specs, root)).encode("utf-8")
        ).hexdigest()
    manifest = RunManifest(
        run_id=resolved_run_id,
        sample_hash=manifest_sample_hash,
        prompt_version=manifest_prompt_version,
        scorer_version=manifest_scorer_version,
        models=[model_spec.ref for model_spec in model_specs],
        started_at=started,
        finished_at=finished,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if recorded_settings:
        resolved_lock_sha = lock_sha256
        if resolved_lock_sha is None and recording_mode is RecordingMode.RECORD:
            resolved_lock_sha = _lock_sha256(resolved_models)
        write_manifest_settings(manifest_path, recorded_settings, lock_sha256=resolved_lock_sha)
    _write_calls(run_dir / "calls.jsonl", all_calls)
    write_cells(run_dir / "cells.json", cells, spec_cells=spec_cells)
    write_recommendations(
        recommendations_path,
        build_recommendations_with_specs(
            cells,
            spec_cells,
            resolved_specs,
            baseline=baseline_candidates(baseline_metrics),
            good_enough_points=GOOD_ENOUGH_POINTS,
        ),
    )
    _write_budget_report(
        run_dir / "budget.json",
        budget_config=budget_config,
        budget=budget,
        estimated_by_model=estimated_by_model,
        measured_by_model=measured_by_model,
        price_by_model=price_by_model,
        reductions=reductions,
        waits=[*waits, *runner.waits],
        overruns=overruns,
    )
    return manifest


def _add_spend(
    raw: _RawCall,
    model_id: str,
    measured_by_model: dict[str, float],
    price_by_model: dict[str, float],
) -> None:
    if raw.measured_cost_usd is not None:
        measured_by_model[model_id] = measured_by_model.get(model_id, 0.0) + raw.measured_cost_usd
    if raw.price_cost_usd is not None:
        price_by_model[model_id] = price_by_model.get(model_id, 0.0) + raw.price_cost_usd


def _write_predictions(path: Path, predictions: Sequence[Prediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(prediction.model_dump_json() + "\n")


def _write_spec_predictions(path: Path, predictions: Sequence[Any]) -> None:
    """One spec prediction per line, with the spec id in ``task`` exactly like v1."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            row = {
                "item_id": prediction.item_id,
                "task": prediction.task,
                "model": prediction.model.model_dump(mode="json"),
                "output": prediction.output,
                "raw_key": prediction.raw_key,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _gap_pct(measured: float, priced: float) -> float | None:
    if priced <= 0 or measured <= 0:
        return None
    return (measured - priced) / priced * 100.0


def _write_budget_report(
    path: Path,
    *,
    budget_config: BudgetConfig,
    budget: Budget,
    estimated_by_model: Mapping[str, float],
    measured_by_model: Mapping[str, float],
    price_by_model: Mapping[str, float],
    reductions: Sequence[str],
    waits: Sequence[str],
    overruns: Sequence[str],
) -> None:
    measured_total = sum(measured_by_model.values())
    price_total = sum(price_by_model.values())
    covered = sorted(set(measured_by_model) & set(price_by_model))
    measured_covered = sum(measured_by_model[model] for model in covered)
    price_covered = sum(price_by_model[model] for model in covered)
    payload = {
        # The effective cap, so a run that overrode the committed budget (the v1.1
        # `--cap-usd`) records the limit it was actually held to.
        "cap_usd": budget.cap_usd,
        "window_hours": budget_config.window_hours,
        "window_ceilings_usd": dict(budget_config.window_ceilings),
        "estimated_usd_by_model": dict(sorted(estimated_by_model.items())),
        "spent_usd_by_model": dict(sorted(budget.spent_by_model().items())),
        "measured_usd_by_model": dict(sorted(measured_by_model.items())),
        "price_table_usd_by_model": dict(sorted(price_by_model.items())),
        "measured_cost_available_by_model": {
            model: model in measured_by_model for model in sorted(price_by_model)
        },
        "total_estimated_usd": (sum(estimated_by_model.values()) if estimated_by_model else None),
        "total_spent_usd": budget.total_spent(),
        "total_measured_usd": measured_total,
        "total_price_table_usd": price_total,
        # The measured/price gap is only meaningful over the models where the gateway
        # actually sent a list_cost. Comparing a partial measured sum with the full
        # price-table sum would report a fake gap.
        "measured_coverage_models": covered,
        "total_measured_covered_usd": measured_covered,
        "total_price_covered_usd": price_covered,
        "measured_gap_pct": _gap_pct(measured_covered, price_covered),
        "unknown_cost_calls": budget.unknown_calls,
        "reductions": list(reductions),
        "waits": list(waits),
        "window_overruns": list(overruns),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
