"""Shared pydantic v2 contracts.

These models are the interface between lanes. The shape is frozen after T00 (Wave 0):
later lanes may add fields only through a captain-approved change, because results,
recordings and the page all depend on them.

No field may hold a secret. ``Connection.credential_ref`` holds an ``env:NAME`` pointer;
the value itself is resolved at call time and never stored here.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Final, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "PROVIDER_DEFAULT",
    "SCHEMA_VERSION",
    "SPEC_ID_PATTERN",
    "BaselineSpec",
    "CallRecord",
    "CardData",
    "CardModelRow",
    "CellResult",
    "Connection",
    "DataSpec",
    "DiscoveredModel",
    "DocsOut",
    "GateFloor",
    "GateSpec",
    "MetricSpec",
    "ModelRef",
    "OutputKind",
    "OutputSpec",
    "Prediction",
    "ProbeReason",
    "ProbeResult",
    "PromptLibraryEntry",
    "PromptSpec",
    "Protocol",
    "ReplyOut",
    "RouteOut",
    "RunManifest",
    "RunSettings",
    "ScopeOut",
    "TaskName",
    "TaskRef",
    "TaskSpec",
    "TriState",
    "VolumeRow",
    "builtin_task",
    "task_key",
]

#: Bumped only when a contract changes shape. Recorded in run manifests and results.
SCHEMA_VERSION = "1.0.0"

_ENV_REF = r"^env:[A-Z0-9_]+$"
_SECRET_LIKE = re.compile(
    r"(?:^|[\s\"'=:])(?:sk-[A-Za-z0-9_-]{8,}|sk-ant-|AIza[0-9A-Za-z_-]{10,}|lsv2_)"
)


class _Contract(BaseModel):
    """Base for every contract: immutable, closed to unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_aware(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        msg = f"{field} must be timezone-aware (UTC)"
        raise ValueError(msg)
    return value


class Protocol(StrEnum):
    """Wire protocol of a connection, not the lab that makes the model."""

    OPENAI_COMPATIBLE = "OPENAI_COMPATIBLE"
    OPENAI_RESPONSES = "OPENAI_RESPONSES"
    ANTHROPIC = "ANTHROPIC"
    GEMINI = "GEMINI"


class ProbeReason(StrEnum):
    """Fixed probe vocabulary from ``PLAN.md`` §5. Never invent a new value."""

    OK = "OK"
    CONFIG_INVALID = "CONFIG_INVALID"
    CREDENTIAL_MISSING = "CREDENTIAL_MISSING"
    CREDENTIAL_REJECTED = "CREDENTIAL_REJECTED"
    ENDPOINT_UNREACHABLE = "ENDPOINT_UNREACHABLE"
    RATE_LIMITED = "RATE_LIMITED"
    PROTOCOL_UNSUPPORTED = "PROTOCOL_UNSUPPORTED"
    UNKNOWN = "UNKNOWN"


class TriState(StrEnum):
    """Discovery answer. ``unknown`` never becomes ``no``."""

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class Connection(_Contract):
    """A BYOK connection: where to call, which protocol, and which env var holds the key."""

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    provider: str = Field(min_length=1)
    protocol: Protocol
    base_url: str | None = None
    credential_ref: str | None = Field(default=None, pattern=_ENV_REF)
    preset: str = Field(min_length=1)

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.startswith(("http://", "https://")):
            msg = "base_url must start with http:// or https://"
            raise ValueError(msg)
        if "@" in value.split("://", 1)[1].split("/", 1)[0]:
            msg = "base_url must not carry credentials in the URL"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _no_secret_values(self) -> Connection:
        for name in ("id", "provider", "preset", "base_url"):
            value = getattr(self, name)
            if isinstance(value, str) and _SECRET_LIKE.search(value):
                msg = f"{name} looks like it contains a secret; use credential_ref=env:NAME"
                raise ValueError(msg)
        return self

    @property
    def credential_env(self) -> str | None:
        """The environment variable name, or ``None`` for keyless connections."""
        if self.credential_ref is None:
            return None
        return self.credential_ref.removeprefix("env:")


class ProbeResult(_Contract):
    """Outcome of one connection probe. ``detail`` is scrubbed of headers and keys."""

    connection_id: str = Field(min_length=1)
    reason: ProbeReason
    latency_ms: int = Field(ge=0)
    detail: str | None = None


class DiscoveredModel(_Contract):
    """One model id seen (or not seen) for a connection."""

    connection_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    listed: TriState
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def _check_observed(cls, value: datetime) -> datetime:
        return _require_aware(value, "observed_at")


class ModelRef(_Contract):
    """A model on a specific connection. Hashable, so usable as a dict key."""

    connection_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)


class TaskName(StrEnum):
    """The four benchmark tasks from ``PLAN.md`` §7."""

    ROUTE = "T-ROUTE"
    SCOPE = "T-SCOPE"
    REPLY = "T-REPLY"
    DOCS = "T-DOCS"


class CallRecord(_Contract):
    """One model call in the local trace — one JSONL row, no headers, no keys."""

    run_id: str = Field(min_length=1)
    task: TaskName
    tool: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    model: ModelRef
    prompt_version: str = Field(min_length=1)
    latency_ms: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    schema_valid: bool | None = None
    error: str | None = None
    response_key: str = Field(min_length=1)


class Prediction(_Contract):
    """One scored answer. ``output`` is the parsed contract model, as JSON."""

    item_id: str = Field(min_length=1)
    task: TaskName
    model: ModelRef
    output: dict[str, Any]
    raw_key: str = Field(min_length=1)


class CellResult(_Contract):
    """Aggregate for one task by model cell."""

    task: TaskName
    model: ModelRef
    n: int = Field(ge=0)
    primary_metric: float
    metrics: dict[str, float]
    ci: dict[str, float]
    latency_p50: float = Field(ge=0)
    latency_p95: float = Field(ge=0)
    cost_per_1k: float | None = Field(default=None, ge=0)


class RunManifest(_Contract):
    """What a run was scored against. The gate refuses mismatched hashes.

    ``settings`` and ``lock_sha256`` are the v1.1 additions (``PLAN-v1.1`` §A5). They are
    read when present and excluded from ``model_dump``, so a v1 manifest keeps exactly
    its old bytes while a new manifest still loads: ``bench.run.write_manifest_settings``
    writes them into the JSON file beside the existing keys.
    """

    run_id: str = Field(min_length=1)
    sample_hash: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    scorer_version: str = Field(min_length=1)
    models: list[ModelRef] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime
    #: Model id -> ``RunSettings`` JSON body (temperature, reasoning, max output tokens).
    settings: dict[str, dict[str, object]] = Field(default_factory=dict, exclude=True)
    #: sha256 of the model lock file the run used; absent on v1 manifests.
    lock_sha256: str | None = Field(default=None, exclude=True)

    @field_validator("started_at")
    @classmethod
    def _check_started(cls, value: datetime) -> datetime:
        return _require_aware(value, "started_at")

    @field_validator("finished_at")
    @classmethod
    def _check_finished(cls, value: datetime) -> datetime:
        return _require_aware(value, "finished_at")


class RouteOut(_Contract):
    """T-ROUTE: exactly one Banking77 intent label."""

    intent: str = Field(min_length=1)


class ScopeOut(_Contract):
    """T-SCOPE: is the message outside the supported scope?"""

    out_of_scope: bool


class ReplyOut(_Contract):
    """T-REPLY: a structured customer-service reply."""

    intent: str = Field(min_length=1)
    urgency: Literal["low", "medium", "high"]
    reply: str = Field(min_length=1)


class DocsOut(_Contract):
    """T-DOCS: an answer with the passage ids it cites, or an explicit abstention."""

    answer: str
    cited_ids: list[str] = Field(default_factory=list)
    abstained: bool = False


# ---------------------------------------------------------------------------
# v1.1 contracts (PLAN-v1.1 §A3, §A5-§A9). Additive: nothing above changed shape.
# ---------------------------------------------------------------------------

#: A spec task id: lower-case, digits and single hyphens. Used in paths and run ids.
SPEC_ID_PATTERN = r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$"

#: The temperature value a run records when the provider's own default was used.
PROVIDER_DEFAULT: Final[Literal["provider-default"]] = "provider-default"

_OUTPUT_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"
_LABEL_METRICS = frozenset({"accuracy", "macro_f1", "coverage", "accuracy_answered"})
_BOOL_METRICS = frozenset({"precision", "recall", "f1"})
_GLOBAL_METRICS = frozenset({"schema_valid_rate"})
_BUILTIN_TASKS = frozenset(task.value for task in TaskName)


def _check_task_ref(value: str) -> str:
    """Accept a built-in ``TaskName`` value or a spec id; reject anything else."""
    if value in _BUILTIN_TASKS:
        return value
    if not re.fullmatch(SPEC_ID_PATTERN, value):
        msg = (
            f"task ref {value!r} is neither a built-in task "
            f"({', '.join(sorted(_BUILTIN_TASKS))}) nor a spec id "
            "(lower-case letters, digits and hyphens, e.g. 'defect')"
        )
        raise ValueError(msg)
    return value


#: A benchmark task: one of the four built-in ``TaskName`` values, or a spec id.
TaskRef = Annotated[str, AfterValidator(_check_task_ref)]


def builtin_task(ref: TaskRef) -> TaskName | None:
    """Return the built-in task for ``ref``, or ``None`` when it names a spec task."""
    try:
        return TaskName(ref)
    except ValueError:
        return None


def task_key(ref: TaskRef) -> str:
    """The path and run-id key of a task: its built-in name or its spec id."""
    task = builtin_task(ref)
    return task.value if task is not None else ref


class OutputKind(StrEnum):
    """What one spec output is: a label from a closed list, or a boolean."""

    LABEL = "label"
    BOOL = "bool"


class DataSpec(_Contract):
    """Where a spec task's labelled items live and which fields to read."""

    file: str = Field(min_length=1)
    id_field: str = Field(default="id", min_length=1)
    text_field: str = Field(default="text", min_length=1)


class PromptSpec(_Contract):
    """The prompt template file; placeholders are ``{labels}`` and ``{text}``."""

    template_file: str = Field(min_length=1)


class OutputSpec(_Contract):
    """One field the model must return, and the truth field it is scored against."""

    name: str = Field(pattern=_OUTPUT_NAME_PATTERN)
    kind: OutputKind
    truth_field: str = Field(min_length=1)
    labels_file: str | None = None
    allow_unclear: bool = False

    @model_validator(mode="after")
    def _check_kind(self) -> OutputSpec:
        if self.kind is OutputKind.LABEL:
            if self.labels_file is None:
                msg = f"label output {self.name!r} needs labels_file"
                raise ValueError(msg)
        else:
            if self.labels_file is not None:
                msg = f"bool output {self.name!r} cannot set labels_file"
                raise ValueError(msg)
            if self.allow_unclear:
                msg = f"bool output {self.name!r} cannot allow 'unclear'"
                raise ValueError(msg)
        return self


class MetricSpec(_Contract):
    """The primary metric and the metrics a card and the gate report."""

    primary: str = Field(min_length=1)
    report: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_report(self) -> MetricSpec:
        if len(set(self.report)) != len(self.report):
            msg = "metrics.report lists the same metric twice"
            raise ValueError(msg)
        return self


class GateFloor(_Contract):
    """A metric that must stay above ``min`` and may not drop by more than a set amount."""

    metric: str = Field(min_length=1)
    min: float | None = Field(default=None, ge=0, le=1)
    max_drop_points: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> GateFloor:
        if self.min is None and self.max_drop_points is None:
            msg = f"gate floor {self.metric!r} needs min, max_drop_points or both"
            raise ValueError(msg)
        return self


class GateSpec(_Contract):
    """Per-task gate rules (PLAN-v1.1 §A5). Built-in tasks keep the v1 thresholds."""

    primary_drop_points: float = Field(default=2.0, ge=0)
    schema_valid_floor: float = Field(default=0.98, ge=0, le=1)
    floors: list[GateFloor] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_floors(self) -> GateSpec:
        metrics = [floor.metric for floor in self.floors]
        if len(set(metrics)) != len(metrics):
            msg = "gate.floors names the same metric twice"
            raise ValueError(msg)
        return self


class BaselineSpec(_Contract):
    """The cost-0 baseline for one output: a trained model or a fixed keyword rule."""

    kind: str = Field(min_length=1)
    train_file: str | None = None
    keywords_file: str | None = None

    @model_validator(mode="after")
    def _check_source(self) -> BaselineSpec:
        if self.train_file is None and self.keywords_file is None:
            msg = f"baseline {self.kind!r} needs train_file or keywords_file"
            raise ValueError(msg)
        return self


def _metric_problem(metric: str, outputs: dict[str, OutputKind]) -> str | None:
    """Return why ``metric`` is not a metric of this spec, or ``None`` when it is."""
    if metric in _GLOBAL_METRICS:
        return None
    name, _, suffix = metric.partition(".")
    kind = outputs.get(name)
    if not suffix or kind is None:
        return f"metric {metric!r} does not name one of the outputs {sorted(outputs)}"
    allowed = _LABEL_METRICS if kind is OutputKind.LABEL else _BOOL_METRICS
    if suffix not in allowed:
        return (
            f"metric {metric!r}: {kind.value} output {name!r} has no {suffix!r} metric; "
            f"choose from {sorted(allowed)} or {sorted(_GLOBAL_METRICS)}"
        )
    return None


class TaskSpec(_Contract):
    """A benchmark task defined by one TOML file (PLAN-v1.1 §A3), no Python needed."""

    id: str = Field(pattern=SPEC_ID_PATTERN)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    scorer_version: str = Field(min_length=1)
    data: DataSpec
    prompt: PromptSpec
    outputs: list[OutputSpec] = Field(min_length=1)
    metrics: MetricSpec
    gate: GateSpec = Field(default_factory=GateSpec)
    baseline: dict[str, BaselineSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_references(self) -> TaskSpec:
        names = [output.name for output in self.outputs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            msg = f"outputs names {duplicates} are not unique"
            raise ValueError(msg)
        outputs = {output.name: output.kind for output in self.outputs}
        for metric in (self.metrics.primary, *self.metrics.report):
            problem = _metric_problem(metric, outputs)
            if problem is not None:
                raise ValueError(problem)
        for name in self.baseline:
            if name not in outputs:
                msg = f"baseline names unknown output {name!r}; choose from {sorted(outputs)}"
                raise ValueError(msg)
        for floor in self.gate.floors:
            problem = _metric_problem(floor.metric, outputs)
            if problem is not None:
                raise ValueError(problem)
        return self


class RunSettings(_Contract):
    """The sampling settings a run used for one model (PLAN-v1.1 §A5).

    A v1 run recorded no temperature, so the gate reads it as ``provider-default``.
    """

    temperature: float | Literal["provider-default"] = PROVIDER_DEFAULT
    reasoning: str = Field(default="minimal", min_length=1)
    max_output_tokens: int = Field(default=1024, ge=1)

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, value: float | str) -> float | str:
        if isinstance(value, float) and not 0.0 <= value <= 2.0:
            msg = "temperature must be between 0.0 and 2.0, or 'provider-default'"
            raise ValueError(msg)
        return value

    @classmethod
    def provider_default(cls) -> RunSettings:
        """The settings a v1 run is read as having used."""
        return cls()

    @classmethod
    def unrecorded(cls) -> RunSettings:
        """A baseline whose manifest predates the settings block (v1).

        Temperature is still read as ``provider-default`` because that is what the v1
        run did, but the reasoning setting is genuinely unknown: ``not-recorded`` is
        skipped by the comparability check instead of being invented as ``minimal``.
        """
        return cls(reasoning="not-recorded")


class CardModelRow(_Contract):
    """One model's line in a use-case card's results table (PLAN-v1.1 §A7)."""

    model_id: str = Field(min_length=1)
    primary_metric: float
    primary_ci: tuple[float, float] | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    cost_per_1k: float | None = Field(default=None, ge=0)
    latency_p50: float | None = Field(default=None, ge=0)
    is_baseline: bool = False


class VolumeRow(_Contract):
    """Monthly cost for one model at one stated volume (PLAN-v1.1 §A8).

    ``cost_usd`` is ``None`` when the measured cost is unknown; it is never 0 for unknown.
    """

    model_id: str = Field(min_length=1)
    items_per_month: int = Field(ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class CardData(_Contract):
    """Everything a use-case card states, before it is rendered (PLAN-v1.1 §A7)."""

    task: TaskRef
    title: str = Field(min_length=1)
    job: str = Field(min_length=1)
    data_source: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    sample_size: int = Field(ge=0)
    prompt_version: str = Field(min_length=1)
    models: list[CardModelRow] = Field(default_factory=list)
    baseline: CardModelRow | None = None
    cheapest: str | None = None
    most_accurate: str | None = None
    fastest: str | None = None
    no_model_meets_floor: bool = False
    picker_rule: str = (
        "within 3.0 points of the best model on the primary metric; "
        "a model below any gate floor is not good enough; ties go to lower cost"
    )
    gate: GateSpec = Field(default_factory=GateSpec)
    recheck: str = Field(min_length=1)
    volumes: list[VolumeRow] = Field(default_factory=list)
    limits: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PromptLibraryEntry(_Contract):
    """A card as a ready-to-use prompt-library entry (PLAN-v1.1 §A9)."""

    task: TaskRef
    title: str = Field(min_length=1)
    when_to_use: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    recommended_model: str | None = None
    fallback_model: str | None = None
    guardrails: list[str] = Field(default_factory=list)
    recheck_trigger: str = Field(min_length=1)
