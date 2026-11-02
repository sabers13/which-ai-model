# Task specifications and re-check API

Public extension types and callable interfaces for task specifications and re-checking.

## 0. Compatibility and data boundaries

- **v1 does not change shape.** `TaskName` keeps its four members. `Prediction`,
  `CellResult`, `CallRecord` and `RunManifest` keep every existing field; v1.1 adds
  optional fields and new types only. A spec task is addressed by its spec `id`
  (`"defect"`), never by a new `TaskName` member.
- **No new `TaskName` member, no spec id in a `TaskName` field.** A spec task's id is a
  `TaskRef` (see §1) and its rows are written beside the v1 rows, not into them.
- **Additive import rules.** `task/` and `recheck/` may import `contracts`, `core`,
  `bench` and `gate`, in the same direction as `docs/INTERFACES.md` §0. `task/` must not
  import `recheck/`. `gate/` must not import `task/` at runtime; the spec gate takes
  duck-typed cells (§3).
- **Offline tests.** Tests use fakes and committed fixtures. `make test` runs
  with sockets blocked.
- **Numbers only from `results/`.** Cards, the prompt library and the page read
  committed files; nothing recomputes a published number from a live call.
- **`docs/guide/` is frozen.** It is the T-DOCS retrieval corpus.

## 1. Contracts (`which_ai_model.contracts`)

All are `frozen=True, extra="forbid"`.

| Type | Fields | Rules |
| --- | --- | --- |
| `TaskRef` | `Annotated[str, AfterValidator]` | A built-in `TaskName` value (`T-ROUTE`, `T-SCOPE`, `T-REPLY`, `T-DOCS`) or a spec id matching `SPEC_ID_PATTERN` (lower-case, digits, single hyphens). Helpers: `builtin_task(ref) -> TaskName \| None`, `task_key(ref) -> str`. |
| `TaskSpec` | `id, title, description, prompt_version, scorer_version, data, prompt, outputs, metrics, gate, baseline` | `outputs` non-empty with unique names; every metric names a known output (or `schema_valid_rate`); `baseline` keys are output names; gate floors name known metrics. |
| `DataSpec` | `file, id_field="id", text_field="text"` | `file` is relative to the repository root. |
| `PromptSpec` | `template_file` | Placeholders `{labels}` and `{text}`; checked by , not here. |
| `OutputSpec` | `name, kind, truth_field, labels_file=None, allow_unclear=False` | `kind` is `label` or `bool`. `label` needs `labels_file`; `bool` must not set `labels_file` or `allow_unclear`. |
| `MetricSpec` | `primary, report=[]` | No duplicate report entries. |
| `GateSpec` | `primary_drop_points=2.0, schema_valid_floor=0.98, floors=[]` | Floors unique by metric. |
| `GateFloor` | `metric, min=None, max_drop_points=None` | At least one bound; `min` in `[0, 1]`. |
| `BaselineSpec` | `kind, train_file=None, keywords_file=None` | At least one source file. |
| `RunSettings` | `temperature="provider-default", reasoning="minimal", max_output_tokens=1024` | Temperature is a float in `[0, 2]` or `"provider-default"`; `RunSettings.unrecorded()` is the v1 reading (temperature `"provider-default"`, reasoning `"not-recorded"`, which a gate skips instead of reporting as a difference). |
| `CardModelRow` | `model_id, primary_metric, primary_ci=None, metrics={}, cost_per_1k=None, latency_p50=None, is_baseline=False` | `cost_per_1k=None` means unknown, never 0. |
| `CardData` | `task, title, job, data_source, licence, sample_size, prompt_version, models, baseline=None, cheapest=None, most_accurate=None, fastest=None, no_model_meets_floor=False, picker_rule, gate, recheck, volumes=[], limits=[], notes=[]` | The task card fields. |
| `VolumeRow` | `model_id, items_per_month, cost_usd=None` | `cost_usd=None` is "unknown", never 0. |
| `PromptLibraryEntry` | `task, title, when_to_use, prompt, prompt_version, output_schema, recommended_model=None, fallback_model=None, guardrails=[], recheck_trigger` | The prompt library entry. |

Metric names are `<output>.<metric>` or `schema_valid_rate`:

| Output kind | Metrics |
| --- | --- |
| `label` | `accuracy`, `macro_f1`, `coverage`, `accuracy_answered` |
| `bool` | `precision`, `recall`, `f1` |

`macro_f1` counts `unclear` as wrong, so abstaining is never free.
`coverage` is the share of items answered, `accuracy_answered` is accuracy among them.

`SCHEMA_VERSION` is **not** bumped: no existing contract changed shape, and v1 manifests
must stay byte-identical.

## 2.  — `which_ai_model.task` (spec loader, predictor, scorer)

```python
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from which_ai_model.bench.run import Call
from which_ai_model.bench.score import TaskScore
from which_ai_model.contracts import ModelRef, TaskSpec


def load_spec(path: Path) -> TaskSpec: ...
def validate_spec(spec: TaskSpec, *, root: Path = Path(".")) -> list[str]:
    """Plain-language problems with the spec and its files; empty means valid."""


def write_skeleton(task_id: str, *, tasks_dir: Path = Path("tasks")) -> tuple[Path, Path]:
    """Write ``<tasks_dir>/<id>.toml`` and ``<tasks_dir>/<id>.prompt.txt`` (spec, template)."""


def load_labels(spec: TaskSpec, *, root: Path = Path(".")) -> dict[str, str]:
    """Label id -> one-line description, from ``labels_file``."""


def render_prompt(
    spec: TaskSpec,
    labels: Mapping[str, str],
    text: str,
    *,
    root: Path = Path("."),
) -> str:
    """The template with ``{labels}`` (id + description per label) and ``{text}`` filled."""


def build_output_model(spec: TaskSpec) -> type[BaseModel]:
    """A pydantic model from ``outputs``: a label is a ``Literal`` of its ids (plus
    ``"unclear"`` when allowed); a bool is ``bool``."""


@dataclass(frozen=True)
class SpecPrediction:
    """One spec answer. ``Prediction`` is unchanged and stays built-in-only."""

    item_id: str
    task: str  # the spec id
    model: ModelRef
    output: dict[str, Any]
    raw_key: str


class SpecPredictor:
    """Implements the runner's predictor shape for one spec task."""

    task: str  # the spec id
    prompt_version: str

    def __init__(self, spec: TaskSpec, *, root: Path = Path(".")) -> None: ...
    def predict(self, item: Mapping[str, Any], model: ModelRef, call: Call) -> SpecPrediction: ...


def score_spec(
    spec: TaskSpec,
    predictions: Sequence[SpecPrediction],
    truth: Mapping[str, Mapping[str, Any]],
    *,
    schema_valid: Mapping[str, bool] | None = None,
) -> TaskScore:
    """Score one model x spec cell; reuse ``bench.metrics`` for the CIs."""
```

Rules: an empty, cut-off or unparseable answer is schema-invalid **and** wrong, with the
reason recorded; it is never retried. The prompt is identical for every model.

## 3.  — `which_ai_model.gate` (spec gate, settings, picker)

```python
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from which_ai_model.contracts import GateFloor, RunSettings, TaskSpec


class SpecCellLike(Protocol):
    """What the spec gate needs from one model x task cell ( supplies it)."""

    model_id: str
    primary_metric: float
    metrics: Mapping[str, float]
    cost_per_1k: float | None
    latency_p50: float | None


@dataclass(frozen=True)
class GateRuleResult:
    name: (
        str  # "primary_drop" | "floor_min:<metric>" | "floor_drop:<metric>" | "schema_valid_floor"
    )
    value: float
    threshold: float
    passed: bool


@dataclass(frozen=True)
class SpecGateResult:
    outcome: GateOutcome  # the existing PASS | FAIL | REFUSED enum
    rules: tuple[GateRuleResult, ...]
    warnings: tuple[str, ...] = ()

    @property
    def exit_code(self) -> int: ...


@dataclass(frozen=True)
class SettingsCheck:
    comparable: bool
    differences: tuple[str, ...] = ()

    @property
    def refused(self) -> bool: ...


def check_settings(
    baseline: Mapping[str, RunSettings],
    candidate: Mapping[str, RunSettings],
    *,
    accept_mismatch: bool = False,
) -> SettingsCheck:
    """A difference is REFUSED unless accepted; an accepted difference is a WARN."""


def evaluate_spec_gate(
    spec: TaskSpec,
    baseline: Mapping[str, SpecCellLike],
    candidate: Mapping[str, SpecCellLike],
    *,
    settings: SettingsCheck | None = None,
) -> SpecGateResult:
    """FAIL on primary drop, a floor below ``min``, a floor drop over ``max_drop_points``
    or schema-valid below the floor; REFUSED when settings are not comparable."""


@dataclass(frozen=True)
class Pick:
    cheapest: str | None
    most_accurate: str | None
    fastest: str | None
    good_enough: tuple[str, ...]
    no_model_meets_floor: bool


def pick(cells: Sequence[SpecCellLike], *, floors: Sequence[GateFloor] = ()) -> Pick:
    """Within 3.0 points of the best primary; a model below a floor is never good enough;
    when no model meets a floor, the three picks are ``None`` and the flag is True."""
```

v1 built-in gate behaviour, thresholds and exit codes stay unchanged. A v1 run with no
recorded temperature reads as `RunSettings.provider_default()`.

## 4.  — `which_ai_model.recheck` and the manifest settings

```python
from collections.abc import Mapping, Sequence
from pathlib import Path

from which_ai_model.contracts import RunSettings, TaskRef


def fresh_root(run_id: str, *, root: Path = Path("recordings/fresh")) -> Path:
    """``<root>/<run_id>``; the only root a fresh run may read or write."""


def run_recheck(
    *,
    models: Path,
    tasks: Sequence[TaskRef],
    against: Path,
    candidate: str | None = None,
    baseline: str | None = None,
    run_id: str | None = None,
    accept_settings_mismatch: bool = False,
) -> Path:
    """Fresh record -> score -> gate -> ``results/runs/<run_id>/recheck.json``."""


# Additive in ``which_ai_model/bench/run.py``:
def manifest_settings(settings: Mapping[str, RunSettings]) -> dict[str, dict[str, object]]:
    """The JSON body of a manifest's ``settings`` block, keyed by model id."""


def write_manifest_settings(
    path: Path,
    settings: Mapping[str, RunSettings],
    *,
    lock_sha256: str | None = None,
) -> None:
    """Add ``settings`` and ``lock_sha256`` to an existing manifest, leaving every other
    key untouched so v1 manifests still load."""
```

`wam bench --fresh` and `wam recheck run` use the same fresh root; a fresh run never
calls `Recorder.lookup()` on a saved root. Default recording behaviour is unchanged.

## 5.  — `which_ai_model.report` (volume and prompt library)

```python
from collections.abc import Sequence
from which_ai_model.contracts import CardData, TaskName, TaskSpec, VolumeRow


def volume_rows(
    card_data: CardData,
    volumes: Sequence[int] = (1_000, 10_000, 100_000),
) -> list[VolumeRow]:
    """``cost_per_month = cost_per_1k * items_per_month / 1000``, linear only.
    A ``None`` cost per 1,000 stays ``None`` (unknown), never 0."""


def export_prompt_library(
    card_data: CardData, spec_or_builtin: TaskSpec | TaskName
) -> tuple[str, str]:
    """``(markdown, json_text)`` for ``results/prompt-library/<task>.{md,json}``."""
```

## 6.  — `which_ai_model.bench` and `which_ai_model.report.cards`

```python
from collections.abc import Sequence
from pathlib import Path

from which_ai_model.contracts import CardData, TaskRef, TaskSpec
from which_ai_model.task import SpecPredictor


# Additive in ``which_ai_model/bench/run.py``:
def spec_predictor(spec: TaskSpec, *, root: Path = Path(".")) -> SpecPredictor: ...
def spec_registry(specs: Sequence[TaskSpec]) -> dict[str, SpecPredictor]:
    """Spec id -> predictor, served to the runner beside ``default_registry()``."""


# ``which_ai_model/report/cards.py``:
def build_card(
    task: TaskRef,
    *,
    runs_dir: Path = Path("results/runs"),
    results_dir: Path = Path("results"),
) -> CardData:
    """Every number from committed files under ``results/``; no model call."""


def write_card(card: CardData, *, out_dir: Path = Path("results/cards")) -> tuple[Path, Path]:
    """``<task>.md`` and ``<task>.json``."""


def build_cards(tasks: Sequence[TaskRef]) -> list[Path]: ...
```

The runner's `--tasks` accepts a `TaskRef`; spec tasks run through 's predictor and
scorer with the same budget, pacing and stop paths as v1, and write
`results/runs/<run_id>/` beside the v1 files. `results/recommendations.json` gains spec
entries through 's `pick`; v1 entries are unchanged.
