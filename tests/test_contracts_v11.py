"""v1.1 contracts (``PLAN-v1.1`` §A3-§A9): a valid spec parses, and the main invalid
ones are rejected with a message a non-programmer can act on."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from which_ai_model.contracts import (
    PROVIDER_DEFAULT,
    BaselineSpec,
    CardData,
    CardModelRow,
    DataSpec,
    GateFloor,
    GateSpec,
    MetricSpec,
    OutputKind,
    OutputSpec,
    PromptLibraryEntry,
    RunSettings,
    TaskName,
    TaskRef,
    TaskSpec,
    VolumeRow,
    builtin_task,
    task_key,
)


def valid_spec(**overrides: Any) -> dict[str, Any]:
    """The §A3 example spec, as a mapping ready for ``TaskSpec``."""
    spec: dict[str, Any] = {
        "id": "defect",
        "title": "Sort electrical complaints and flag safety",
        "description": "Sort vehicle owner complaints into component groups and flag safety.",
        "prompt_version": "d1",
        "scorer_version": "s1",
        "data": {
            "file": "data/defect/test.jsonl",
            "id_field": "id",
            "text_field": "text",
        },
        "prompt": {"template_file": "tasks/defect.prompt.txt"},
        "outputs": [
            {
                "name": "component_group",
                "kind": "label",
                "labels_file": "data/defect/groups.toml",
                "allow_unclear": True,
                "truth_field": "component_group",
            },
            {
                "name": "safety_relevant",
                "kind": "bool",
                "truth_field": "safety_relevant",
            },
        ],
        "metrics": {
            "primary": "component_group.macro_f1",
            "report": [
                "component_group.accuracy",
                "component_group.coverage",
                "component_group.accuracy_answered",
                "safety_relevant.recall",
                "safety_relevant.precision",
                "schema_valid_rate",
            ],
        },
        "gate": {
            "primary_drop_points": 2.0,
            "schema_valid_floor": 0.98,
            "floors": [{"metric": "safety_relevant.recall", "min": 0.90, "max_drop_points": 2.0}],
        },
        "baseline": {
            "component_group": {
                "kind": "tfidf_logreg",
                "train_file": "data/defect/train.jsonl.gz",
            },
            "safety_relevant": {
                "kind": "keyword",
                "keywords_file": "data/defect/safety_keywords.txt",
            },
        },
    }
    spec.update(overrides)
    return spec


def test_valid_spec_parses_as_frozen() -> None:
    spec = TaskSpec.model_validate(valid_spec())
    assert spec.id == "defect"
    assert spec.prompt_version == "d1"
    assert spec.scorer_version == "s1"
    assert [output.name for output in spec.outputs] == ["component_group", "safety_relevant"]
    assert spec.outputs[0].kind is OutputKind.LABEL
    assert spec.outputs[0].allow_unclear is True
    assert spec.outputs[1].kind is OutputKind.BOOL
    assert spec.metrics.primary == "component_group.macro_f1"
    assert spec.gate.schema_valid_floor == 0.98
    assert spec.gate.floors[0].metric == "safety_relevant.recall"
    assert spec.baseline["component_group"].kind == "tfidf_logreg"
    with pytest.raises(ValidationError):
        spec.title = "changed"  # type: ignore[misc]


def test_spec_round_trips_through_json() -> None:
    spec = TaskSpec.model_validate(valid_spec())
    again = TaskSpec.model_validate_json(spec.model_dump_json())
    assert again == spec


def test_defaults_are_the_documented_ones() -> None:
    spec = TaskSpec.model_validate(
        {
            **valid_spec(),
            "data": {"file": "data/defect/test.jsonl"},
            "gate": {"floors": [{"metric": "safety_relevant.recall", "min": 0.9}]},
            "baseline": {},
        }
    )
    assert spec.data.id_field == "id"
    assert spec.data.text_field == "text"
    assert spec.gate.primary_drop_points == 2.0
    assert spec.gate.schema_valid_floor == 0.98
    assert spec.baseline == {}
    assert spec.metrics.report == valid_spec()["metrics"]["report"]


@pytest.mark.parametrize(
    "task_id",
    ["Defect", "T-defect", "defect_1", "-defect", "defect-", "", "9defect"],
)
def test_spec_id_must_be_lower_case(task_id: str) -> None:
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(valid_spec(id=task_id))


def test_output_names_must_be_unique() -> None:
    spec = valid_spec()
    spec["outputs"] = [spec["outputs"][1], spec["outputs"][1]]
    with pytest.raises(ValidationError, match="not unique"):
        TaskSpec.model_validate(spec)


def test_metric_must_name_a_known_output() -> None:
    spec = valid_spec()
    spec["metrics"] = {"primary": "safety_related.recall", "report": []}
    with pytest.raises(ValidationError, match="does not name one of the outputs"):
        TaskSpec.model_validate(spec)


def test_metric_must_fit_the_output_kind() -> None:
    spec = valid_spec()
    spec["metrics"] = {"primary": "safety_relevant.macro_f1", "report": []}
    with pytest.raises(ValidationError, match="has no 'macro_f1' metric"):
        TaskSpec.model_validate(spec)


def test_label_output_needs_labels_file() -> None:
    with pytest.raises(ValidationError, match="needs labels_file"):
        OutputSpec(name="component_group", kind=OutputKind.LABEL, truth_field="component_group")


def test_bool_output_cannot_have_labels_file() -> None:
    with pytest.raises(ValidationError, match="cannot set labels_file"):
        OutputSpec(
            name="safety_relevant",
            kind=OutputKind.BOOL,
            truth_field="safety_relevant",
            labels_file="data/defect/groups.toml",
        )


def test_bool_output_cannot_allow_unclear() -> None:
    with pytest.raises(ValidationError, match="cannot allow 'unclear'"):
        OutputSpec(
            name="safety_relevant",
            kind=OutputKind.BOOL,
            truth_field="safety_relevant",
            allow_unclear=True,
        )


def test_gate_floor_needs_a_bound() -> None:
    with pytest.raises(ValidationError, match="needs min, max_drop_points or both"):
        GateFloor(metric="safety_relevant.recall")


def test_gate_floor_min_is_a_rate() -> None:
    with pytest.raises(ValidationError):
        GateFloor(metric="safety_relevant.recall", min=1.5)


def test_schema_valid_floor_is_a_rate() -> None:
    with pytest.raises(ValidationError):
        GateSpec(schema_valid_floor=98.0)


def test_gate_floors_are_unique() -> None:
    floor = GateFloor(metric="safety_relevant.recall", min=0.9)
    with pytest.raises(ValidationError, match="same metric twice"):
        GateSpec(floors=[floor, floor])


def test_gate_floor_metric_must_be_known() -> None:
    spec = valid_spec()
    spec["gate"] = {"floors": [{"metric": "safety_relevant.specificity", "min": 0.9}]}
    with pytest.raises(ValidationError, match="has no 'specificity' metric"):
        TaskSpec.model_validate(spec)


def test_report_metrics_are_unique() -> None:
    with pytest.raises(ValidationError, match="same metric twice"):
        MetricSpec(primary="component_group.macro_f1", report=["schema_valid_rate"] * 2)


def test_baseline_must_name_a_known_output() -> None:
    spec = valid_spec()
    spec["baseline"] = {"safety": {"kind": "keyword", "keywords_file": "k.txt"}}
    with pytest.raises(ValidationError, match="unknown output 'safety'"):
        TaskSpec.model_validate(spec)


def test_baseline_needs_a_source() -> None:
    with pytest.raises(ValidationError, match="needs train_file or keywords_file"):
        BaselineSpec(kind="tfidf_logreg")


def test_data_spec_requires_a_file() -> None:
    with pytest.raises(ValidationError):
        DataSpec(file="")


def test_task_ref_accepts_builtins_and_spec_ids() -> None:
    for value in ("T-ROUTE", "T-SCOPE", "T-REPLY", "T-DOCS", "defect", "my-task-2"):
        card = CardData(
            task=value,
            title="t",
            job="j",
            data_source="d",
            licence="MIT",
            sample_size=1,
            prompt_version="d1",
            recheck="after a new model",
        )
        assert card.task == value


def test_task_ref_accepts_a_task_name_member() -> None:
    card = CardData(
        task=TaskName.ROUTE,
        title="t",
        job="j",
        data_source="d",
        licence="MIT",
        sample_size=1,
        prompt_version="v1",
        recheck="after a new model",
    )
    assert card.task == "T-ROUTE"


@pytest.mark.parametrize("value", ["Defect", "T-route", "defect_1", "", "T-UNKNOWN"])
def test_task_ref_rejects_anything_else(value: str) -> None:
    with pytest.raises(ValidationError):
        CardData(
            task=value,
            title="t",
            job="j",
            data_source="d",
            licence="MIT",
            sample_size=1,
            prompt_version="d1",
            recheck="after a new model",
        )


def test_task_ref_helpers() -> None:
    assert builtin_task(TaskName.ROUTE) is TaskName.ROUTE
    assert builtin_task("T-DOCS") is TaskName.DOCS
    assert builtin_task("defect") is None
    assert task_key(TaskName.REPLY) == "T-REPLY"
    assert task_key("defect") == "defect"


def test_task_ref_is_a_validated_type() -> None:
    adapter = TypeAdapter(TaskRef)
    assert adapter.validate_python("defect") == "defect"
    assert adapter.validate_python(TaskName.SCOPE) == "T-SCOPE"
    with pytest.raises(ValidationError):
        adapter.validate_python("Defect")


def test_run_settings_defaults_to_provider_default() -> None:
    settings = RunSettings.provider_default()
    assert settings.temperature == PROVIDER_DEFAULT
    assert settings.reasoning == "minimal"
    assert settings.max_output_tokens == 1024
    assert RunSettings() == settings


def test_run_settings_accepts_a_float_temperature() -> None:
    settings = RunSettings(temperature=0.0, reasoning="low", max_output_tokens=2048)
    assert settings.temperature == 0.0
    assert settings.reasoning == "low"
    assert settings.max_output_tokens == 2048


@pytest.mark.parametrize("temperature", [-0.1, 2.5])
def test_run_settings_temperature_is_bounded(temperature: float) -> None:
    with pytest.raises(ValidationError, match=r"between 0\.0 and 2\.0"):
        RunSettings(temperature=temperature)


def test_run_settings_rejects_a_non_positive_token_allowance() -> None:
    with pytest.raises(ValidationError):
        RunSettings(max_output_tokens=0)


def test_volume_row_keeps_an_unknown_cost_unknown() -> None:
    unknown = VolumeRow(model_id="deepseek-v4.1-flash", items_per_month=1_000, cost_usd=None)
    assert unknown.cost_usd is None
    known = VolumeRow(model_id="deepseek-v4.1-flash", items_per_month=10_000, cost_usd=1.25)
    assert known.cost_usd == 1.25
    with pytest.raises(ValidationError):
        VolumeRow(model_id="m", items_per_month=-1)


def test_card_data_holds_the_nine_card_items() -> None:
    card = CardData.model_validate(
        {
            "task": "defect",
            "title": "Sort electrical complaints and flag safety",
            "job": "Sort complaints into component groups and flag safety.",
            "data_source": "NHTSA ODI complaints",
            "licence": "US government work",
            "sample_size": 600,
            "prompt_version": "d1",
            "models": [
                {
                    "model_id": "deepseek-v4.1-flash",
                    "primary_metric": 0.71,
                    "primary_ci": [0.68, 0.74],
                    "metrics": {"component_group.coverage": 0.95},
                    "cost_per_1k": 0.42,
                    "latency_p50": 810.0,
                }
            ],
            "baseline": {
                "model_id": "tfidf_logreg",
                "primary_metric": 0.55,
                "is_baseline": True,
            },
            "cheapest": "deepseek-v4.1-flash",
            "most_accurate": "gemini-4-argon",
            "fastest": "gpt-6-luna",
            "gate": {"floors": [{"metric": "safety_relevant.recall", "min": 0.9}]},
            "recheck": "wam recheck run --tasks defect --against results/runs/live-v1.1",
            "volumes": [
                {"model_id": "deepseek-v4.1-flash", "items_per_month": 1_000, "cost_usd": 0.42}
            ],
            "limits": ["one gateway"],
        }
    )
    assert card.task == "defect"
    assert card.models[0].primary_ci == (0.68, 0.74)
    assert card.baseline is not None
    assert card.baseline.is_baseline
    assert card.no_model_meets_floor is False
    assert "3.0 points" in card.picker_rule
    assert card.volumes[0].cost_usd == 0.42


def test_card_model_row_cost_is_optional() -> None:
    row = CardModelRow(model_id="m", primary_metric=0.5)
    assert row.cost_per_1k is None
    assert row.latency_p50 is None


def test_prompt_library_entry() -> None:
    entry = PromptLibraryEntry.model_validate(
        {
            "task": "defect",
            "title": "Sort electrical complaints and flag safety",
            "when_to_use": "When sorting owner complaints by component.",
            "prompt": "Groups: {labels}\nComplaint: {text}",
            "prompt_version": "d1",
            "output_schema": {"component_group": "string", "safety_relevant": "boolean"},
            "recommended_model": "deepseek-v4.1-flash",
            "fallback_model": "gpt-6-luna",
            "guardrails": ["unclear goes to a human", "never auto-close"],
            "recheck_trigger": "a new model, or a model updated under the same id",
        }
    )
    assert entry.recommended_model == "deepseek-v4.1-flash"
    assert entry.output_schema["safety_relevant"] == "boolean"
    assert entry.guardrails[0] == "unclear goes to a human"


def test_prompt_library_entry_requires_a_prompt() -> None:
    with pytest.raises(ValidationError):
        PromptLibraryEntry(
            task="defect",
            title="t",
            when_to_use="w",
            prompt="",
            prompt_version="d1",
            recheck_trigger="r",
        )
