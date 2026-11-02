"""Contracts are the Wave 0 interface: prove the frozen shape behaves as documented."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from which_ai_model.contracts import (
    SCHEMA_VERSION,
    CallRecord,
    CellResult,
    Connection,
    DiscoveredModel,
    DocsOut,
    ModelRef,
    Prediction,
    ProbeReason,
    ProbeResult,
    Protocol,
    ReplyOut,
    RouteOut,
    RunManifest,
    ScopeOut,
    TaskName,
    TriState,
)


def test_enum_vocabularies_are_exact() -> None:
    assert [p.value for p in Protocol] == [
        "OPENAI_COMPATIBLE",
        "OPENAI_RESPONSES",
        "ANTHROPIC",
        "GEMINI",
    ]
    assert [r.value for r in ProbeReason] == [
        "OK",
        "CONFIG_INVALID",
        "CREDENTIAL_MISSING",
        "CREDENTIAL_REJECTED",
        "ENDPOINT_UNREACHABLE",
        "RATE_LIMITED",
        "PROTOCOL_UNSUPPORTED",
        "UNKNOWN",
    ]
    assert [t.value for t in TriState] == ["yes", "no", "unknown"]
    assert [t.value for t in TaskName] == ["T-ROUTE", "T-SCOPE", "T-REPLY", "T-DOCS"]
    assert SCHEMA_VERSION == "1.0.0"


def test_connection_accepts_an_env_reference_and_exposes_the_name() -> None:
    conn = Connection(
        id="opencode-go-anthropic",
        provider="opencode-go",
        protocol=Protocol.ANTHROPIC,
        base_url="https://opencode.ai/inference/go/anthropic",
        credential_ref="env:OPENCODE_GO_API_KEY",
        preset="opencode-go",
    )
    assert conn.credential_env == "OPENCODE_GO_API_KEY"
    assert conn.base_url == "https://opencode.ai/inference/go/anthropic"


@pytest.mark.parametrize(
    "credential_ref",
    ["opencode-go", "env:lowercase", "env:", "sk-live-123", "env:KEY-WITH-DASH"],
)
def test_connection_rejects_non_env_credential_refs(credential_ref: str) -> None:
    with pytest.raises(ValidationError):
        Connection(
            id="custom",
            provider="custom",
            protocol=Protocol.OPENAI_COMPATIBLE,
            credential_ref=credential_ref,
            preset="custom",
        )


def test_connection_rejects_url_credentials_and_secret_looking_values() -> None:
    with pytest.raises(ValidationError, match="base_url must not carry credentials"):
        Connection(
            id="custom",
            provider="custom",
            protocol=Protocol.OPENAI_COMPATIBLE,
            base_url="https://user:pass@example.com/v1",
            preset="custom",
        )
    with pytest.raises(ValidationError, match="credential_ref"):
        Connection(
            id="custom",
            provider="custom",
            protocol=Protocol.OPENAI_COMPATIBLE,
            base_url="https://example.com/?api_key=sk-abcdefghijklmnop",
            preset="custom",
        )


def test_contracts_are_frozen() -> None:
    conn = Connection(
        id="local", provider="local", protocol=Protocol.OPENAI_COMPATIBLE, preset="local"
    )
    with pytest.raises(ValidationError):
        conn.id = "other"  # type: ignore[misc]


def test_model_ref_is_hashable_and_probe_result_validates() -> None:
    ref = ModelRef(connection_id="opencode-go-openai", model_id="deepseek-v4.1-flash")
    assert ref in {ref}
    probe = ProbeResult(connection_id=ref.connection_id, reason=ProbeReason.OK, latency_ms=42)
    assert probe.detail is None


def test_discovered_model_requires_an_aware_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        DiscoveredModel(
            connection_id="c",
            model_id="m",
            listed=TriState.UNKNOWN,
            observed_at=datetime(2026, 1, 1),  # noqa: DTZ001 - deliberately naive
        )
    model = DiscoveredModel(
        connection_id="c",
        model_id="m",
        listed=TriState.UNKNOWN,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert model.listed is TriState.UNKNOWN


def test_run_manifest_and_cell_result_accept_the_benchmark_shape() -> None:
    ref = ModelRef(connection_id="c", model_id="m")
    manifest = RunManifest(
        run_id="run-1",
        sample_hash="s" * 16,
        prompt_version="p1",
        scorer_version="v1",
        models=[ref],
        started_at=datetime(2026, 10, 28, tzinfo=UTC),
        finished_at=datetime(2026, 10, 28, 1, tzinfo=UTC),
    )
    assert manifest.models == [ref]
    cell = CellResult(
        task=TaskName.ROUTE,
        model=ref,
        n=616,
        primary_metric=0.91,
        metrics={"accuracy": 0.92},
        ci={"macro_f1_low": 0.89},
        latency_p50=500.0,
        latency_p95=900.0,
        cost_per_1k=None,
    )
    assert cell.cost_per_1k is None


def test_call_record_and_prediction_round_trip() -> None:
    ref = ModelRef(connection_id="c", model_id="m")
    record = CallRecord(
        run_id="run-1",
        task=TaskName.REPLY,
        tool="L2",
        item_id="item-1",
        model=ref,
        prompt_version="p1",
        latency_ms=12,
        response_key="k" * 16,
    )
    assert record.schema_valid is None
    prediction = Prediction(
        item_id="item-1",
        task=TaskName.REPLY,
        model=ref,
        output={"intent": "card_arrival"},
        raw_key="k" * 16,
    )
    assert prediction.output["intent"] == "card_arrival"


def test_task_output_contracts() -> None:
    assert RouteOut(intent="card_arrival").intent == "card_arrival"
    assert ScopeOut(out_of_scope=True).out_of_scope is True
    reply = ReplyOut(intent="card_arrival", urgency="low", reply="Your card is on its way.")
    assert reply.model_dump() == {
        "intent": "card_arrival",
        "urgency": "low",
        "reply": "Your card is on its way.",
    }
    docs = DocsOut(answer="See guide.", cited_ids=["chunk-1"])
    assert docs.abstained is False
    with pytest.raises(ValidationError):
        ReplyOut(intent="x", urgency="urgent", reply="no")  # type: ignore[arg-type]
