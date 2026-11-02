"""Run-settings comparability (``PLAN-v1.1`` §A5): REFUSED, WARN, provider-default."""

from __future__ import annotations

from which_ai_model.contracts import RunSettings
from which_ai_model.gate import SettingsCheck, check_settings

MODEL = "gpt-6-luna"


def test_identical_settings_are_comparable() -> None:
    settings = RunSettings(temperature=0.0, reasoning="minimal", max_output_tokens=1024)
    check = check_settings({MODEL: settings}, {MODEL: settings})
    assert check == SettingsCheck(comparable=True, differences=())
    assert check.refused is False


def test_a_v1_run_reads_as_provider_default() -> None:
    """v1 manifests carry no temperature, so the caller reads them as provider-default."""
    v1 = {MODEL: RunSettings.provider_default()}
    same = {MODEL: RunSettings(temperature="provider-default")}
    assert check_settings(v1, same) == SettingsCheck(comparable=True, differences=())


def test_a_recorded_temperature_differs_from_a_v1_run() -> None:
    check = check_settings(
        {MODEL: RunSettings.provider_default()},
        {MODEL: RunSettings(temperature=0.0)},
    )
    assert check.refused is True
    assert check.differences == (
        f"temperature differs for {MODEL}: baseline 'provider-default', candidate 0.0",
    )


def test_every_differing_setting_is_named() -> None:
    baseline = {MODEL: RunSettings(temperature=0.0, reasoning="minimal", max_output_tokens=1024)}
    candidate = {MODEL: RunSettings(temperature=0.7, reasoning="high", max_output_tokens=2048)}
    check = check_settings(baseline, candidate)
    assert check.refused is True
    assert check.differences == (
        f"temperature differs for {MODEL}: baseline 0.0, candidate 0.7",
        f"reasoning differs for {MODEL}: baseline 'minimal', candidate 'high'",
        f"max_output_tokens differs for {MODEL}: baseline 1024, candidate 2048",
    )


def test_an_accepted_mismatch_is_comparable_and_keeps_the_differences() -> None:
    baseline = {MODEL: RunSettings(temperature=0.0)}
    candidate = {MODEL: RunSettings(temperature=0.7)}
    check = check_settings(baseline, candidate, accept_mismatch=True)
    assert check.comparable is True
    assert check.refused is False
    assert check.differences == (f"temperature differs for {MODEL}: baseline 0.0, candidate 0.7",)


def test_settings_are_compared_per_model() -> None:
    check = check_settings(
        {"a": RunSettings(temperature=0.0), "b": RunSettings(temperature=0.0)},
        {"a": RunSettings(temperature=0.0), "b": RunSettings(temperature=1.0)},
    )
    assert check.differences == ("temperature differs for b: baseline 0.0, candidate 1.0",)


def test_a_model_only_in_the_candidate_is_not_a_settings_difference() -> None:
    check = check_settings(
        {"a": RunSettings()},
        {"a": RunSettings(), "new": RunSettings(temperature=0.0)},
    )
    assert check == SettingsCheck(comparable=True, differences=())
