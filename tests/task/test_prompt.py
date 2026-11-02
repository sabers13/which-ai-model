"""Labels and prompt rendering: every id and description reaches the one prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

from which_ai_model.task import SpecError, load_labels, load_spec, render_prompt

FIXTURE = Path("tests/fixtures/task")


def test_render_prompt_fills_labels_and_text(repo_root: Path) -> None:
    spec = load_spec(repo_root / FIXTURE / "example.toml")
    labels = load_labels(spec, root=repo_root)
    prompt = render_prompt(spec, labels, "The invoice is wrong.", root=repo_root)
    assert "- billing: Invoices, payments, refunds and plan changes" in prompt
    assert "- technical: Errors, outages, setup and anything that needs a fix" in prompt
    assert "The invoice is wrong." in prompt
    assert "{labels}" not in prompt
    assert "{text}" not in prompt


def test_render_prompt_keeps_label_order(repo_root: Path) -> None:
    spec = load_spec(repo_root / FIXTURE / "example.toml")
    prompt = render_prompt(
        spec,
        {"technical": "Second", "billing": "First"},
        "Text",
        root=repo_root,
    )
    assert prompt.index("- technical: Second") < prompt.index("- billing: First")


def test_render_prompt_without_labels_says_so(repo_root: Path) -> None:
    spec = load_spec(repo_root / FIXTURE / "example.toml")
    prompt = render_prompt(spec, {}, "Text", root=repo_root)
    assert "- (no labels)" in prompt


def test_render_prompt_leaves_json_braces_alone(tmp_path: Path) -> None:
    template = tmp_path / "template.txt"
    template.write_text('For {text} answer {"queue": "<id>"} from {labels}', encoding="utf-8")
    spec = load_spec(Path("tests/fixtures/task/example.toml"))
    spec = spec.model_copy(
        update={"prompt": spec.prompt.model_copy(update={"template_file": str(template)})}
    )
    prompt = render_prompt(spec, {"billing": "Billing"}, "Text")
    assert prompt == 'For Text answer {"queue": "<id>"} from - billing: Billing'
    assert "{labels}" not in prompt


def test_missing_template_is_a_plain_error(tmp_path: Path) -> None:
    spec = load_spec(Path("tests/fixtures/task/example.toml"))
    spec = spec.model_copy(
        update={
            "prompt": spec.prompt.model_copy(update={"template_file": str(tmp_path / "no.txt")})
        }
    )
    with pytest.raises(SpecError, match="Cannot read the prompt template"):
        render_prompt(spec, {}, "Text")
