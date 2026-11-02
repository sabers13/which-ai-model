"""T06 scanner tests: fixture rows, provider URLs, renderers and the CLI.

The fixtures are source files the scanner parses; no test imports or executes them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from which_ai_model.cli import app as wam
from which_ai_model.inventory import scan, write_json, write_markdown

FIXTURES = Path(__file__).parent / "fixtures"

CASES: dict[str, dict[str, object]] = {
    "raw_http_openai.py": {
        "tool": "route_message",
        "provider": "openai-compatible",
        "call_style": "raw HTTP",
        "model": "gpt-4o-mini",
        "model_source": "literal",
        "credentials": ("OPENAI_API_KEY",),
        "structured_output": "JSON mode",
        "inputs": ("message",),
        "needle": "httpx.post(",
    },
    "anthropic_sdk_regex.py": {
        "tool": "draft_reply",
        "provider": "anthropic",
        "call_style": "vendor SDK",
        "model": "ANTHROPIC_MODEL",
        "model_source": "env",
        "credentials": ("ANTHROPIC_API_KEY",),
        "structured_output": "regex",
        "inputs": ("message",),
        "needle": "client.messages.create(",
    },
    "gemini_sdk_substring.py": {
        "tool": "check_scope",
        "provider": "gemini",
        "call_style": "vendor SDK",
        "model": "gemini-2.5-flash",
        "model_source": "literal",
        "credentials": ("GEMINI_API_KEY",),
        "structured_output": "substring",
        "inputs": ("message",),
        "needle": "client.models.generate_content(",
    },
    "langchain_structured.py": {
        "tool": "route_message",
        "provider": "langchain",
        "call_style": "LangChain",
        "model": None,
        "model_source": None,
        "credentials": (),
        "structured_output": "LangChain structured",
        "inputs": ("message",),
        "needle": "structured.invoke(",
    },
    "tool_loop_openai.py": {
        "tool": "run_agent",
        "provider": "openai-compatible",
        "call_style": "hand-written tool loop",
        "model": "gpt-4o-mini",
        "model_source": "literal",
        "credentials": ("OPENAI_API_KEY",),
        "structured_output": "none",
        "inputs": ("question",),
        "needle": "httpx.post(",
    },
    "unknown_sdk.py": {
        "tool": "complete",
        "provider": "unknown",
        "call_style": "vendor SDK",
        "model": "mystery-1",
        "model_source": "literal",
        "credentials": ("MYSTERY_API_KEY",),
        "structured_output": "none",
        "inputs": ("message",),
        "needle": "client.complete(",
    },
    "legacy_wiring.py": {
        "tool": "ask",
        "provider": "openai-compatible",
        "call_style": "raw HTTP",
        "model": "ASK_MODEL",
        "model_source": "env",
        "credentials": ("ASK_KEY",),
        "structured_output": "regex",
        "inputs": ("message",),
        "needle": "client.post(",
    },
}

FIELDS = (
    "tool",
    "provider",
    "call_style",
    "model",
    "model_source",
    "credentials",
    "structured_output",
    "inputs",
)


def _line_of(path: Path, needle: str) -> int:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return number
    raise AssertionError(f"{needle!r} not found in {path}")


@pytest.mark.parametrize(("filename", "case"), CASES.items(), ids=list(CASES))
def test_fixture_pattern_rows(filename: str, case: dict[str, object]) -> None:
    inventory = scan(FIXTURES)
    sites = [site for site in inventory.sites if site.path.endswith(filename)]
    assert len(sites) == 1, [site.path for site in inventory.sites]
    site = sites[0]
    for field in FIELDS:
        assert getattr(site, field) == case[field], f"{filename}: {field}"
    assert site.line == _line_of(FIXTURES / filename, str(case["needle"]))
    assert site.path.endswith(f"tests/inventory/fixtures/{filename}")


def test_every_fixture_has_exactly_one_site() -> None:
    inventory = scan(FIXTURES)
    assert inventory.summary["modules"] == len(CASES)
    assert inventory.summary["sites"] == len(CASES)
    assert len(inventory.sites) == len(CASES)


def test_sites_are_sorted_and_summary_counts_patterns() -> None:
    inventory = scan(FIXTURES)
    keys = [(site.path, site.line) for site in inventory.sites]
    assert keys == sorted(keys)
    assert inventory.summary["provider:unknown"] == 1
    assert inventory.summary["provider:langchain"] == 1
    assert inventory.summary["call_style:hand-written tool loop"] == 1
    assert inventory.summary["structured_output:JSON mode"] == 1
    assert inventory.summary["structured_output:regex"] == 2
    assert inventory.summary["structured_output:substring"] == 1


def test_scan_single_file(tmp_path: Path) -> None:
    target = FIXTURES / "raw_http_openai.py"
    inventory = scan(target)
    assert inventory.summary["modules"] == 1
    assert inventory.summary["sites"] == 1
    assert inventory.sites[0].path.endswith("raw_http_openai.py")


@pytest.mark.parametrize(
    ("url", "provider"),
    [
        ("https://api.anthropic.com/v1/messages", "anthropic"),
        ("https://generativelanguage.googleapis.com/v1beta/models/m:generateContent", "gemini"),
        ("https://example.invalid/v1/chat/completions", "openai-compatible"),
    ],
)
def test_provider_comes_from_url_literal(tmp_path: Path, url: str, provider: str) -> None:
    module = tmp_path / "caller.py"
    module.write_text(
        "import httpx\n\n"
        "def ask(message: str) -> object:\n"
        f'    return httpx.post("{url}", json={{"model": "m", "messages": [message]}})\n',
        encoding="utf-8",
    )
    inventory = scan(module)
    assert len(inventory.sites) == 1
    assert inventory.sites[0].provider == provider
    assert inventory.sites[0].call_style == "raw HTTP"


def test_model_comes_from_payload_variable(tmp_path: Path) -> None:
    module = tmp_path / "payload.py"
    module.write_text(
        "import httpx\n\n"
        'MODEL = "gpt-4o-mini"\n\n'
        "def ask(message: str) -> object:\n"
        '    payload = {"model": MODEL, "messages": [message]}\n'
        '    return httpx.post("https://api.openai.com/v1/chat/completions", json=payload)\n',
        encoding="utf-8",
    )
    inventory = scan(module)
    assert len(inventory.sites) == 1
    site = inventory.sites[0]
    assert site.model == "gpt-4o-mini"
    assert site.model_source == "literal"
    assert site.provider == "openai-compatible"
    assert site.inputs == ("message",)


def test_for_loop_tool_agent_is_a_tool_loop(tmp_path: Path) -> None:
    module = tmp_path / "agent.py"
    module.write_text(
        "import httpx\n\n"
        "def run_agent(question: str) -> str:\n"
        "    messages = [question]\n"
        "    for _ in range(3):\n"
        "        response = httpx.post(\n"
        '            "http://localhost:8080/v1/chat/completions",\n'
        '            json={"model": "agent-small", "messages": messages, "tools": []},\n'
        "        )\n"
        '        calls = response.json().get("tool_calls")\n'
        "        if not calls:\n"
        '            return ""\n'
        '    return ""\n',
        encoding="utf-8",
    )
    inventory = scan(module)
    assert len(inventory.sites) == 1
    site = inventory.sites[0]
    assert site.call_style == "hand-written tool loop"
    assert site.structured_output == "none"
    assert site.model == "agent-small"
    assert site.inputs == ("question",)


def test_after_tool_through_shared_structured_helper(tmp_path: Path) -> None:
    module = tmp_path / "after_tool.py"
    module.write_text(
        "def route_message(message: str, model: object) -> object:\n"
        "    structured = make_structured_model(model, object)\n"
        "    return structured.invoke(message)\n",
        encoding="utf-8",
    )
    inventory = scan(module)
    assert len(inventory.sites) == 1
    site = inventory.sites[0]
    assert site.provider == "langchain"
    assert site.call_style == "LangChain"
    assert site.structured_output == "LangChain structured"
    assert site.inputs == ("message",)


def test_plain_http_without_llm_evidence_is_ignored(tmp_path: Path) -> None:
    module = tmp_path / "download.py"
    module.write_text(
        "import httpx\n\n"
        "def fetch(url: str) -> bytes:\n"
        "    return httpx.get(url, timeout=10.0).content\n",
        encoding="utf-8",
    )
    inventory = scan(tmp_path)
    assert inventory.sites == []
    assert inventory.summary == {"modules": 1, "sites": 0}


def test_scan_never_imports_fixtures() -> None:
    stems = {path.stem for path in FIXTURES.glob("*.py")}
    assert "raw_http_openai" in stems
    inventory = scan(FIXTURES)
    assert inventory.sites
    assert stems.isdisjoint(sys.modules)


def test_write_json_and_markdown(tmp_path: Path) -> None:
    inventory = scan(FIXTURES)
    json_path = tmp_path / "out" / "inventory.json"
    md_path = tmp_path / "out" / "inventory.md"
    write_json(inventory, json_path)
    write_markdown(inventory, md_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["sites"] == len(CASES)
    assert payload["sites"]
    assert set(payload["sites"][0]) == {
        "path",
        "line",
        "tool",
        "provider",
        "call_style",
        "model",
        "model_source",
        "credentials",
        "structured_output",
        "inputs",
    }

    text = md_path.read_text(encoding="utf-8")
    assert text.startswith("# LLM call-site inventory")
    assert "| File:line | Tool | Provider |" in text
    assert "hand-written tool loop" in text
    assert "env:ANTHROPIC_MODEL" in text
    assert text.endswith("\n")


def test_empty_inventory_markdown(tmp_path: Path) -> None:
    inventory = scan(tmp_path)
    md_path = tmp_path / "empty.md"
    write_markdown(inventory, md_path)
    assert "No LLM call sites found." in md_path.read_text(encoding="utf-8")


def test_markdown_notes_record_a_blind_spot(tmp_path: Path) -> None:
    inventory = scan(FIXTURES)
    md_path = tmp_path / "noted.md"
    write_markdown(inventory, md_path, notes=("Only the visible invokes are counted.",))
    text = md_path.read_text(encoding="utf-8")
    assert "## Notes" in text
    assert text.rstrip().endswith("Only the visible invokes are counted.")
    assert text.endswith("\n")


def test_cli_passes_notes_through(tmp_path: Path) -> None:
    md_path = tmp_path / "after.md"
    result = CliRunner().invoke(
        wam,
        [
            "inventory",
            str(FIXTURES),
            "--md",
            str(md_path),
            "--note",
            "A lower bound, not a call total.",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "A lower bound, not a call total." in md_path.read_text(encoding="utf-8")


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    json_path = tmp_path / "before.json"
    md_path = tmp_path / "before.md"
    result = CliRunner().invoke(
        wam,
        ["inventory", str(FIXTURES), "--json", str(json_path), "--md", str(md_path)],
    )
    assert result.exit_code == 0, result.output
    assert f"{len(CASES)} call site(s)" in result.output
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["sites"] == len(CASES)
    assert md_path.exists()
