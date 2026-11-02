"""The built page is static, deterministic and every visible number comes from page data."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from which_ai_model.report.page import DOWNLOADS, build_page, write_page
from which_ai_model.report.page_data import REPO_ROOT, render_json


def _visible_text(source: str) -> str:
    without_scripts = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", source)
    without_tags = re.sub(r"(?s)<[^>]+>", " ", without_scripts)
    return html.unescape(without_tags)


def _numbers(text: str) -> set[str]:
    return {token.strip(".,") for token in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def test_every_visible_number_is_in_page_data(page: dict[str, Any], repo_root: Path) -> None:
    built = build_page(repo_root, data=page)
    rendered = _visible_text(built)
    payload = render_json(page)
    # German thousands separators may differ; the underlying evidence must match.
    missing = sorted(
        number
        for number in _numbers(rendered)
        if number not in payload and number.replace(".", ",") not in payload
    )
    assert missing == [], f"numbers on the page without a page-data value: {missing}"


def test_build_is_byte_identical_twice(page: dict[str, Any], tmp_path: Path) -> None:
    first = write_page(REPO_ROOT, tmp_path / "one", data=page)
    second = write_page(REPO_ROOT, tmp_path / "two", data=page)
    first_files = {path.relative_to(first) for path in first.rglob("*") if path.is_file()}
    second_files = {path.relative_to(second) for path in second.rglob("*") if path.is_file()}
    assert first_files == second_files
    for relative in sorted(first_files):
        assert (first / relative).read_bytes() == (second / relative).read_bytes(), relative
    assert (first / "index.html").read_bytes() == (second / "index.html").read_bytes()


def test_page_is_static_and_needs_no_network(page: dict[str, Any], repo_root: Path) -> None:
    built = build_page(repo_root, data=page)
    assert '<script src="http' not in built
    assert "@import" not in built
    assert "url(http" not in built
    assert "fetch(" not in built
    assert "XMLHttpRequest" not in built
    allowed = (
        "#",
        "assets/",
        "data:",
        "data/",
        "downloads/",
        "https://github.com/sabers13",
        "https://sabersojudi.com/",
        "https://huggingface.co/datasets/PolyAI/banking77",
    )
    for href in re.findall(r'(?:href|src)="([^"]+)"', built):
        assert href.startswith(allowed), href


def test_written_page_opens_from_disk(page: dict[str, Any], tmp_path: Path) -> None:
    output = write_page(REPO_ROOT, tmp_path / "dist", data=page)
    index = output / "index.html"
    assert index.is_file()
    for asset in re.findall(r'(?:href|src)="(assets/[^"]+)"', index.read_text(encoding="utf-8")):
        assert (output / asset).is_file(), asset
    copied = output / "data/page-data.json"
    assert copied.is_file()
    assert copied.read_text(encoding="utf-8") == render_json(page)


def test_downloads_preserve_the_source_documents(page: dict[str, Any], tmp_path: Path) -> None:
    output = write_page(REPO_ROOT, tmp_path / "dist", data=page)
    for relative in DOWNLOADS:
        source = REPO_ROOT / relative
        assert (output / "downloads" / source.name).read_bytes() == source.read_bytes()
    for licence in ("OFL-inter.txt", "OFL-plex.txt"):
        assert (output / "assets" / "fonts" / licence).is_file()
    assert (output / "assets" / "icons" / "LICENSE-feather.txt").is_file()


def test_accessibility_basics(page: dict[str, Any], repo_root: Path) -> None:
    built = build_page(repo_root, data=page)
    assert built.count("<h1") == 1
    assert '<html lang="en"' in built
    assert 'class="skip"' in built
    assert 'role="radiogroup"' in built
    assert 'aria-live="polite"' in built
    assert "<details" in built
    assert built.count('role="radiogroup"') == 2
    assert built.count('role="radio"') == 6


def test_picker_has_every_priority_for_every_task(page: dict[str, Any], repo_root: Path) -> None:
    built = build_page(repo_root, data=page)
    for task in page["tasks"]:
        for priority in ("cheapest", "most_accurate", "fastest"):
            assert f'data-priority="{priority}"' in built
        assert task["name"] in built


def test_example_shows_all_models_and_verdicts(page: dict[str, Any], repo_root: Path) -> None:
    built = build_page(repo_root, data=page)
    assert built.count('class="example-row is-') == len(page["example"]["rows"])
    for row in page["example"]["rows"]:
        assert row["name"] in built
    verdicts = re.findall(r'class="verdict"[^>]*>([✓✗])<', built)
    assert verdicts.count("✓") == 4
    assert verdicts.count("✗") == 3


def test_no_forbidden_placeholder_text(page: dict[str, Any], repo_root: Path) -> None:
    built = build_page(repo_root, data=page)
    lowered = built.lower()
    assert "todo t12" not in lowered
    assert "lorem ipsum" not in lowered
