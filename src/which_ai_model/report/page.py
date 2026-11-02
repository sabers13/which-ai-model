"""Render the static HR page into ``dist/which-ai-model/``.

The page is plain HTML, self-hosted fonts and dependency-free JavaScript.
Evidence is baked into the HTML. It works offline and under a Pages subdirectory.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape

from which_ai_model.report.page_data import PAGE_DATA_PATH, REPO_ROOT, build_page_data

__all__ = ["DIST_DIR", "WEB_DIR", "build_page", "render_html", "write_page"]

WEB_DIR = Path("web")
DIST_DIR = Path("dist/which-ai-model")
ASSETS = ("page.css", "page.js")
DOWNLOADS = (
    "METHOD.md",
    "THIRD_PARTY.md",
    "results/SUMMARY.md",
    "results/cards/T-ROUTE.md",
    "results/cards/T-SCOPE.md",
    "results/cards/T-REPLY.md",
    "results/cards/T-DOCS.md",
    "results/cards/defect.md",
)


def _inline_code(value: str) -> Markup:
    """Escape a recorded answer, then turn its backtick spans into inline code."""
    escaped = escape(value)
    return Markup(re.sub(r"`([^`]+)`", r"<code>\1</code>", str(escaped)))


def _environment(root: Path) -> Environment:
    environment = Environment(
        loader=FileSystemLoader(str(root / WEB_DIR / "templates")),
        autoescape=select_autoescape(("html", "j2")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["inline_code"] = _inline_code
    return environment


def render_html(page: dict[str, Any], root: Path = REPO_ROOT) -> str:
    """Render the page template with ``page`` and return the HTML string."""
    template = _environment(root).get_template("index.html.j2")
    return template.render(page=page)


def build_page(root: Path = REPO_ROOT, *, data: dict[str, Any] | None = None) -> str:
    """Build the page in memory, reading ``results/page/page-data.json`` by default."""
    if data is None:
        path = root / PAGE_DATA_PATH
        if not path.is_file():
            raise FileNotFoundError(
                f"{PAGE_DATA_PATH.as_posix()} is missing; run make page-data first"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
    return render_html(data, root)


def write_page(
    root: Path = REPO_ROOT,
    output: Path | None = None,
    *,
    data: dict[str, Any] | None = None,
) -> Path:
    """Write the built page and its assets, returning the output directory."""
    if data is None:
        path = root / PAGE_DATA_PATH
        if not path.is_file():
            raise FileNotFoundError(
                f"{PAGE_DATA_PATH.as_posix()} is missing; run make page-data first"
            )
        data = json.loads(path.read_text(encoding="utf-8"))

    destination = output if output is not None else root / DIST_DIR
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "assets").mkdir(parents=True)
    (destination / "data").mkdir(parents=True)

    (destination / "index.html").write_text(render_html(data, root), encoding="utf-8")
    for asset in ASSETS:
        source = root / WEB_DIR / "assets" / asset
        shutil.copyfile(source, destination / "assets" / asset)
    shutil.copytree(root / WEB_DIR / "assets" / "fonts", destination / "assets" / "fonts")
    shutil.copytree(root / WEB_DIR / "assets" / "icons", destination / "assets" / "icons")
    (destination / "downloads").mkdir()
    for relative in DOWNLOADS:
        source = root / relative
        shutil.copyfile(source, destination / "downloads" / source.name)
    (destination / "data" / "page-data.json").write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:  # pragma: no cover - module entry point
    payload = build_page_data()
    destination = write_page(data=payload)
    sys.stdout.write(f"page: wrote {destination.relative_to(REPO_ROOT).as_posix()}/index.html\n")


if __name__ == "__main__":  # pragma: no cover - module entry point
    main()
