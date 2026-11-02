"""Static inventory of LLM call sites.

The scanner reads Python source with :mod:`ast` and never imports or executes the
modules it inspects: the legacy tools read credentials at import time, so importing
them would both need keys and hide the very wiring the inventory is meant to show.

Detection is deliberately heuristic. It covers the five "before" tools in
``PLAN.md`` §4 and the LangChain "after" tools, and reports ``unknown`` rather than
guessing when it cannot tell a provider.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

PROVIDER_OPENAI: Final = "openai-compatible"
PROVIDER_ANTHROPIC: Final = "anthropic"
PROVIDER_GEMINI: Final = "gemini"
PROVIDER_LANGCHAIN: Final = "langchain"
PROVIDER_UNKNOWN: Final = "unknown"

STYLE_RAW_HTTP: Final = "raw HTTP"
STYLE_SDK: Final = "vendor SDK"
STYLE_LANGCHAIN: Final = "LangChain"
STYLE_TOOL_LOOP: Final = "hand-written tool loop"

STRUCT_NONE: Final = "none"
STRUCT_REGEX: Final = "regex"
STRUCT_SUBSTRING: Final = "substring"
STRUCT_JSON: Final = "JSON mode"
STRUCT_LANGCHAIN: Final = "LangChain structured"

_LLM_URL_MARKERS: Final[tuple[tuple[str, str], ...]] = (
    ("api.anthropic.com", PROVIDER_ANTHROPIC),
    ("/v1/messages", PROVIDER_ANTHROPIC),
    ("generativelanguage", PROVIDER_GEMINI),
    (":generatecontent", PROVIDER_GEMINI),
    ("/v1beta/models/", PROVIDER_GEMINI),
    ("/chat/completions", PROVIDER_OPENAI),
    ("/v1/responses", PROVIDER_OPENAI),
    ("api.openai.com", PROVIDER_OPENAI),
)

_HTTP_METHODS: Final = frozenset(
    {"get", "post", "put", "patch", "delete", "request", "stream", "send"}
)
_HTTP_RECEIVERS: Final = frozenset(
    {
        "client",
        "http",
        "http_client",
        "httpx_client",
        "session",
        "async_client",
        "async_http_client",
    }
)
_INVOKE_METHODS: Final = frozenset({"invoke", "ainvoke", "stream", "astream", "batch", "abatch"})
_LANGCHAIN_RECEIVER_HINTS: Final = (
    "model",
    "chain",
    "llm",
    "chat",
    "structured",
    "runnable",
    "agent",
    "graph",
)
_GENERIC_SDK_METHODS: Final = frozenset(
    {"complete", "generate", "chat", "predict", "generate_content"}
)
_PAYLOAD_MARKERS: Final = frozenset(
    {"messages", "chat/completions", "response_format", "tool_calls", "prompt", "model"}
)
_CREDENTIAL_MARKERS: Final = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CRED", "AUTH")
_NON_PROMPT_PARAMETERS: Final = frozenset({"model", "client", "llm", "chain", "connection", "conn"})
_PROMPT_KEYWORDS: Final = frozenset(
    {"json", "data", "content", "body", "messages", "contents", "prompt", "input", "system"}
)


@dataclass(frozen=True)
class CallSite:
    """One place where a module talks to a model.

    The first four fields are the frozen ``docs/INTERFACES.md`` contract; the rest carry
    the per-pattern detail ``plan/tasks/T06-inventory.md`` asks for.
    """

    path: str
    line: int
    tool: str
    provider: str
    call_style: str = "unknown"
    model: str | None = None
    model_source: str | None = None
    credentials: tuple[str, ...] = ()
    structured_output: str = STRUCT_NONE
    inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Inventory:
    """A deterministic inventory of the call sites found under one root."""

    sites: list[CallSite]
    summary: dict[str, int]
    root: str = ""


@dataclass(frozen=True)
class _Imports:
    modules: tuple[str, ...]
    aliases: dict[str, str]


@dataclass(frozen=True)
class _Parsing:
    """How this module turns a model answer into a value, looked at module-wide.

    The legacy tools parse in a separate function from the call (L2's ``run`` calls
    ``extract_json``, L3's ``run`` calls ``parse_scope``), so the strategy is a module
    property rather than a property of the call expression.
    """

    regex: bool = False
    substring: bool = False
    structured: bool = False


def scan(root: Path) -> Inventory:
    """Scan a file or directory tree without importing anything it reads."""
    root = root.expanduser()
    sites: list[CallSite] = []
    modules = 0
    for path in _python_files(root):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        modules += 1
        sites.extend(_scan_module(path, tree, root))
    sites.sort(key=lambda site: (site.path, site.line, site.tool, site.provider, site.call_style))
    counts: Counter[str] = Counter()
    for site in sites:
        counts[f"provider:{site.provider}"] += 1
        counts[f"call_style:{site.call_style}"] += 1
        counts[f"structured_output:{site.structured_output}"] += 1
    summary: dict[str, int] = {"modules": modules, "sites": len(sites)}
    summary.update(sorted(counts.items()))
    return Inventory(sites=sites, summary=summary, root=_display_path(root, root))


def write_json(inventory: Inventory, path: Path) -> None:
    """Write the inventory as stable, indented JSON."""
    payload = {
        "root": inventory.root,
        "summary": inventory.summary,
        "sites": [asdict(site) for site in inventory.sites],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(inventory: Inventory, path: Path, *, notes: Sequence[str] = ()) -> None:
    """Write the inventory as a readable Markdown report.

    ``notes`` is for a blind spot the scan itself cannot see; it is printed after the
    table so the report never reads as a complete call count when it is a lower bound.
    """
    lines = [
        "# LLM call-site inventory",
        "",
        f"Root: `{inventory.root}`",
        "",
        f"Modules scanned: **{inventory.summary.get('modules', 0)}**",
        "",
        f"Call sites found: **{inventory.summary.get('sites', 0)}**",
        "",
        "## Summary",
        "",
    ]
    if inventory.sites:
        lines += ["| Dimension | Value | Count |", "| --- | --- | ---: |"]
        for key, value in inventory.summary.items():
            if key in {"modules", "sites"}:
                continue
            dimension, _, label = key.partition(":")
            lines.append(f"| {dimension} | {_cell(label)} | {value} |")
        lines += ["", "## Call sites", ""]
        lines += [
            "| File:line | Tool | Provider | Call style | Model | Structured output | Inputs | Credentials |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for site in inventory.sites:
            model = _site_model(site)
            inputs = ", ".join(site.inputs) if site.inputs else "—"
            credentials = ", ".join(site.credentials) if site.credentials else "—"
            lines.append(
                f"| `{_cell(site.path)}:{site.line}` | {_cell(site.tool)} | "
                f"{_cell(site.provider)} | {_cell(site.call_style)} | {_cell(model)} | "
                f"{_cell(site.structured_output)} | {_cell(inputs)} | {_cell(credentials)} |"
            )
    else:
        lines.append("No LLM call sites found.")
    if notes:
        lines += ["", "## Notes"]
        lines += [""]
        for note in notes:
            lines += [note, ""]
        if lines[-1] == "":
            lines.pop()
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _site_model(site: CallSite) -> str:
    if site.model is None:
        return "—"
    if site.model_source == "env":
        return f"env:{site.model}"
    return site.model


def _cell(value: str) -> str:
    return value.replace("|", "\\|")


def _python_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    files = [
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and ".venv" not in path.parts
    ]
    return sorted(files)


def _display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        pass
    if root.is_dir():
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def _scan_module(path: Path, tree: ast.Module, root: Path) -> list[CallSite]:
    display = _display_path(path, root)
    imports = _imports(tree)
    all_nodes = list(ast.walk(tree))
    parsing = _Parsing(
        regex=_uses_regex(all_nodes),
        substring=_uses_substring(all_nodes),
        structured=_uses_structured_output(all_nodes),
    )
    module_nodes = list(_walk_own(tree))
    module_consts = _string_assignments(module_nodes)
    module_env = tuple(_env_reads(module_nodes))
    sites: list[CallSite] = []
    for scope in _scopes(tree):
        nodes = list(_walk_own(scope))
        sites.extend(
            _scan_scope(scope, nodes, display, module_consts, module_env, imports, parsing)
        )
    return sites


def _scopes(tree: ast.Module) -> Iterator[ast.Module | ast.FunctionDef | ast.AsyncFunctionDef]:
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _walk_own(node: ast.AST) -> Iterator[ast.AST]:
    """Yield descendants, stopping at nested scopes (scanned separately)."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield child
        yield from _walk_own(child)


def _scan_scope(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
    nodes: Sequence[ast.AST],
    display: str,
    module_consts: dict[str, str],
    module_env: Sequence[str],
    imports: _Imports,
    parsing: _Parsing,
) -> list[CallSite]:
    consts = {**module_consts, **_string_assignments(nodes)}
    env_names = tuple(sorted(set(module_env) | set(_env_reads(nodes))))
    credentials = _credential_names(env_names)
    params = (
        _parameters(scope) if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) else ()
    )
    assignments = _assignments(nodes)
    tool_loop = _is_tool_loop(nodes)
    tool = (
        scope.name
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
        else Path(display).stem
    )
    sites: list[CallSite] = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        classified = _classify(node, consts, imports)
        if classified is None:
            continue
        provider, style = classified
        if tool_loop:
            style = STYLE_TOOL_LOOP
        model, model_source = _find_model(node, consts, assignments)
        sites.append(
            CallSite(
                path=display,
                line=node.lineno,
                tool=tool,
                provider=provider,
                call_style=style,
                model=model,
                model_source=model_source,
                credentials=credentials,
                structured_output=_structured_output(node, parsing, style),
                inputs=_inputs(node, params, assignments),
            )
        )
    return sites


def _classify(call: ast.Call, consts: dict[str, str], imports: _Imports) -> tuple[str, str] | None:
    name = _dotted(call.func)
    if not name:
        return None
    last = name.rsplit(".", 1)[-1]

    if _is_langchain_call(name, last):
        return PROVIDER_LANGCHAIN, STYLE_LANGCHAIN

    if last in {"create", "stream"} and ".messages." in f".{name}":
        return PROVIDER_ANTHROPIC, STYLE_SDK
    if last == "create" and (".chat.completions." in f".{name}" or ".responses." in f".{name}"):
        return PROVIDER_OPENAI, STYLE_SDK
    if last in {"generate_content", "generate_content_async"} and ".models." in f".{name}":
        return PROVIDER_GEMINI, STYLE_SDK

    strings = _string_values(call, consts)
    urls = [value for value in strings if _looks_like_llm_url(value)]
    if _is_http_call(name, imports) and (urls or _has_llm_payload(strings)):
        provider = _provider_from_urls(urls) or _provider_from_imports(imports) or PROVIDER_UNKNOWN
        return provider, STYLE_RAW_HTTP

    if last in _GENERIC_SDK_METHODS and _has_model_argument(call, consts):
        provider = _provider_from_imports(imports) or PROVIDER_UNKNOWN
        return provider, STYLE_SDK
    return None


def _has_model_argument(call: ast.Call, consts: dict[str, str]) -> bool:
    return any(
        keyword.arg in {"model", "model_id", "model_name"} for keyword in call.keywords
    ) or bool(_find_model(call, consts, {})[0])


def _is_langchain_call(name: str, last: str) -> bool:
    if last not in _INVOKE_METHODS:
        return False
    receiver = name.rsplit(".", 1)[0].lower()
    return any(hint in receiver for hint in _LANGCHAIN_RECEIVER_HINTS)


def _is_http_call(name: str, imports: _Imports) -> bool:
    resolved = _resolve_dotted(name, imports)
    last = resolved.rsplit(".", 1)[-1]
    if last not in _HTTP_METHODS:
        return False
    if resolved.split(".", 1)[0] in {"httpx", "requests"}:
        return True
    receiver = name.rsplit(".", 1)[0].rsplit(".", 1)[-1].lower()
    return receiver in _HTTP_RECEIVERS


def _looks_like_llm_url(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker, _ in _LLM_URL_MARKERS)


def _provider_from_urls(urls: Iterable[str]) -> str | None:
    joined = "\n".join(url.lower() for url in urls)
    for marker, provider in _LLM_URL_MARKERS:
        if marker in joined:
            return provider
    return None


def _provider_from_imports(imports: _Imports) -> str | None:
    for module in imports.modules:
        if module == "anthropic" or module.startswith("anthropic."):
            return PROVIDER_ANTHROPIC
        if module == "openai" or module.startswith("openai."):
            return PROVIDER_OPENAI
        if module == "google.genai" or module.startswith("google.genai."):
            return PROVIDER_GEMINI
        if module == "langchain" or module.startswith(("langchain.", "langchain_")):
            return PROVIDER_LANGCHAIN
    return None


def _has_llm_payload(strings: Iterable[str]) -> bool:
    return any(value.lower() in _PAYLOAD_MARKERS for value in strings)


def _find_model(
    call: ast.Call, consts: dict[str, str], assignments: dict[str, list[ast.AST]]
) -> tuple[str | None, str | None]:
    for keyword in call.keywords:
        if keyword.arg in {"model", "model_id", "model_name", "deployment_name"}:
            resolved = _resolve_model(keyword.value, consts)
            if resolved[0] is not None:
                return resolved
    for node in _payload_nodes(call, assignments):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=False):
                if isinstance(key, ast.Constant) and key.value in {"model", "model_id"}:
                    resolved = _resolve_model(value, consts)
                    if resolved[0] is not None:
                        return resolved
    return None, None


def _payload_nodes(call: ast.Call, assignments: dict[str, list[ast.AST]]) -> list[ast.AST]:
    """The call plus the local values assigned to names it mentions (one hop deep)."""
    nodes = list(ast.walk(call))
    for node in list(nodes):
        if isinstance(node, ast.Name):
            nodes.extend(assignments.get(node.id, []))
    return nodes


def _resolve_model(node: ast.AST, consts: dict[str, str]) -> tuple[str | None, str | None]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, "literal"
    if isinstance(node, ast.BoolOp):
        # `model or env("ROUTER_MODEL", "router-small")`: report the env name, not the
        # fallback literal, because that is what supplies the model in a real run.
        for value in node.values:
            resolved = _resolve_model(value, consts)
            if resolved[0] is not None:
                return resolved
        return None, None
    if isinstance(node, ast.Name):
        if node.id in consts:
            return consts[node.id], "literal"
        return None, None
    source = _env_reference(node)
    if source is not None:
        return source, "env"
    return None, None


def _env_reference(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        name = _dotted(node.func)
        last = name.rsplit(".", 1)[-1]
        if (
            name in {"os.getenv", "getenv", "os.environ.get", "environ.get"} or last == "env"
        ) and node.args:
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                return argument.value
    if isinstance(node, ast.Subscript):
        name = _dotted(node.value)
        if name in {"os.environ", "environ"}:
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                return key.value
    return None


def _env_reads(nodes: Iterable[ast.AST]) -> list[str]:
    names: set[str] = set()
    for node in nodes:
        name = _env_reference(node)
        if name is not None:
            names.add(name)
    return sorted(names)


def _credential_names(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        name for name in names if any(marker in name.upper() for marker in _CREDENTIAL_MARKERS)
    )


def _structured_output(call: ast.Call, parsing: _Parsing, style: str) -> str:
    if parsing.structured:
        return STRUCT_LANGCHAIN
    if _uses_json_mode(call):
        return STRUCT_JSON
    if style == STYLE_TOOL_LOOP:
        return STRUCT_NONE
    if parsing.regex:
        return STRUCT_REGEX
    if parsing.substring:
        return STRUCT_SUBSTRING
    return STRUCT_NONE


def _uses_structured_output(nodes: Iterable[ast.AST]) -> bool:
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        if _dotted(node.func).rsplit(".", 1)[-1] in {
            "with_structured_output",
            "make_structured_model",
        }:
            return True
    return False


def _uses_json_mode(call: ast.Call) -> bool:
    for node in ast.walk(call):
        if (
            isinstance(node, ast.keyword)
            and node.arg == "response_format"
            and _contains_json(node.value)
        ):
            return True
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=False):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "response_format"
                    and _contains_json(value)
                ):
                    return True
    return False


def _contains_json(node: ast.AST) -> bool:
    return any(
        isinstance(sub, ast.Constant) and isinstance(sub.value, str) and "json" in sub.value.lower()
        for sub in ast.walk(node)
    )


def _uses_regex(nodes: Iterable[ast.AST]) -> bool:
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        last = name.rsplit(".", 1)[-1]
        if last in {"compile", "sub", "search", "match", "fullmatch", "findall", "finditer"}:
            receiver = name.rsplit(".", 1)[0].lower()
            if receiver == "re" or receiver.endswith((".re", "pattern", "regex", "compiled")):
                return True
    return False


def _uses_substring(nodes: Iterable[ast.AST]) -> bool:
    """A literal token tested against the answer (``"yes" in text``, ``startswith``).

    Equality checks are deliberately excluded: ``finish_reason != "stop"`` is control
    flow, not answer parsing.
    """
    for node in nodes:
        if isinstance(node, ast.Compare):
            literals = [node.left, *node.comparators]
            if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops) and any(
                _is_string_constant(operand) for operand in literals
            ):
                return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"startswith", "endswith", "find", "index"}
            and node.args
            and _is_string_constant(node.args[0])
        ):
            return True
    return False


def _is_string_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and bool(node.value)


def _is_tool_loop(nodes: Sequence[ast.AST]) -> bool:
    has_loop = any(isinstance(node, (ast.While, ast.For, ast.AsyncFor)) for node in nodes)
    if not has_loop:
        return False
    return any(
        (isinstance(node, ast.Name) and node.id == "tool_calls")
        or (isinstance(node, ast.Attribute) and node.attr == "tool_calls")
        or (isinstance(node, ast.Constant) and node.value == "tool_calls")
        for node in nodes
    )


def _inputs(
    call: ast.Call, params: Sequence[str], assignments: dict[str, list[ast.AST]]
) -> tuple[str, ...]:
    seen: set[str] = set()
    frontier = [
        node.id
        for root in _prompt_roots(call)
        for node in ast.walk(root)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    ]
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        for value in assignments.get(name, []):
            frontier.extend(
                node.id
                for node in ast.walk(value)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            )
    return tuple(param for param in params if param in seen and param not in _NON_PROMPT_PARAMETERS)


def _prompt_roots(call: ast.Call) -> list[ast.AST]:
    """The call arguments that carry prompt content, not transport/configuration."""
    roots: list[ast.AST] = [
        keyword.value for keyword in call.keywords if keyword.arg in _PROMPT_KEYWORDS
    ]
    if roots:
        return roots
    fallback: list[ast.AST] = [*call.args, *(keyword.value for keyword in call.keywords)]
    return fallback


def _parameters(scope: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    arguments = scope.args
    names = [arg.arg for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)]
    if arguments.vararg is not None:
        names.append(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.append(arguments.kwarg.arg)
    return tuple(name for name in names if name not in {"self", "cls"})


def _string_assignments(nodes: Iterable[ast.AST]) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in nodes:
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        result[target.id] = node.value.value
        elif isinstance(node, ast.AnnAssign) and (
            isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            result[node.target.id] = node.value.value
    return result


def _assignments(nodes: Iterable[ast.AST]) -> dict[str, list[ast.AST]]:
    result: dict[str, list[ast.AST]] = {}
    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result.setdefault(target.id, []).append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            result.setdefault(node.target.id, []).append(node.value)
    return result


def _imports(tree: ast.Module) -> _Imports:
    modules: list[str] = []
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
                aliases[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not module:
                continue
            modules.append(module)
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
    return _Imports(modules=tuple(sorted(set(modules))), aliases=aliases)


def _resolve_dotted(name: str, imports: _Imports) -> str:
    head, _, tail = name.partition(".")
    resolved = imports.aliases.get(head)
    if resolved is None:
        return name
    return f"{resolved}.{tail}" if tail else resolved


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    return ""


def _string_values(node: ast.AST, consts: dict[str, str]) -> set[str]:
    values: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            values.add(sub.value)
        elif isinstance(sub, ast.Name) and sub.id in consts:
            values.add(consts[sub.id])
    return values
