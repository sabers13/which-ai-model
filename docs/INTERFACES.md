# API reference

Public types, functions and package dependency boundaries.

## 0. Package dependencies

```
contracts  <-  connect  <-  core  <-  tools / bench / rag / gate / report
                  ^          ^
                  |          |
                legacy       |
             (no core)   inventory (stdlib/ast only)
```

- `which_ai_model.contracts` imports nothing from this project. Its public shape is versioned.
- `legacy/` is deliberately **pre-LangChain**: it may import the provider SDKs
  (`httpx`, `anthropic`, `google-genai`, `openai`) and `contracts`, but never `core/`,
  `tools/` or `bench/`. That is what makes the "before" measurements honest.
- `core/` is the only place that constructs provider chat models, reads a
  credential, or writes traces/recordings.
- `tools/`, `bench/`, `rag/`, `gate/`, `report/` may import `contracts` and `core`.
  `tools/` may import `legacy/` only in tests, for parity comparison.
- `inventory/` is standalone: standard library (`ast`, `pathlib`) and `contracts`.
- No module except `connect/credentials.py` and `bench/run.py` (live mode) may touch
  `os.environ` for a secret. Environment files are read by the live benchmark runner.
- Tests never open an INET socket. `tests/conftest.py` enforces this with
  `pytest-socket`, allowing unix sockets only.

## 2. `which_ai_model.connect`

```python
from pathlib import Path
from collections.abc import Sequence

from which_ai_model.contracts import (
    Connection,
    DiscoveredModel,
    ProbeReason,
    ProbeResult,
    TriState,
)


def load_connections(path: Path = Path("config/providers.toml")) -> list[Connection]: ...
def get_connection(connection_id: str, connections: Sequence[Connection]) -> Connection: ...
def resolve_credential(connection: Connection) -> str | None:
    """Return the secret for credential_ref, or None for keyless connections."""


def probe_connection(connection: Connection, *, timeout_s: float = 20.0) -> ProbeResult: ...
def discover_models(
    connection: Connection, *, timeout_s: float = 20.0
) -> list[DiscoveredModel]: ...
def doctor(connections: Sequence[Connection]) -> list[ProbeResult]: ...
def format_doctor(results: Sequence[ProbeResult]) -> str: ...
```

Rules: outcomes use only `ProbeReason`; discovery is tri-state and `unknown` never
becomes `no`; `detail` is scrubbed; `doctor` also reports installed subscription CLIs
(`claude`, `codex`, `opencode`, `gemini`, `agy`) as information only and never calls
them. Keyless presets (`local-openai-compatible`) resolve to `None` and are not a
credential failure. Presets other than `opencode-go` are probed only with mocked
transports in tests.

`wam doctor [--json]` prints one row per connection: id, protocol, reason, latency.

## 3. `which_ai_model.core`

```python
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from which_ai_model.contracts import (
    CallRecord,
    Connection,
    ModelRef,
    TaskName,
)
from which_ai_model.core import Price

T = TypeVar("T", bound=BaseModel)


def make_chat_model(
    connection: Connection,
    model_id: str,
    *,
    reasoning: str = "minimal",
    max_tokens: int = 1024,
    session_id: str | None = None,
) -> BaseChatModel: ...


def make_structured_model(
    model: BaseChatModel,
    schema: type[T],
    *,
    method: str = "function_calling",
) -> Runnable[Any, dict[str, Any]]: ...


def get_model(
    ref: ModelRef,
    *,
    fallback: ModelRef | None = None,
) -> BaseChatModel:
    """A runnable with five transport retries and an optional fallback model."""


def structured(
    model: BaseChatModel,
    schema: type[T],
    request: Any = "",
    *,
    method: str = "function_calling",
) -> tuple[T | None, bool, str | None]:
    """``(parsed, schema_valid, raw_content)``; free-text JSON is never parsed."""


def make_agent(
    primary: Connection,
    fallback: Connection | None,
    model_id: str,
    tools: Sequence[Any],
    *,
    system_prompt: str,
    response_format: Any,
) -> Any: ...


class TraceWriter:
    def __init__(self, path: Path) -> None: ...
    def write(self, record: CallRecord) -> None: ...
    def close(self) -> None: ...


def request_digest(messages: Sequence[Any]) -> str: ...
def recording_key(model_id: str, prompt_version: str, digest: str) -> str: ...


class Recorder:
    def record(
        self, key: str, model_id: str, prompt_version: str, digest: str, response: dict[str, Any]
    ) -> None: ...
    def lookup(self, key: str) -> dict[str, Any] | None: ...


def cost_usd(
    model_id: str,
    input_tokens: int | None,
    output_tokens: int | None,
    prices: Mapping[str, Price] | Path | str | None = None,
) -> float | None: ...
def usage_cost_usd(usage: dict[str, Any] | None, model: ModelRef) -> float | None: ...
```

`make_chat_model` implements §2–§7 of `docs/LANGCHAIN_NOTES.md`: protocol → class,
custom `base_url`, `default_headers`/`additional_headers`, lowest reasoning level,
`max_retries=0` (retries are middleware), and the session header. `make_agent` wires
`ModelFallbackMiddleware` + `ModelRetryMiddleware` and disables structured-output
self-repair. `TraceWriter` writes one file per run: `traces/replay/<run_id>.jsonl`
for replay runs and `traces/live/<run_id>.jsonl` for live/record runs (both
gitignored).

`cost_usd` in this implementation prices tokens: `cost_usd(model_id, input_tokens,
output_tokens, prices)` returns `None` for an unknown model or currency.
`usage_cost_usd(usage, model)` is the measured-first wrapper: the gateway's
`usage.opencode.list_cost` when present, else the price table, else `None`.

### Replay example — "call any model with a schema", offline

This block runs as written from the repository root against the committed fixture
recording; it opens no socket. `tests/core/test_interfaces_example.py` executes exactly
this block, so the documentation cannot drift.

```python
from pathlib import Path

from which_ai_model.contracts import RouteOut
from which_ai_model.core import Recorder, recording_key, replay_model, request_digest, structured

messages = [
    {"role": "system", "content": "Route the customer message to one Banking77 intent."},
    {"role": "user", "content": "My card was swallowed by the ATM."},
]
key = recording_key("deepseek-v4.1-flash", "route-v1", request_digest(messages))
recorder = Recorder(Path("tests/core/fixtures/recordings"), mode="replay")
model = replay_model(recorder, key, "deepseek-v4.1-flash")
parsed, schema_valid, raw_content = structured(model, RouteOut, messages)

assert schema_valid and isinstance(parsed, RouteOut)
assert parsed.intent == "card_not_working"
assert raw_content == ""
```


## 4. `which_ai_model.legacy`

```python
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LegacyResult:
    content: str
    tool_calls: list[dict[str, Any]]
    latency_ms: int
    error: str | None


def route_message(message: str) -> LegacyResult: ...  # L1, raw httpx
def draft_reply(message: str) -> LegacyResult: ...  # L2, Anthropic SDK + regex
def check_scope(message: str) -> LegacyResult: ...  # L3, google-genai SDK
def docs_answer(question: str) -> LegacyResult: ...  # L4, whole guide in the prompt
def run_agent(question: str) -> LegacyResult: ...  # L5, hand-written tool loop
```

These are the "before" tools. They read their own credentials through
`connect.resolve_credential` (or a plain env var only in a test double) and return the
raw content — no parsing beyond the deliberately brittle parsing described in
the documented migration baseline. Tests use `respx`/`httpx.MockTransport` or recorded fixtures; never a live
call.

## 5. data, samples, metrics, baseline

```python
from pathlib import Path
import pandas as pd


def fetch_data(dest: Path = Path("data/raw")) -> None: ...
def verify_checksums(checksums: Path = Path("data/CHECKSUMS.txt")) -> None: ...
def load_banking77(split: str = "test") -> pd.DataFrame: ...
def load_clinc150(split: str = "plus") -> pd.DataFrame: ...
def build_samples(seed: int = 20261028) -> dict[str, pd.DataFrame]: ...


def sample_route(df: pd.DataFrame, *, per_intent: int = 8, seed: int) -> pd.DataFrame: ...
def sample_scope(df: pd.DataFrame, *, per_class: int = 150, seed: int) -> pd.DataFrame: ...
def sample_reply(route_sample: pd.DataFrame, *, n: int = 150, seed: int) -> pd.DataFrame: ...


def accuracy(y_true: list[str], y_pred: list[str]) -> float: ...
def macro_f1(y_true: list[str], y_pred: list[str]) -> float: ...
def bootstrap_ci(
    values: list[float], *, n_resamples: int = 2000, seed: int
) -> tuple[float, float]: ...
def wilson_ci(successes: int, n: int) -> tuple[float, float]: ...


class Baseline:
    def fit(self, texts: list[str], labels: list[str]) -> "Baseline": ...
    def predict(self, texts: list[str]) -> list[str]: ...
```

`fetch_data` is a networked dataset download function; it is never called by
tests or by `make test`. `build_samples` writes id lists under `data/samples/` and
records the sample hash used in `RunManifest`.

## 6. RAG corpus and store

```python
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


def load_corpus(root: Path = Path("docs/guide")) -> list[Chunk]: ...
def chunk_documents(
    chunks: Sequence[Chunk], *, max_chars: int = 1200, overlap: int = 200
) -> list[Chunk]: ...
def embed_texts(texts: Sequence[str], *, batch_size: int = 32) -> list[list[float]]: ...


class PgVectorStore:
    def __init__(self, dsn: str, *, table: str = "wam_chunks") -> None: ...
    def ensure_schema(self, *, dimensions: int = 384) -> None: ...
    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...
    def search(self, embedding: Sequence[float], *, k: int = 4) -> list[SearchHit]: ...


def load_gold(path: Path = Path("data/rag/gold.jsonl")) -> list[dict]: ...
```

Embeddings are `BAAI/bge-small-en-v1.5` on CPU (384 dimensions). `ensure_schema` uses
pgvector's `CREATE EXTENSION IF NOT EXISTS vector` and a cosine index. Tests use a fake
embedder and never require Postgres unless `WAM_TEST_POSTGRES` is set.

## 7. inventory

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CallSite:
    path: str
    line: int
    tool: str
    provider: str


@dataclass(frozen=True)
class Inventory:
    sites: list[CallSite]
    summary: dict[str, int]


def scan(root: Path) -> Inventory: ...
def write_json(inventory: Inventory, path: Path) -> None: ...
def write_markdown(inventory: Inventory, path: Path) -> None: ...
```

The scanner is static (`ast`); it never imports the legacy modules (they would read
keys). `make inventory` writes `inventory.json` and `inventory.md` at the repo root.

## 8.  /  — the "after" tools

```python
from langchain_core.language_models import BaseChatModel
from which_ai_model.contracts import DocsOut, ReplyOut, RouteOut, ScopeOut


def route_message(message: str, model: BaseChatModel) -> RouteOut: ...
def draft_reply(message: str, model: BaseChatModel) -> ReplyOut: ...
def check_scope(message: str, model: BaseChatModel) -> ScopeOut: ...


def answer_docs(question: str, model: BaseChatModel) -> DocsOut: ...  #  owns docs.py
def run_agent(question: str, model: BaseChatModel) -> str: ...


def build_mcp_server() -> "MCPServer": ...  # mcp.server.mcpserver.MCPServer
```

L5/MCP rules from `docs/LANGCHAIN_NOTES.md` §10: the server is `mcp` 2.x's `MCPServer`,
run over stdio; the LangChain side is `langchain.mcp.MCPAdapter`; MCP tools are
async-only, so the agent path uses `ainvoke`. Parity tests (/) run in replay
mode with a fake model returning the same raw content as the legacy fixture.

## 9. runner and gate

```python
from pathlib import Path
from collections.abc import Sequence

from which_ai_model.contracts import CellResult, ModelRef, Prediction, RunManifest, TaskName


def run(mode: str = "replay", *, config: Path = Path("config")) -> RunManifest: ...
def score_task(
    task: TaskName, predictions: Sequence[Prediction]
) -> tuple[float, dict[str, float]]: ...
def evaluate_cell(predictions: Sequence[Prediction], *, ci: dict[str, float]) -> CellResult: ...
def estimate_run_cost(models: Sequence[ModelRef], tasks: Sequence[TaskName]) -> float: ...


class Budget:
    def __init__(self, cap_usd: float) -> None: ...
    def can_spend(self, estimate_usd: float) -> bool: ...
    def spend(self, usd: float) -> None: ...


def check_gate(baseline: Path, candidate: Path) -> tuple[bool, list[str]]: ...
```

`run(mode="replay")` is the mode `make all` uses; `run(mode="live")` is  only and
must refuse without a resolved key and a budget. The gate refuses incomparable runs
(different `sample_hash`, `prompt_version` or `scorer_version`) — a refusal, not a pass.

## 10. RAG graph

```python
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from which_ai_model.contracts import DocsOut
from which_ai_model.rag.store import PgVectorStore


def build_rag_graph(store: PgVectorStore, model: BaseChatModel) -> CompiledStateGraph: ...
def answer_question(question: str, *, store: PgVectorStore, model: BaseChatModel) -> DocsOut: ...
def evaluate_rag(
    questions: list[dict], *, store: PgVectorStore, model: BaseChatModel
) -> dict[str, float]: ...
```

The graph is retrieve → answer → cite; citations are passage ids from `Chunk.id`.
`abstained=True` must be explicit for the five unanswerable questions.

## 11. report

```python
from pathlib import Path
from typing import Any


def export_results(run: Path, out: Path = Path("results")) -> None: ...
def build_page_data(results: Path = Path("results")) -> dict[str, Any]: ...
def render_page(
    data: dict[str, Any], out: Path = Path("dist/which-ai-model/index.html")
) -> None: ...
```

`results/page/page-data.json` is the only input to the page. Numbers on the page come
only from `results/` (the published evidence policy).

## 12. Files produced by the pipeline

```
recordings/<task>/<model_slug>.jsonl   one recorded response per line, replay input
traces/replay/<run_id>.jsonl      replay traces (gitignored)
traces/live/<run_id>.jsonl        live/record traces (gitignored)
results/runs/<run_id>/         predictions.jsonl + cells.json + manifest.json
results/SUMMARY.md             every headline number with its file
results/page/page-data.json     input
dist/which-ai-model/index.html committed page build
```

### Recording row (exact)

```json
{"key": "<64 hex>", "model_id": "<id>", "prompt_version": "<version>",
 "request_digest": "<64 hex>",
 "response": {"content": "<text or null>", "tool_calls": [], "usage": {}},
 "recorded_at": "2026-10-28T12:00:00Z"}
```

- `key = sha256(f"{model_id}\n{prompt_version}\n{request_digest}")`.
- `request_digest = sha256(canonical JSON of the request messages)` — sorted keys, no
  whitespace.
- `response.usage` is the provider's usage object after scrubbing; it may carry the
  gateway's measured cost under a provider key. No headers, no keys, no base URLs with
  credentials.
- Replay reads by `key`; the last matching row wins. It never opens a socket.
- `make all` exports `WAM_MODE=replay` so rebuilds stay on the replay path.

### CallRecord ↔ trace row

`CallRecord` is the pydantic contract; the trace row is its JSON form
(`model_dump(mode="json")`). `CallRecord.model` is a `ModelRef`; `response_key` is the
recording `key`. The trace is written even when a call fails (`error` set,
`schema_valid=None`).
