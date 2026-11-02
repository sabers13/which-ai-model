# Tracing and the local trace file

Tracing is always on. Every model call appends exactly one row to a local
JSONL trace file, whether the call succeeded or failed, so the full cost
and quality evidence for a run can be rebuilt from files alone. The
optional hosted tracing backend is enabled additionally only when its key
is set; the local file is the committed evidence and needs no network.
Live traces are written under `traces/live/`, which is gitignored, while
replay traces land beside the run they belong to for later scoring. The
sections below list every field in a row, then show the callback that
collects the fields without ever touching secrets. Read this page with
the record-replay page: traces say what happened, recordings keep the raw
bytes, and the key in the row joins the two.

## What one trace row contains

One row is one `CallRecord`: run id, task name, tool name, item id, model
reference (connection id plus model id), prompt version, wall-clock
latency in milliseconds, input and output token counts, dollar cost,
whether the output validated against its schema, the error class on
failure, and the recording key locating the raw response. Token counts
come from the message `usage_metadata`, where thinking tokens appear
under output details when reported. Cost prefers the gateway's measured
per-response value, else tokens times the prices table, else `None` shown
as unknown — a cost is never guessed. The writer is deliberately small:

```python
from pathlib import Path
from which_ai_model.core import TraceWriter

writer = TraceWriter(Path("traces/run-2026-10-28.jsonl"))
writer.write(record)
writer.close()
```

One writer per run file is the norm; opening a second writer on the same
path from another process interleaves rows, so give each run its own file.

## Collecting the fields with callbacks

Fields are gathered by a callback handler passed via
`config={"callbacks": [...]}`. On model end it reads the first
generation's message, pulls `usage_metadata` for token counts, the finish
reason from response metadata, and the model name from the result output,
then appends one scrubbed JSON row. Key material, headers, and
credential-bearing URLs must never reach the row; it carries counts, the
finish reason, and identifiers only. Rows are the JSON form of the pydantic
contract, so readers validate each line against `CallRecord` and reject a
corrupt file loudly instead of averaging over garbage silently.

Schema validity is decided beside the callback: structured calls use
`include_raw=True`, and the parse is valid exactly when the returned
`parsing_error` is `None`. Failed calls are written too, with the error
field set and the schema flag empty — a missing row would look like a call
that never happened, which is worse than an honest failure. When only
totals are needed, the standard `UsageMetadataCallbackHandler` aggregates
input, output, and total tokens across the calls it observes.
