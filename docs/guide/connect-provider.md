# Connect a provider with BYOK and `wam doctor`

Every model call in this project goes through a `Connection`: a small
record that says where to call, which wire protocol to speak, and which
environment variable holds the key. You bring your own key (BYOK). Keys
live only in the gitignored `.env` file at the repository root and are
referenced from configuration as `env:NAME` pointers, never pasted into
code, logs, traces, or recordings. The shared presets live in
`config/providers.toml`, and a `wam doctor` run tells you which of them
actually work on your machine before you spend anything. The rest of this
page describes the record field by field, then shows the one command that
verifies the whole setup end to end.

## How a connection is described

A connection has an id, a provider label, a protocol, a base URL, and a
credential reference. The protocol is one of four fixed values:
`OPENAI_COMPATIBLE` for Chat Completions endpoints, `OPENAI_RESPONSES` for
the Responses API, `ANTHROPIC` for the Messages API, and `GEMINI` for the
Google Generative Language API. The base URL never carries credentials; a
URL with an `@` before its first path slash is rejected by the contracts.
Keyless presets such as `local-openai-compatible` set no credential
reference at all and resolve to no secret, which is a normal setup rather
than a failure. In code you load the file and pick a connection by id:

```python
from pathlib import Path
from which_ai_model.connect import get_connection, load_connections

connections = load_connections(Path("config/providers.toml"))
connection = get_connection("opencode-go-openai", connections)
```

A missing id raises immediately rather than returning a half-built
default, so a typo in a script fails fast at startup instead of probing
the wrong endpoint halfway through a run.

## Presets and the credential rule

The gateway preset used for the benchmark is exposed as four connections
sharing one key, one per API format, beside the standard `openai`,
`anthropic`, `gemini`, `openrouter`, `local-openai-compatible`, and
`custom` presets. `resolve_credential` reads the named variable at call
time and returns the key value, or `None` for keyless connections. Only
the credentials module and the live benchmark runner may touch the
environment for a secret; every other module receives the value as an
argument and must never print it, log it, or store it in a trace.
Configuration files hold only the `env:NAME` pointer, which the secrets
scanner recognises as safe, so even a committed config stays clean. When
a connection misbehaves, the probe outcome tells you which of the two
sides — your configuration or the remote endpoint — to investigate.

## Checking what works with `wam doctor`

Run `wam doctor` before spending anything. It probes every configured
connection and prints one row per connection with its id, protocol,
outcome, and latency in milliseconds. The outcome vocabulary is fixed:
`OK`, `CONFIG_INVALID`, `CREDENTIAL_MISSING`, `CREDENTIAL_REJECTED`,
`ENDPOINT_UNREACHABLE`, `RATE_LIMITED`, `PROTOCOL_UNSUPPORTED`, or
`UNKNOWN`. Scripts and the runner switch on these values without guessing,
and the `detail` field is scrubbed of headers and key material first. A
`--json` flag gives the same fields for machines. Doctor also lists which
subscription CLIs are installed (`claude`, `codex`, `opencode`, `gemini`,
`agy`) as information only — they are never used for model calls, since
CLIs give no comparable latency or per-token cost. Model discovery is
tri-state (`yes`, `no`, `unknown`), and `unknown` never becomes `no`.
Make doctor the first step of any new checkout or key rotation: a green
row per connection is the cheapest possible proof the run can proceed.
