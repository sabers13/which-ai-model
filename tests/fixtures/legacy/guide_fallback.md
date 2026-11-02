# Fallback guide excerpt (test fixture)

This is a short, committed excerpt used by the legacy L4 docs helper when
The real guide in `docs/guide/` is the pinned retrieval corpus.

## Connecting a provider

Connections live in `config/providers.toml`. Each connection names a provider, a
protocol, an optional base URL and a credential reference of the form `env:NAME`. The
key itself lives only in `.env`, which is gitignored and never read by the tools.

## Adding a tool

A tool is a small module with a `run`-style entry point and a test that mocks its
transport. Register the module with its CLI so `wam` discovers it.

## Replay and tests

`make test` runs with INET sockets blocked. `make all` rebuilds results in replay mode
from committed recordings, so no model call and no network are needed.
