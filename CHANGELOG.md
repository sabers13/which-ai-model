# Changelog

All notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-11-01

Your own task, a re-check for a changed model, and a second small paid run.

### Added

- **Task specs.** A benchmark task is one TOML file in `tasks/` plus a labelled data
  file; no Python. `wam task new <id>` writes a commented skeleton, `wam task check
  <spec>` validates ids, labels, files and prompt placeholders without a key. A spec
  can allow an `unclear` answer and score it as coverage plus accuracy on answered items,
  while counting it as wrong in the primary metric. Spec tasks run beside the four
  built-in v1 tasks through the same runner, scorer and gate.
- **T-DEFECT**, the first spec task: sort public NHTSA electrical complaints into eight
  component groups and flag the safety-relevant ones. Its 600-item sample, its TF-IDF and
  keyword baselines, its prompt and its card are committed; the NHTSA snapshot and
  licence are recorded in `THIRD_PARTY.md` and `data/defect/STATS.md`.
- **Per-task gates.** A spec sets its own primary-drop threshold, schema-valid floor and
  metric floors (T-DEFECT: safety recall at least 0.90). A model below a floor is never
  recommended. Run manifests record temperature, reasoning and maximum output tokens;
  the gate refuses runs whose settings differ unless `--accept-settings-mismatch` is
  given, and then returns its normal verdict plus a warning per difference.
- **Re-checking.** `wam recheck run` records fresh answers under `recordings/fresh/`,
  never reads a saved answer, scores them and gates them against an earlier run. `wam
  bench --fresh` uses the same fresh root without the gate.
- **Use-case cards.** `results/cards/<task>.{md,json}` for the four v1 tasks and
  T-DEFECT: the job, the data, a per-model table with intervals and cost, the baseline,
  the recommendation and its rule, the gate rule, when to re-check, cost at volume and
  the known limits.
- **Cost at volume.** Monthly cost at 1,000 / 10,000 / 100,000 items, linear from the
  measured cost per 1,000 items; an unknown cost stays unknown.
- **Prompt-library export.** `results/prompt-library/<task>.{md,json}`: the prompt and
  its version, the output schema, a recommended and a fallback model, guardrails and the
  re-check trigger.
- **Run controls.** `wam bench --cap-usd` sets a hard cap for one run, `--no-peak-wait`
  skips the v1 off-peak schedule, and the reduction path cuts a cell as a stratified
  subset over its primary label rather than taking a prefix.

### Changed

- `make all` also rebuilds the cards and the prompt library from committed files, and
  `make release-check` still runs lint, strict mypy, the offline tests, both scans and the
  facts check.
- New make targets: `task-check`, `cards`, `prompt-library`, `recheck-demo` and
  `bench-live-v11`.
- The v1.1 run reached the same seven models through the same gateway, made 4,000 calls
  and spent USD 0.8937 of its USD 5.00 cap. Gemini 4 Argon was scored on 400 of 600 items
  by the sample-reduction procedure; the card says so.

### Unchanged

- The four built-in tasks, their prompts (`v1`), samples, recordings, results and gate
  thresholds. Every v1 file rebuilds byte for byte except two shared files whose
  manifest lines changed because v1.1 data was appended: `results/recommendations.json`
  gained the `defect` entry and `results/page/page-data.json` gained the v1.1 section.
  Every v1 entry in both files is unchanged.
- `docs/guide/` stays frozen: it is the T-DOCS retrieval corpus, and editing it would
  change the committed recordings.

### Known limits

- T-DEFECT is a dated snapshot of public NHTSA complaints (2026-10-31), not a live feed
  and not a safety determination. The sample oversamples safety-relevant complaints
  (20.0% against a 10.7% natural rate in its pool), so its safety precision and recall
  are not natural rates.
- Gemini 4 Argon's T-DEFECT cell covers 400 of 600 items after the budget cut; compare
  its wider interval before treating a gap as real.
- Claude Haiku 5.5 runs at its provider default temperature because its API rejects the
  parameter; the manifest records that per model.

## [1.0.0] — 2026-10-29


First complete release: the benchmark, the shared layer and the published evidence.

### Added

- **BYOK connection layer.** Four wire protocols (Chat Completions, Responses,
  Anthropic Messages, Gemini), presets for the benchmark gateway and the major
  providers, `wam doctor` with a fixed probe vocabulary and tri-state model discovery,
  and a keyless local OpenAI-compatible preset.
- **Shared model layer.** Provider-agnostic chat models built from a connection,
  pydantic structured output through the `function_calling` strategy, fallback and retry
  on transport errors, always-on local JSONL tracing, token and cost accounting, pacing
  limits, and deterministic record/replay.
- **Five "before" tools and five migrated "after" tools.** A ticket router, reply
  drafter, scope checker, docs helper and tool-calling agent; the legacy versions are
  deliberately inconsistent, the migrated versions share one layer, and each keeps a
  parity check on the committed fixtures. Four legacy tools are staged reconstructions
  and labelled as such; the agent is adapted from real earlier code (see
  `THIRD_PARTY.md`).
- **Benchmark.** Seeded samples from BANKING77 and CLINC150, deterministic metrics with
  confidence intervals, a TF-IDF baseline, a runner that records and replays every call,
  and a per-task picker (`cheapest`, `most accurate`, `fastest` within a fixed
  good-enough band).
- **Regression gate.** PASS, FAIL or REFUSED, with a demonstration on the live run and a
  refusal demonstration for incomparable runs.
- **RAG documentation assistant.** A guide corpus, CPU embeddings, a Postgres + pgvector
  store, an in-memory twin for replay, and a LangGraph retrieve → answer → cite flow.
- **Inventory scanner.** A static AST scanner that reports LLM call sites without
  importing the code it reads.
- **Live run evidence.** Seven models on four tasks, 7,742 calls, committed recordings
  and results with a checksum manifest; `make all` rebuilds them offline and
  byte-identically.
- **Release documentation.** `README.md`, `METHOD.md`, `THIRD_PARTY.md`,
  `project-facts.yml`, this changelog, and a `make release-check` gate that includes a
  facts check tying every published number to the results file that proves it.

### Known limits

- All seven models were reached through one gateway; direct provider endpoints are
  mock-tested only.
- Measured per-response cost is unavailable on the Responses and Gemini paths, so three
  models use the committed price table as an upper bound.
- The benchmark is a dated snapshot of public data, not a live leaderboard.
- The recorded run used each provider's sampling default: the model layer pinned no
  temperature at run time. It now passes 0 on every protocol, but re-recording the
  7,742 calls would need new paid calls, so the recorded evidence stands as run. See
  `METHOD.md`.

[1.1.0]: https://github.com/sabers13/which-ai-model/releases/tag/v1.1.0
[1.0.0]: https://github.com/sabers13/which-ai-model/releases/tag/v1.0.0
