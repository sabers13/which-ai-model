# Method

This documents the recorded evaluation, including the differences between the
intended settings and the actual run listed under [Deviations](#deviations).
Every number below is read from a file under `results/` or
`recordings/`; the files are named beside it.

## 1. What was measured

Seven models, four tasks, one prompt per task, the same items for every model. The run
started at `2026-10-29T04:42:49Z` and finished at `2026-10-29T10:04:22Z`
(`results/runs/live/manifest.json`), with prompt version `v1` and scorer version `v1`.

| Task | Items per model | What a model is asked to do | Primary metric |
| --- | ---: | --- | --- |
| T-ROUTE | 616 | route one customer message to one of 77 BANKING77 intents | macro-F1 |
| T-SCOPE | 300 | decide whether a message is outside the supported banking scope | F1 for `out_of_scope` |
| T-REPLY | 150 | return `{intent, urgency, reply}` as structured data | intent-correct among schema-valid |
| T-DOCS | 40 | answer a question about this stack using retrieved guide passages, citing ids | retrieval hit@4 |

The live run made 7,742 calls in 28 model × task cells (`results/runs/live/calls.jsonl`).
T-DOCS has 35 answerable questions and 5 deliberately unanswerable ones; abstaining is the
correct answer to the latter.

## 2. Samples

Samples are stratified and seeded with seed `20261028`, and the id lists are committed
under `data/samples/` together with a sample hash in the run manifest. A gate comparison
between runs with different hashes is refused, so sample drift cannot masquerade as a
model change.

- **T-ROUTE:** 8 messages per intent from the BANKING77 test split, all 77 intents, 616
  items.
- **T-SCOPE:** 150 `oos` messages plus 150 in-scope messages from the CLINC150 `plus` test
  split, 300 items.
- **T-REPLY:** a seeded 150-item subset of the T-ROUTE sample.
- **T-DOCS:** 40 hand-written questions in `data/rag/gold.jsonl`, each with gold passage
  ids; 5 have none and must be refused.

No dataset text is committed. `scripts/fetch_data.py` downloads the four parquet files
into the gitignored `data/raw/`, checks them against `data/CHECKSUMS.txt`, and refuses a
mismatch.

## 3. Models, dates and prices

All seven models were reached through **one gateway** (OpenCode Go) using one
credential, each in the API format the gateway lists as native for that lab. Prices were re-read
from <https://opencode.ai/docs/go/> on **2026-10-29**; they match the committed table
in `config/prices.toml`. The lock file
`config/models.lock.toml` records the model ids, connection, protocol, native format,
endpoint, thinking setting and token floor used by the run.

| Model | Lab | Tier | Wire format | USD per 1M tokens, in / out |
| --- | --- | --- | --- | --- |
| Claude Haiku 5.5 | Anthropic | small | Anthropic Messages | 0.10 / 0.50 |
| GPT 6 Luna | OpenAI | small | OpenAI Responses | 0.10 / 0.50 |
| Gemini 4 Argon | Google | flagship | Gemini generateContent | 2.00 / 10.00 (50% off until 2026-11-30; then 4.00 / 20.00) |
| Muse Spark 1.3 Contributor | Muse | small | OpenAI Responses | 0.10 / 0.20 |
| DeepSeek V4.1 Flash | DeepSeek | small | Chat Completions | 0.15 / 0.60 off-peak (0.30 / 1.20 peak) |
| MiMo-V2.6-Flash | Xiaomi | small | Chat Completions | 0.14 / 0.28 |
| GLM-5.3-Flash | Zhipu (Z.ai) | small | Chat Completions | 0.15 / 0.50 |

**Thinking/reasoning is off or at the lowest level each API accepts**, recorded per model
in the lock file: GPT 6 Luna `reasoning.effort = none`, Gemini 4 Argon
`thinking_level = low`, GLM-5.3-Flash `low`, Muse Spark and DeepSeek `minimal`, MiMo sends
no reasoning parameter (it rejects every value), Claude sends no thinking block.

**Tiers are mixed on purpose:** one flagship and six small models. The page and the README
say so; a reader must not read the result as a same-tier comparison.

**Subscriptions were not used for any benchmark call.** Claude, ChatGPT/Codex and Google
AI Pro subscription CLIs are used only as the crew's coding harnesses. A CLI agent
harness would not give comparable latency or per-token cost, and consumer terms restrict
scripted use outside the products' own interfaces.

## 4. Prompts

One prompt per task, byte-identical for every model, versioned `v1`, no per-model tuning.
The prompts live in `src/which_ai_model/bench/prompts.py`; T-DOCS appends the four
retrieved passages to the same base prompt. **Temperature was not set on the shared
model layer**, so the live run used each provider's default rather than the intended
0. This difference is part of the recorded evidence. A later code change can pass 0, but it cannot change the
recorded run without a paid re-run. Every call allows at least 1,024 output tokens,
because some models spend output tokens on reasoning before answering (a verified case
spent 14 of 16 tokens on reasoning and returned no content).

An empty, cut-off (`finish_reason = length`) or unparseable answer counts as wrong or
schema-invalid, with the reason recorded. It never crashes the run and is never retried
to fish for a better answer; only transport errors are retried.

## 5. Metrics, without a judge

Every score is deterministic. No model judges another model, and no human rates an
answer.

- **T-ROUTE:** accuracy and macro-F1, with a 95% bootstrap interval from 2,000
  fixed-seed resamples.
- **T-SCOPE:** precision, recall and F1 for `out_of_scope`, with Wilson 95% intervals.
- **T-REPLY:** schema-valid rate under the pydantic contract, and intent-correct rate
  among valid outputs, with a Wilson interval.
- **T-DOCS:** retrieval hit@4 (any gold passage in the top 4, over answerable questions),
  citation precision (cited ids must be a subset of the gold ids and at least one must be
  cited), and correct abstention on the five unanswerable questions.
- **Every task:** latency p50 and p95, input/output/reasoning tokens, and cost per 1,000
  items.

**Cost source.** The runner prefers the gateway's per-response measured value
(`usage.opencode.list_cost`) when the response carries it; otherwise it multiplies counted
tokens by `config/prices.toml`. If neither exists the cost is `null` and is shown as
unknown, never guessed. Both values are recorded per call. The live run reports
**USD 1.9385** total (`results/runs/live/budget.json`); the gateway measured **USD 0.3748**
of it across the four models whose API path returns the block, against a **USD 0.5040**
price-table estimate for those same models, a **−25.63%** gap. The gap is larger than the
5% reporting threshold and is therefore reported: prompt caching bills the repeated
routing prompt at a much lower rate than the table's fresh-input price, so the table is an
upper bound.

## 6. Baseline

A TF-IDF baseline (word 1–2-grams) plus logistic regression is trained on the public train
splits and scored on the same samples at cost 0. On T-ROUTE it reaches macro-F1
**0.8593** and accuracy 0.8620 (`results/baseline/metrics.json`); on T-SCOPE it reaches F1
**0.6840**, precision 0.9753 and recall 0.5267. The routing baseline lands just outside
the 3.0-point "good enough" band described below, which is why the fixed rule does not
recommend the free baseline for routing; the rule was not changed after seeing the number.

## 7. "Good enough" and the picker

For each task a model is **good enough** if its primary metric is within **3.0 points** of
the best model on that task. The picker then recommends:

- **cheapest** — the lowest cost per 1,000 items among good-enough models;
- **most accurate** — the best point estimate;
- **fastest** — the lowest latency p50 among good-enough models.

Ties go to the lower cost. The recommendations and the full good-enough lists are in
`results/recommendations.json`. The picker is a rule, not a post-hoc story: on T-ROUTE only
the flagship clears the band, so the "cheapest" recommendation is the flagship itself; on
T-SCOPE six models clear it and the cheapest is GPT 6 Luna; on T-DOCS every model reaches
hit@4 1.0000 and cost decides.

## 8. Regression gate

`wam gate --baseline results/runs/<a> --candidate results/runs/<b>` compares two runs and
returns PASS (0), FAIL (1) or REFUSED (2):

- **FAIL** if the primary metric drops by more than **2.0 points** on any task, or if the
  T-REPLY schema-valid rate drops below **98%**;
- **REFUSED** if the runs are not comparable — different sample hash, prompt version or
  scorer version. A refusal is never a pass.

The live demonstration in `results/gate_demo.json` compares the cheapest overall model
against the one flagship, for every task. It returns FAIL: the cheap models lose 9.30
points on routing and 78.00 points on replies. The refusal demonstration in
`results/gate_refused_demo.json` shows two runs with different prompt versions being
refused before any metric is compared.

## 9. Replay and reproducibility

Every live response is committed under `recordings/` (7,742 rows, 28 files) keyed by
`sha256(model id + prompt version + request digest)` and scrubbed of headers. The two
public training datasets are not committed; `make fetch-data` downloads them to
`data/raw/` once, and the pinned embedding model (about 130 MB) is downloaded once into
the local Hugging Face cache on the first replay. With both present, `make all` replays
the committed responses with no key and no paid call, re-scores all 28 cells, and
rebuilds `results/` byte for byte; `results/MANIFEST.sha256` covers every committed
result file. `HF_HUB_OFFLINE=1` makes a warm-cache replay skip the model-availability
check. The live
artifacts are also kept under `recordings/_run/live/` so `results/` can be deleted and
rebuilt.

## 10. Deviations

The following recorded conditions affect interpretation of the numbers:

1. **No spend cap.** The intended USD 8 hard cap was removed before the run. Spend was tracked and reported instead: USD 1.9385 total. Pacing (2 concurrent
   requests, 30 per minute per model), the stop-on-403/abuse rule and the never-spend-
   without-tracking rule still held. Gemini 4 Argon's per-5-hour allowance was respected
   as a window ceiling.
2. **DeepSeek ran in its off-peak window.** The off-peak schedule was removed after DeepSeek's cell had completed at 05:20Z, inside the
   off-peak window; no peak-price call was made and the extra spend is USD 0.00.
3. **Thinking settings follow the wire format.** The gateway rejects `minimal` on four
   models, so each model uses the lowest setting its API accepts (listed in §3). This is a
   provider constraint, not a change to the method.
4. **No protocol fallback.** All seven models answered in their native API format on the
   first probe, so nothing was switched to the Chat Completions fallback.
5. **Prices unchanged.** Re-read on 2026-10-29; every value matches the frozen table,
   including Gemini 4 Argon's discount through 2026-11-30 and DeepSeek's off-peak rate.
6. **Measured cost is not sent on every path.** The Responses API and Gemini responses
   arrived without `usage.opencode.list_cost`, so GPT 6 Luna, Muse Spark and Gemini 4
   Argon use the price-table estimate (an upper bound). The gap in §5 is computed only
   over the four models where both values exist.
7. **T-DOCS scoring uses the in-memory retrieval twin.** The pgvector service was built
   and `make rag-index` indexed all 35 chunks, but the scored run retrieves from an
   in-memory store over the same `docs/guide` corpus with the same pinned embeddings.
   Top-4 ids were compared on all 40 gold questions and are identical to pgvector on
   40/40. This keeps replay byte-identical without requiring Postgres.
8. **The guide was verified against the final code but not rewritten.** Seven statements
   in `docs/guide/` are stale: a helper that does not exist (`get_connection`), the
   `cost_usd` usage example, the removed spending cap, the temperature-0 claim in
   `benchmark.md`, and three wording drifts about the gate comparison, the `WAM_MODE`
   override and where replay traces are written. Editing the guide changes the RAG
   corpus, which changes the T-DOCS request digests and invalidates the committed
   recordings; new recordings would be needed to measure the changed corpus. These items require a separately versioned corpus snapshot or new recordings to correct.
9. **Temperature is the provider default in the recorded run.** The shared model
   layer did not pass a temperature when the calls were made. The recorded answers
   therefore use the providers' defaults rather than the intended 0. The current
   layer passes 0 by default on every protocol; this does not change existing
   recordings. Measuring the difference would require a new recorded run.

## 11. Reproducing a number

Every headline number is stated with its file in `results/SUMMARY.md`, and every number in
the README's result table is checked against its named file by `make facts` (part of
`make release-check`). To reproduce the whole results tree without a key:

```bash
uv sync
make fetch-data   # public datasets into data/raw/ (gitignored); checksums verified
make all
sha256sum -c results/MANIFEST.sha256
```

## 12. v1.1: a task you define yourself

v1 measured four fixed tasks. v1.1 keeps every v1 number exactly as it was and adds a way
to put **your own task** through the same runner, scorer, gate and card, plus a way to
**re-test a model** when it changes. The following sections explain the task format
and evaluation workflow through the T-DEFECT worked example. Every number named
here is read from `results/`, `data/defect/` or the task spec, and the file is named
beside it.

### 12.1 The task spec

A task is **one TOML file** in `tasks/`, plus the data, label and prompt files that the
TOML names. You do not write Python. `wam task new my-task` writes a commented skeleton
and its prompt template; `wam task check tasks/my-task.toml` reads every file the spec
names and prints each problem in plain language, without calling a model or needing a
key.

The spec says:

- `id`, `title`, `description`, `prompt_version` and `scorer_version` — the id names the
  run files and must be lower case with single hyphens;
- `[data]` — the JSONL file (one object per line), the id field and the text field;
- `[prompt]` — the template file. `{labels}` becomes one `- id: description` line per
  label; `{text}` becomes the item's text; every other brace is left alone;
- `[[outputs]]` — one or more answers. A `label` output names a labels file and may allow
  `unclear`; a `bool` output is yes/no. Each output names the truth field in the data;
- `[metrics]` — the primary metric and the others to report;
- `[gate]` — the thresholds this task must hold, including optional floors;
- `[baseline]` — the free comparison: TF-IDF plus logistic regression for a label,
  a keyword rule for yes/no.

Labels are scored on accuracy, macro-F1, coverage (the share answered) and accuracy among
answered items. **`unclear` counts as wrong in the primary metric**, so abstaining is
never free; coverage and accuracy on answered items show what abstaining bought. A
yes/no output is scored on precision, recall and F1. Every cell also reports the
schema-valid rate, with 95% intervals: a 2,000-resample bootstrap for macro-F1 and F1,
Wilson for rates. An empty, cut-off or unparseable answer is schema-invalid and wrong,
with the reason recorded, and is never retried to fish for a better answer.

### 12.2 T-DEFECT, the worked example

T-DEFECT is the first spec task: sort a public vehicle owner's electrical complaint into
one of **eight** component groups, and say whether it is safety-relevant (a crash, a
fire, an injury or a death). Its spec is `tasks/defect.toml`; its groups are in
`data/defect/groups.toml`.

- **Data.** NHTSA ODI complaints and recalls from the public flat files, pinned to the
  snapshot of **2026-10-31** in `data/defect/nhtsa-spec.toml`. The licence is U.S.
  government work, public domain; the wording and URLs are in `THIRD_PARTY.md`.
- **Selection.** Keep complaints whose component rows all map to exactly one group, whose
  narrative is at least 80 characters, and whose narrative passes the personal-data
  screen. Drop identity and location columns. `data/defect/STATS.md` reports every count,
  including the **1,465** narratives the screen dropped.
- **Split.** The train window (2020–2024) is capped at **20,000** items and is used only
  to train the TF-IDF baseline. The test pool is 2025 or later and is halved; only the
  `wam` half is sampled.
- **Sample.** **600** items, seed **20261031**, stratified by group with at least 30 per
  group and **120** safety-relevant items. Safety complaints are oversampled: the sample
  rate is **20.0%** against a natural `wam`-pool rate of **10.7%**. Both rates are
  reported, and safety precision/recall from this sample are not the natural rate.
- **Prompt `d1`** is identical for every model, at temperature 0 where the API accepts it
  (Claude rejects it and runs at its provider default, recorded in the manifest). Every
  call allows at least 1,024 output tokens.
- **Baselines.** TF-IDF (word 1–2-grams) plus logistic regression for the group, and a
  fixed keyword rule for safety. Both cost 0; the keyword list was committed before any
  T-DEFECT model result existed.
- **Seven models.** The same seven ids as v1, from the same lock file; no model was
  added, removed or substituted.

The run is in `results/runs/live-v1.1/`; its card is `results/cards/defect.md`. The
flagship reached **0.7616** macro-F1, the zero-cost baseline **0.6918**, and the cheapest
good-enough model — which had to clear the safety floor — cost **$0.09** per 1,000 items
(`results/cards/defect.md`).

### 12.3 Gate rules for a spec task

A spec task gates against an earlier run with its **own** thresholds. It **FAILs** when:

- the primary metric drops by more than `primary_drop_points` (T-DEFECT: 2.0 points);
- a floor metric is below its `min` (T-DEFECT: safety recall at least 0.90);
- a floor metric drops by more than its `max_drop_points` (T-DEFECT: 2.0 points);
- the schema-valid rate is below `schema_valid_floor` (T-DEFECT: 98%).

Comparability comes first, exactly as in v1: a different sample hash, prompt version or
scorer version is **REFUSED**, and a refusal is never a pass. v1.1 adds one more check:
the run manifest records each model's temperature, reasoning setting and maximum output
tokens, and a difference is REFUSED unless `--accept-settings-mismatch` is given. With the
flag, the normal verdict stands and a `WARN` line names each difference. A v1 run with no
recorded temperature is read as `provider-default`, so a new run at 0.0 is a real
difference. Built-in v1 tasks keep their v1 thresholds and behaviour.

A model below a floor is never "good enough", whatever its primary metric. If no model
meets the floor, the three picks are empty, the picker says so, and the card says to keep
the current baseline plus a human check and shows no cheapest pick. Otherwise the picker
is v1's rule: within 3.0 points of the best primary, cheapest / most accurate / fastest,
ties to lower cost.

### 12.4 Re-checking a model

A benchmark number is about **one model at one moment**, with one prompt and one set of
sampling settings. `wam recheck` records fresh answers, scores them the same way and gates
them against the earlier run:

```bash
uv run wam recheck run --models config/models.lock.toml --tasks T-SCOPE \
  --against results/runs/live --candidate gpt-6-luna --run-id recheck-gpt-6-luna
```

- It writes fresh answers to `recordings/fresh/<run_id>/` and **never reads a saved
  answer**; the committed v1 recordings are never overwritten or mixed in.
- It writes the scored run and its verdict under `results/runs/<run_id>/`.
- For a brand-new model, an ordinary record run with a one-model lock file is enough:
  only the new model is called.
- For a model updated under the same id, a periodic re-test, or a changed setting, use
  `recheck`. The settings check from §12.3 decides whether the two runs are comparable.
- `wam bench --mode record --fresh --run-id <id>` uses the same fresh root without the
  gate, for answers you will score later.

The committed demonstration re-checks GPT 6 Luna on T-SCOPE at temperature 0. Without the
flag it is **REFUSED** on the one recorded difference (temperature); the v1 manifest
recorded no reasoning setting, so that field is `not-recorded` and is not reported as a
difference. With `--accept-settings-mismatch` it is **PASS** with one `WARN` line. Both
reports are under `results/runs/recheck-demo/` and `results/runs/recheck-demo-accepted/`.

### 12.5 The card

`results/cards/<task>.{md,json}` is one page per task, built only from committed files
under `results/`. It states the job in one sentence; the data, its licence and sample
size; a per-model table with the primary metric and its interval, the other reported
metrics, cost per 1,000 items and p50 latency; the zero-cost baseline row; the
recommendation and the rule that produced it; the gate rule; when to re-check and the
exact command; the cost at 1,000 / 10,000 / 100,000 items per month; and the known
limits. `make cards` rebuilds the four v1 cards and every spec task that has a committed
run.

### 12.6 Cost at volume

`cost_per_month = cost_per_1k × items_per_month / 1000`, from the measured cost per 1,000
items. It is linear only: no bulk discount, no learning curve, no other extrapolation. An
unknown cost stays `unknown`, never 0. The page does the same arithmetic in the browser
from the same committed page data, with the price date shown.

### 12.7 The prompt library

`results/prompt-library/<task>.{md,json}` is the card as something you can use: the task
and when to use it; the recommended model and a fallback (the next good-enough model by
cost); the exact prompt with its version and the output schema; guardrails — an `unclear`
answer goes to a human, the task's floors apply, never auto-close or auto-approve; and
the re-check trigger. `make prompt-library` rebuilds it from the same files.

### 12.8 What v1.1 did not change

The four built-in tasks, their prompts, samples, recordings, results and gate thresholds
are unchanged, and `docs/guide/` is frozen because it is the T-DOCS retrieval corpus. A
spec task is addressed by its spec id; `TaskName` still has its four members. The v1 gate
keeps its own thresholds and exit codes. Two v1.1 result files are the shared hosts of the
new task's data, so their manifest lines changed: `results/recommendations.json` gained
the `defect` entry and `results/page/page-data.json` gained the v1.1 section data. Every
v1 entry inside both files is byte-identical, and no other v1 result changed.

### 12.9 What the v1.1 run cost

The v1.1 run had a task cap of **USD 5.00** and was invoked at **USD 4.99**, leaving a
cent for the re-check demonstration. It made **4,000** calls — 600 items for six models
and 400 for Gemini 4 Argon, after the sample-reduction procedure cut that one cell to fit the
cap — and spent **USD 0.8937**, recorded per model in
`results/runs/live-v1.1/budget.json`. The reduced sample is stated on the T-DEFECT card. The same evidence also produced the T-DEFECT gate
demonstration (**PASS**, a **1.92**-point primary drop for the cheapest good-enough model,
`results/gate_demo_defect.json`) and the re-check demonstration in §12.4.
