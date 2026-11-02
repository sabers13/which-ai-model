# Running the benchmark

The benchmark scores every model on the same four tasks with the same
prompts: routing messages to intents, detecting out-of-scope messages,
drafting structured replies, and answering docs questions with citations.
One prompt per task is used for all models, versioned and frozen before
the run, with temperature 0 wherever the API allows and no per-model
tuning. Every call allows at least 1,024 output tokens, since some models
spend output tokens on reasoning before answering; an empty, cut-off, or
unparseable answer is scored wrong with the reason recorded, never
retried, and never crashes the run. Scores, latencies, token counts, and costs land in
result files under `results/runs/`, and those files alone feed both the
page and the regression gate. This page explains the samples, the metrics,
and the run discipline; the gate and cost pages cover what happens next.

## Samples, seeds, and the baseline

Samples are stratified and seeded (`20261028`) so they are identical on
every machine: 8 messages per intent from the Banking77 test split for
routing (616 items), 150 out-of-scope plus 150 in-scope messages for scope
detection (300 items), and a 150-item seeded subset of the routing sample
for replies. The id lists are committed under `data/samples/` with a hash
in the run manifest, and the gate refuses to compare runs whose hashes
differ. A TF-IDF plus logistic-regression baseline trained on the public
train splits is scored on the same samples at cost zero, giving every
model result a non-AI reference point it must beat. Because the split is
fixed and committed, a rerun next month scores the identical items, and
any score change traces to the model or the prompt rather than to silent
sample drift.

## Metrics without judges

Every score is deterministic — no model judges another model. Routing
reports accuracy and macro-F1 with a bootstrap 95% interval from 2,000
fixed-seed resamples; scope detection reports precision, recall, and F1
with Wilson intervals; replies report the schema-valid rate and the
intent-correct rate among valid outputs; docs retrieval reports hit@4,
citation precision, and correct abstention on five unanswerable questions.
Beside quality, every task records latency percentiles, token counts split
into input, output, and reasoning, and cost per 1,000 items. Reasoning
tokens get their own column because thinking-heavy models can look cheap
on input while spending heavily before they answer. Together the four
metric groups let the picker trade quality, speed, and price openly, and
the worked example in the cost page shows the arithmetic.

## Replay first, live once, politely

Develop against replay: `run(mode="replay")` serves recorded responses and
is what `make all` uses. Live mode is reserved for the single paid run; it
refuses to start without a resolved key and a budget, prices each
model-by-task cell before spending, and backs off on rate limits while
honouring `Retry-After`, resuming from recordings already written. The
runner stays polite to the shared gateway: at most 2 concurrent requests
and 30 per minute per model, an identifying `User-Agent` and stable
session header per request, and an immediate stop on any forbidden or
abuse-style response. Afterwards `score_task` and `evaluate_cell`
aggregate predictions with intervals, and the manifest files the run id,
hashes, versions, models, and timestamps under `results/runs/`. Treat those
directories as immutable once scored: the gate compares them byte for byte,
and overwriting a baseline in place would destroy the comparison.
