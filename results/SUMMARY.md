# Results summary

Every number below is read from a file in `results/`; the file is named next to it.
The live run is the only one that made paid calls. `make all` replays the committed
recordings and rebuilds this file byte for byte.

- Run date: 2026-10-29T04:42:49.595085Z
- Models: 7 (source: `results/runs/live/manifest.json`)
- Prompt/scorer versions: v1 / v1 (source: `results/runs/live/manifest.json`)
- Items per model: 616 routing, 300 scope, 150 reply, 40 docs (7 models, 7,742 calls in total; source: `results/runs/live/cells.json`)

Every cell also carries 95% intervals (2,000-resample bootstrap for accuracy and
macro-F1, Wilson for proportions) in `results/runs/live/cells.json`; the tables
below show point estimates only.

## Headline numbers

- Total spend recorded by the runner: **$1.94** (source: `results/runs/live/budget.json`)
- Gateway-measured spend where the gateway sent `usage.opencode.list_cost`: **$0.37** over 4 models (source: `results/runs/live/budget.json`)
- Price-table estimate for the same models: **$0.50**; measured gap -25.63% (source: `results/runs/live/budget.json`)
- TF-IDF baseline: routing macro-F1 **0.8593**, scope F1 **0.6840**, cost 0 (source: `results/baseline/metrics.json`)
- Inventory call sites: **5** before, **2** after (source: `results/inventory/before.json`, `results/inventory/after.json`)

## Per task

### T-ROUTE — primary metric `macro_f1`

| Model | Primary | Cost / 1k items (USD) | Latency p50 (ms) |
| --- | ---: | ---: | ---: |
| gemini-4-argon | 0.8894 | 1.5679 | 2788 |
| muse-spark-1.3-contributor | 0.8177 | 0.0797 | 1881 |
| claude-haiku-5-5 | 0.8088 | 0.1469 | 1256 |
| gpt-6-luna | 0.7964 | 0.0610 | 1387 |
| glm-5.3-flash | 0.7881 | 0.0279 | 1376 |
| deepseek-v4.1-flash | 0.7874 | 0.0916 | 1118 |
| mimo-v2.6-flash | 0.7797 | 0.0282 | 3265 |
- cheapest: **gemini-4-argon** (0.8894, $1.5679/1k, p50 2788 ms)
- most_accurate: **gemini-4-argon** (0.8894, $1.5679/1k, p50 2788 ms)
- fastest: **gemini-4-argon** (0.8894, $1.5679/1k, p50 2788 ms)

(source: `results/runs/live/cells.json`, `results/recommendations.json`)

### T-SCOPE — primary metric `f1`

| Model | Primary | Cost / 1k items (USD) | Latency p50 (ms) |
| --- | ---: | ---: | ---: |
| claude-haiku-5-5 | 0.7306 | 0.0404 | 968 |
| mimo-v2.6-flash | 0.7268 | 0.0207 | 2616 |
| muse-spark-1.3-contributor | 0.7222 | 0.0415 | 1874 |
| glm-5.3-flash | 0.7204 | 0.0174 | 1680 |
| deepseek-v4.1-flash | 0.7118 | 0.0556 | 826 |
| gpt-6-luna | 0.7097 | 0.0171 | 1222 |
| gemini-4-argon | 0.7048 | 0.4364 | 4206 |
- cheapest: **gpt-6-luna** (0.7097, $0.0171/1k, p50 1222 ms)
- most_accurate: **claude-haiku-5-5** (0.7306, $0.0404/1k, p50 968 ms)
- fastest: **deepseek-v4.1-flash** (0.7118, $0.0556/1k, p50 826 ms)

(source: `results/runs/live/cells.json`, `results/recommendations.json`)

### T-REPLY — primary metric `intent_correct`

| Model | Primary | Cost / 1k items (USD) | Latency p50 (ms) |
| --- | ---: | ---: | ---: |
| gemini-4-argon | 0.7933 | 1.2561 | 6810 |
| claude-haiku-5-5 | 0.6906 | 0.2215 | 2666 |
| deepseek-v4.1-flash | 0.5859 | 0.3654 | 5689 |
| muse-spark-1.3-contributor | 0.3741 | 0.0896 | 4999 |
| glm-5.3-flash | 0.2609 | 0.0819 | 3976 |
| mimo-v2.6-flash | 0.2051 | 0.1151 | 9289 |
| gpt-6-luna | 0.0133 | 0.0419 | 1771 |
- cheapest: **gemini-4-argon** (0.7933, $1.2561/1k, p50 6810 ms)
- most_accurate: **gemini-4-argon** (0.7933, $1.2561/1k, p50 6810 ms)
- fastest: **gemini-4-argon** (0.7933, $1.2561/1k, p50 6810 ms)

(source: `results/runs/live/cells.json`, `results/recommendations.json`)

### T-DOCS — primary metric `hit_at_4`

| Model | Primary | Cost / 1k items (USD) | Latency p50 (ms) |
| --- | ---: | ---: | ---: |
| claude-haiku-5-5 | 1.0000 | 0.2693 | 1558 |
| gemini-4-argon | 1.0000 | 3.4760 | 4852 |
| deepseek-v4.1-flash | 1.0000 | 0.2970 | 1718 |
| glm-5.3-flash | 1.0000 | 0.2048 | 2774 |
| mimo-v2.6-flash | 1.0000 | 0.1995 | 6298 |
| gpt-6-luna | 1.0000 | 0.1483 | 1928 |
| muse-spark-1.3-contributor | 1.0000 | 0.1580 | 3237 |
- cheapest: **gpt-6-luna** (1.0000, $0.1483/1k, p50 1928 ms)
- most_accurate: **gpt-6-luna** (1.0000, $0.1483/1k, p50 1928 ms)
- fastest: **claude-haiku-5-5** (1.0000, $0.2693/1k, p50 1558 ms)

(source: `results/runs/live/cells.json`, `results/recommendations.json`)

## Regression gate

| Baseline (flagship) | Candidate (cheapest) | Outcome |
| --- | --- | --- |
| gemini-4-argon | glm-5.3-flash | **FAIL** |
| gemini-4-argon | gpt-6-luna | **FAIL** |

A `FAIL` is a real quality loss the gate blocks; a refusal is never a pass. The refusal demonstration (different prompt versions) is in `results/gate_refused_demo.json`.

## Spend by model

| Model | Runner spend (USD) | Measured (USD) | Price table (USD) |
| --- | ---: | ---: | ---: |
| claude-haiku-5-5 | 0.146603 | 0.143325 | 0.146603 |
| deepseek-v4.1-flash | 0.139782 | 0.139782 | 0.184411 |
| gemini-4-argon | 1.424230 | not sent | 1.424230 |
| glm-5.3-flash | 0.042896 | 0.042896 | 0.084060 |
| gpt-6-luna | 0.054910 | not sent | 0.054910 |
| mimo-v2.6-flash | 0.048814 | 0.048814 | 0.088918 |
| muse-spark-1.3-contributor | 0.081309 | not sent | 0.081309 |

The gateway sends `usage.opencode.list_cost` on Chat Completions and Anthropic responses; the Responses and Gemini responses reached this project without it, so those rows fall back to the committed price table. Both are recorded and the gap above is computed only over models where both exist (source: `results/runs/live/budget.json`).

## Limits

- All seven models were reached through one gateway (OpenCode Go), each in its lab's own API format.
- Tiers are mixed on purpose: Gemini 4 Argon is a flagship, the other six are small models.
- Public datasets may be known to the models.
- Prices retrieved 2026-10-29 from https://opencode.ai/docs/go/ (source: `config/prices.toml`).
