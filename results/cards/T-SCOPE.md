# Check whether a message is in scope

**Task:** `T-SCOPE`

## The job

Decide whether a support message is inside the supported banking scope or outside it.

## The data

- Source: CLINC150 plus (CLINC), test split, 150 out-of-scope + 150 in-scope.
- Licence: CC-BY-3.0.
- Sample: 300 items.
- Prompt version: `v1`.

## Results

| Model | Primary (95% CI) | `precision` | `recall` | `accuracy` | Cost / 1k items (USD) | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| claude-haiku-5-5 | 0.7306 (95% CI 0.6817 to 0.7799) | 0.5975 | 0.9400 | 0.6533 | $0.04 | 968 ms |
| mimo-v2.6-flash | 0.7268 (95% CI 0.6773 to 0.7748) | 0.5924 | 0.9400 | 0.6467 | $0.02 | 2616 ms |
| muse-spark-1.3-contributor | 0.7222 (95% CI 0.6748 to 0.7708) | 0.5813 | 0.9533 | 0.6333 | $0.04 | 1874 ms |
| glm-5.3-flash | 0.7204 (95% CI 0.6718 to 0.7692) | 0.5789 | 0.9533 | 0.6300 | $0.02 | 1680 ms |
| deepseek-v4.1-flash | 0.7118 (95% CI 0.6615 to 0.7617) | 0.5703 | 0.9467 | 0.6167 | $0.06 | 826 ms |
| gpt-6-luna | 0.7097 (95% CI 0.6596 to 0.7585) | 0.5652 | 0.9533 | 0.6100 | $0.02 | 1222 ms |
| gemini-4-argon | 0.7048 (95% CI 0.6546 to 0.7542) | 0.5481 | 0.9867 | 0.5867 | $0.44 | 4206 ms |
| tfidf-logreg (baseline) | 0.6840 (95% CI 0.6119 to 0.7490) | 0.9753 | 0.5267 | 0.7567 | $0 | — |

Primary metric: `f1`. Rate intervals are Wilson 95% intervals, as in `results/`.

## Recommendation

- Cheapest good enough: `gpt-6-luna`.
- Most accurate: `claude-haiku-5-5`.
- Fastest good enough: `deepseek-v4.1-flash`.

The rule that produced them: within 3.0 points of the best model on the primary metric; a model below any gate floor is not good enough; ties go to lower cost.

## Gate rule

- The primary metric may not drop by more than 2 points against the baseline run.
- The schema-valid rate must stay at or above 98%.

## When to re-check

`wam recheck run --tasks T-SCOPE --against results/runs/live` when a new model is released, when a model changes under the same id, or after a prompt change (that needs a new baseline run first).

## Cost at volume

| Model | 1,000 items/month | 10,000 items/month | 100,000 items/month |
| --- | ---: | ---: | ---: |
| claude-haiku-5-5 | $0.04 | $0.40 | $4.04 |
| mimo-v2.6-flash | $0.02 | $0.21 | $2.07 |
| muse-spark-1.3-contributor | $0.04 | $0.41 | $4.15 |
| glm-5.3-flash | $0.02 | $0.17 | $1.74 |
| deepseek-v4.1-flash | $0.06 | $0.56 | $5.56 |
| gpt-6-luna | $0.02 | $0.17 | $1.71 |
| gemini-4-argon | $0.44 | $4.36 | $43.64 |

Linear only: `cost_per_month = cost_per_1k * items_per_month / 1000`. An unknown measured cost stays unknown.

## Known limits

- Binary decision on one fixed English-language sample; the in-scope mix decides the score.
- Public CLINC150 messages may be known to the models.
- One fixed sample; a different domain, language or message mix needs its own run.
- Cost is linear from the measured cost per 1,000 items; the price date is in config/prices.toml.
- Point estimates carry 95% intervals; compare intervals before treating a small gap as real.

## Sources

- Every number is read from `results/runs/live/cells.json`, `results/baseline/metrics.json` and the committed recommendations. No model call was made to build this card.
