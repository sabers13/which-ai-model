# Answer from the guide

**Task:** `T-DOCS`

## The job

Answer a question about this stack from retrieved guide passages and cite the passages used.

## The data

- Source: docs/guide in this repository (this project's own text).
- Licence: MIT (this repository).
- Sample: 40 items.
- Prompt version: `v1`.

## Results

| Model | Primary (95% CI) | `citation_precision` | `abstain_correct` | `n_answerable` | `n_unanswerable` | Cost / 1k items (USD) | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| claude-haiku-5-5 | 1.0000 (95% CI 1.0000 to 1.0000) | 0.3250 | 1.0000 | 35 | 5 | $0.27 | 1558 ms |
| deepseek-v4.1-flash | 1.0000 (95% CI 1.0000 to 1.0000) | 0.4250 | 1.0000 | 35 | 5 | $0.30 | 1718 ms |
| gemini-4-argon | 1.0000 (95% CI 1.0000 to 1.0000) | 0.3750 | 1.0000 | 35 | 5 | $3.48 | 4852 ms |
| glm-5.3-flash | 1.0000 (95% CI 1.0000 to 1.0000) | 0.5250 | 1.0000 | 35 | 5 | $0.20 | 2774 ms |
| gpt-6-luna | 1.0000 (95% CI 1.0000 to 1.0000) | 0.4750 | 1.0000 | 35 | 5 | $0.15 | 1928 ms |
| mimo-v2.6-flash | 1.0000 (95% CI 1.0000 to 1.0000) | 0.1250 | 0.4000 | 35 | 5 | $0.20 | 6298 ms |
| muse-spark-1.3-contributor | 1.0000 (95% CI 1.0000 to 1.0000) | 0.4000 | 1.0000 | 35 | 5 | $0.16 | 3237 ms |

No non-AI baseline is defined for this task.

Primary metric: `hit_at_4`. Rate intervals are Wilson 95% intervals, as in `results/`.

## Recommendation

- Cheapest good enough: `gpt-6-luna`.
- Most accurate: `gpt-6-luna`.
- Fastest good enough: `claude-haiku-5-5`.

The rule that produced them: within 3.0 points of the best model on the primary metric; a model below any gate floor is not good enough; ties go to lower cost.

## Gate rule

- The primary metric may not drop by more than 2 points against the baseline run.
- The schema-valid rate must stay at or above 98%.

## When to re-check

`wam recheck run --tasks T-DOCS --against results/runs/live` when a new model is released, when a model changes under the same id, or after a prompt change (that needs a new baseline run first).

## Cost at volume

| Model | 1,000 items/month | 10,000 items/month | 100,000 items/month |
| --- | ---: | ---: | ---: |
| claude-haiku-5-5 | $0.27 | $2.69 | $26.93 |
| deepseek-v4.1-flash | $0.30 | $2.97 | $29.70 |
| gemini-4-argon | $3.48 | $34.76 | $347.61 |
| glm-5.3-flash | $0.20 | $2.05 | $20.48 |
| gpt-6-luna | $0.15 | $1.48 | $14.83 |
| mimo-v2.6-flash | $0.20 | $2.00 | $19.95 |
| muse-spark-1.3-contributor | $0.16 | $1.58 | $15.80 |

Linear only: `cost_per_month = cost_per_1k * items_per_month / 1000`. An unknown measured cost stays unknown.

## Known limits

- The corpus is this project's guide, not a general knowledge base.
- Five questions are deliberately unanswerable; abstaining is the correct answer there.
- One fixed sample; a different domain, language or message mix needs its own run.
- Cost is linear from the measured cost per 1,000 items; the price date is in config/prices.toml.
- Point estimates carry 95% intervals; compare intervals before treating a small gap as real.

## Sources

- Every number is read from `results/runs/live/cells.json`, `results/baseline/metrics.json` and the committed recommendations. No model call was made to build this card.
