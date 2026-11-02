# Draft a structured reply

**Task:** `T-REPLY`

## The job

Turn a customer message into one typed object: intent, urgency and reply text.

## The data

- Source: Banking77 (PolyAI), a seeded 150-item subset of the routing sample.
- Licence: CC-BY-4.0.
- Sample: 150 items.
- Prompt version: `v1`.

## Results

| Model | Primary (95% CI) | `schema_valid_rate` | `intent_correct_all` | Cost / 1k items (USD) | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| gemini-4-argon | 0.7933 (95% CI 0.7216 to 0.8504) | 1.0000 | 0.7933 | $1.26 | 6810 ms |
| claude-haiku-5-5 | 0.6906 (95% CI 0.6095 to 0.7615) | 0.9267 | 0.6400 | $0.22 | 2666 ms |
| deepseek-v4.1-flash | 0.5859 (95% CI 0.4993 to 0.6675) | 0.8533 | 0.5000 | $0.37 | 5689 ms |
| muse-spark-1.3-contributor | 0.3741 (95% CI 0.3001 to 0.4546) | 0.9800 | 0.3667 | $0.09 | 4999 ms |
| glm-5.3-flash | 0.2609 (95% CI 0.1820 to 0.3589) | 0.6133 | 0.1600 | $0.08 | 3976 ms |
| mimo-v2.6-flash | 0.2051 (95% CI 0.1419 to 0.2871) | 0.7800 | 0.1600 | $0.12 | 9289 ms |
| gpt-6-luna | 0.0133 (95% CI 0.0037 to 0.0473) | 1.0000 | 0.0133 | $0.04 | 1771 ms |

No non-AI baseline is defined for this task.

Primary metric: `intent_correct`. Rate intervals are Wilson 95% intervals, as in `results/`.

## Recommendation

- Cheapest good enough: `gemini-4-argon`.
- Most accurate: `gemini-4-argon`.
- Fastest good enough: `gemini-4-argon`.

The rule that produced them: within 3.0 points of the best model on the primary metric; a model below any gate floor is not good enough; ties go to lower cost.

## Gate rule

- The primary metric may not drop by more than 2 points against the baseline run.
- The schema-valid rate must stay at or above 98%.

## When to re-check

`wam recheck run --tasks T-REPLY --against results/runs/live` when a new model is released, when a model changes under the same id, or after a prompt change (that needs a new baseline run first).

## Cost at volume

| Model | 1,000 items/month | 10,000 items/month | 100,000 items/month |
| --- | ---: | ---: | ---: |
| gemini-4-argon | $1.26 | $12.56 | $125.61 |
| claude-haiku-5-5 | $0.22 | $2.22 | $22.15 |
| deepseek-v4.1-flash | $0.37 | $3.65 | $36.54 |
| muse-spark-1.3-contributor | $0.09 | $0.90 | $8.96 |
| glm-5.3-flash | $0.08 | $0.82 | $8.19 |
| mimo-v2.6-flash | $0.12 | $1.15 | $11.51 |
| gpt-6-luna | $0.04 | $0.42 | $4.19 |

Linear only: `cost_per_month = cost_per_1k * items_per_month / 1000`. An unknown measured cost stays unknown.

## Known limits

- Intent-correct is measured among schema-valid answers; reply text is not judged by a model or a person.
- No non-AI baseline is defined for free-text replies.
- One fixed sample; a different domain, language or message mix needs its own run.
- Cost is linear from the measured cost per 1,000 items; the price date is in config/prices.toml.
- Point estimates carry 95% intervals; compare intervals before treating a small gap as real.

## Sources

- Every number is read from `results/runs/live/cells.json`, `results/baseline/metrics.json` and the committed recommendations. No model call was made to build this card.
