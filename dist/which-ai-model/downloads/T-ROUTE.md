# Route a support message

**Task:** `T-ROUTE`

## The job

Send each customer message to the one of 77 banking intents that fits it.

## The data

- Source: Banking77 (PolyAI), test split, 8 messages per intent.
- Licence: CC-BY-4.0.
- Sample: 616 items.
- Prompt version: `v1`.

## Results

| Model | Primary (95% CI) | `accuracy` | Cost / 1k items (USD) | p50 latency |
| --- | ---: | ---: | ---: | ---: |
| gemini-4-argon | 0.8894 (95% CI 0.8541 to 0.9074) | 0.8912 | $1.57 | 2788 ms |
| muse-spark-1.3-contributor | 0.8177 (95% CI 0.7801 to 0.8371) | 0.8263 | $0.08 | 1881 ms |
| claude-haiku-5-5 | 0.8088 (95% CI 0.7691 to 0.8275) | 0.8198 | $0.15 | 1256 ms |
| gpt-6-luna | 0.7964 (95% CI 0.7556 to 0.8163) | 0.8052 | $0.06 | 1387 ms |
| glm-5.3-flash | 0.7881 (95% CI 0.7463 to 0.8092) | 0.7987 | $0.03 | 1376 ms |
| deepseek-v4.1-flash | 0.7874 (95% CI 0.7478 to 0.8066) | 0.8101 | $0.09 | 1118 ms |
| mimo-v2.6-flash | 0.7797 (95% CI 0.7393 to 0.8001) | 0.7906 | $0.03 | 3265 ms |
| tfidf-logreg (baseline) | 0.8593 (95% CI 0.8242 to 0.8790) | 0.8620 | $0 | — |

Primary metric: `macro_f1`. Its interval is a 2,000-resample bootstrap.

## Recommendation

- Cheapest good enough: `gemini-4-argon`.
- Most accurate: `gemini-4-argon`.
- Fastest good enough: `gemini-4-argon`.

The rule that produced them: within 3.0 points of the best model on the primary metric; a model below any gate floor is not good enough; ties go to lower cost.

## Gate rule

- The primary metric may not drop by more than 2 points against the baseline run.
- The schema-valid rate must stay at or above 98%.

## When to re-check

`wam recheck run --tasks T-ROUTE --against results/runs/live` when a new model is released, when a model changes under the same id, or after a prompt change (that needs a new baseline run first).

## Cost at volume

| Model | 1,000 items/month | 10,000 items/month | 100,000 items/month |
| --- | ---: | ---: | ---: |
| gemini-4-argon | $1.57 | $15.68 | $156.79 |
| muse-spark-1.3-contributor | $0.08 | $0.80 | $7.97 |
| claude-haiku-5-5 | $0.15 | $1.47 | $14.69 |
| gpt-6-luna | $0.06 | $0.61 | $6.10 |
| glm-5.3-flash | $0.03 | $0.28 | $2.79 |
| deepseek-v4.1-flash | $0.09 | $0.92 | $9.16 |
| mimo-v2.6-flash | $0.03 | $0.28 | $2.82 |

Linear only: `cost_per_month = cost_per_1k * items_per_month / 1000`. An unknown measured cost stays unknown.

## Known limits

- 77 fixed English-language banking intents; a different intent set needs its own task.
- Public Banking77 messages may be known to the models.
- One fixed sample; a different domain, language or message mix needs its own run.
- Cost is linear from the measured cost per 1,000 items; the price date is in config/prices.toml.
- Point estimates carry 95% intervals; compare intervals before treating a small gap as real.

## Sources

- Every number is read from `results/runs/live/cells.json`, `results/baseline/metrics.json` and the committed recommendations. No model call was made to build this card.
