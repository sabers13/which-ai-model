# Sort electrical complaints and flag safety

**Task:** `defect`

## The job

Sort a vehicle owner's electrical complaint into a component group and say whether it describes a crash, a fire or an injury.

## The data

- Source: NHTSA ODI vehicle owner complaints (2020-2024 and 2025-2026 received-date flat files), sampled into data/defect/test.jsonl.
- Licence: U.S. government work, public domain.
- Sample: 600 items.
- Prompt version: `d1`.

## Results

| Model | Primary (95% CI) | `component_group.accuracy` | `component_group.coverage` | `component_group.accuracy_answered` | `safety_relevant.recall` | `safety_relevant.precision` | `schema_valid_rate` | Cost / 1k items (USD) | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemini-4-argon | 0.7616 (95% CI 0.7307 to 0.7899) | 0.8400 | 0.8775 | 0.9573 | 0.9872 | 0.9625 | 1.0000 | $1.31 | 2624 ms |
| muse-spark-1.3-contributor | 0.7425 (95% CI 0.7131 to 0.7680) | 0.8183 | 0.8550 | 0.9571 | 0.9750 | 0.9590 | 1.0000 | $0.09 | 2261 ms |
| mimo-v2.6-flash | 0.7393 (95% CI 0.7121 to 0.7647) | 0.8217 | 0.8767 | 0.9373 | 0.9833 | 0.9365 | 0.9967 | $0.10 | 4016 ms |
| glm-5.3-flash | 0.7388 (95% CI 0.7104 to 0.7637) | 0.8233 | 0.8867 | 0.9286 | 0.9667 | 0.9748 | 1.0000 | $0.09 | 1012 ms |
| deepseek-v4.1-flash | 0.7210 (95% CI 0.6914 to 0.7476) | 0.7967 | 0.8400 | 0.9484 | 0.9833 | 0.9516 | 0.9950 | $0.13 | — |
| gpt-6-luna | 0.6922 (95% CI 0.6644 to 0.7190) | 0.7683 | 0.8367 | 0.9183 | 0.9250 | 0.9823 | 1.0000 | $0.06 | — |
| claude-haiku-5-5 | 0.6819 (95% CI 0.6510 to 0.7102) | 0.7483 | 0.8067 | 0.9277 | 0.9750 | 0.9590 | 1.0000 | $0.16 | — |
| tfidf-logreg + keyword rule (baseline) | 0.6918 (95% CI 0.6405 to 0.7328) | 0.8133 | 1.0000 | 0.8133 | 0.8833 | 0.5222 | 1.0000 | $0 | — |

Primary metric: `component_group.macro_f1`. Its interval is a 2,000-resample bootstrap.

## Recommendation

- Cheapest good enough: `muse-spark-1.3-contributor`.
- Most accurate: `gemini-4-argon`.
- Fastest good enough: `glm-5.3-flash`.

The rule that produced them: within 3.0 points of the best model on the primary metric; a model below any gate floor is not good enough; ties go to lower cost.

## Gate rule

- The primary metric may not drop by more than 2 points against the baseline run.
- The schema-valid rate must stay at or above 98%.
- `safety_relevant.recall` must stay at or above 0.90 and may not drop by more than 2 points.

## When to re-check

`wam recheck run --tasks defect --against results/runs/live-v1.1` when a new model is released, when a model changes under the same id, or after a prompt change (that needs a new baseline run first).

## Cost at volume

| Model | 1,000 items/month | 10,000 items/month | 100,000 items/month |
| --- | ---: | ---: | ---: |
| gemini-4-argon | $1.31 | $13.07 | $130.65 |
| muse-spark-1.3-contributor | $0.09 | $0.89 | $8.89 |
| mimo-v2.6-flash | $0.10 | $0.96 | $9.62 |
| glm-5.3-flash | $0.09 | $0.89 | $8.92 |
| deepseek-v4.1-flash | $0.13 | $1.28 | $12.77 |
| gpt-6-luna | $0.06 | $0.60 | $6.00 |
| claude-haiku-5-5 | $0.16 | $1.56 | $15.64 |

Linear only: `cost_per_month = cost_per_1k * items_per_month / 1000`. An unknown measured cost stays unknown.

## Known limits

- The sample oversamples safety-relevant complaints; the natural rate and the PII drops are in data/defect/STATS.md.
- A model below the safety-recall floor is never recommended, whatever its group accuracy.
- One fixed English-language sample; another fleet, period or language needs its own run.
- Scored on a reduced sample after the §A10 budget cut: gemini-4-argon on 400 of 600 items; its interval is wider, so compare intervals before treating a gap as real.
- One fixed sample; a different domain, language or message mix needs its own run.
- Cost is linear from the measured cost per 1,000 items; the price date is in config/prices.toml.
- Point estimates carry 95% intervals; compare intervals before treating a small gap as real.

## Sources

- Every number is read from `results/runs/live-v1.1/cells.json` and `results/baseline/defect.json`. No model call was made to build this card.
