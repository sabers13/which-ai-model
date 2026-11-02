# Cost and budget

Every model call in the benchmark carries a dollar cost, recorded per call
in the trace and aggregated per 1,000 items beside each quality metric.
Cost is always measured or looked up, never guessed: the runner prefers
the gateway's per-response measured value when the response carries it,
otherwise multiplies the counted tokens by the prices table in
`config/prices.toml`, and records `None` — displayed as unknown — when
neither source exists. Any gap between measured and tabulated cost above
5% is reported rather than smoothed over, so a stale price cannot quietly
flatter a model, and unknown costs propagate to the page as unknown. The
rest of this page traces where each number originates, then shows how the
runner keeps total spend under the hard cap without cutting corners.

## Where the numbers come from

Token counts are split three ways on every call: input tokens, output
tokens, and reasoning tokens. The split matters because the seven models
price them differently and some spend output tokens on thinking before they
answer — one verified run spent 14 of 16 output tokens on reasoning and
returned no content at all. The prices table records per-model input and
output rates in dollars per million tokens together with the source URL
and the retrieval date, and the live run re-checks every rate before the
first call, writing the confirmed values to `config/models.lock.toml`.
Thinking controls are set to the lowest level each API allows and recorded
per model in the same lock file, so the cost comparison reflects settings
anyone can audit.

The helper that turns usage into dollars is a single function:

```python
from which_ai_model.contracts import ModelRef
from which_ai_model.core import cost_usd

price = cost_usd({"input_tokens": 120, "output_tokens": 45}, model)
```

It returns a float when a source exists and `None` otherwise. Callers
treat `None` as unknown all the way to the page; there is no fallback
constant and no per-model fudge factor.

## Staying inside the budget

The whole live run has a hard spending cap recorded in
`config/budget.toml`. Before starting each model-by-task cell, the runner
estimates its cost from the sample size and the locked prices; if the
remaining budget cannot cover the cell, it shrinks the most expensive
routing sample first (from 8 to 5 messages per intent), then the next most
expensive model, records the reduction as a deviation, and never exceeds
the cap. If even the reduced plan would overspend, the run stops instead
of spending — that is a stop condition, not a suggestion.

Politeness is part of budgeting too: at most 2 concurrent requests and 30
requests per minute per model, backoff that honours `Retry-After`, resume
from recordings so paid responses are never bought twice, and an immediate
halt on any forbidden or abuse-style reply from the gateway. Spending that
must stretch across rate-limit windows is spread over time rather than
forced through, because a blocked account costs more than a slow night.
The final report states total spend against the cap with per-model
breakdown, closing the loop from per-call cents to the headline number.
