# Using the gate

The regression gate guards model switches: it compares a candidate run
against a baseline run and fails loudly when the candidate loses quality.
Both runs must be replay runs scored by the same code, so the verdict is
deterministic — the same two result directories decide the same way on
every machine. A typical comparison pits the cheapest candidate model
against the flagship of the same provider and records the outcome
whatever it is, pass or fail. The sections below list the failure
conditions first, then explain why incomparable runs are refused rather
than passed. Keep this page beside the benchmark page: one measures, the
other judges, and neither works without the other's files.

## When the gate fails

The gate fails a candidate on three conditions. First, the primary metric
— macro-F1 for routing, F1 for scope detection, intent-correct rate for
replies, hit@4 for docs — drops by more than 2.0 points on any task.
Second, the reply schema-valid rate drops below 98%, since malformed
replies break downstream consumers regardless of intent accuracy. Either
drop means the candidate is measurably worse where it counts, and the
failure output names the task and the size of the drop so there is no
detective work left to do. The page's picker builds on the same idea of a
defensible choice: a model is *good enough* within 3.0 points of the best
model on a task, and the picker recommends the cheapest, the most accurate,
and the fastest of those, breaking ties toward lower cost.

## Refusing incomparable runs

Third, and most strictly, the two runs may be incomparable: if the sample
hash, the prompt version, or the scorer version differ between baseline
and candidate, the gate refuses. A refusal is not a pass; it is an
explicit statement that the comparison would be meaningless, and it fails
the command so no pipeline mistakes it for approval:

```python
from pathlib import Path
from which_ai_model.gate import check_gate

passed, reasons = check_gate(Path("results/runs/baseline"), Path("results/runs/candidate"))
```

The function returns a boolean plus one human-readable reason per violated
condition; an empty list with a passing flag means every task held its
quality within tolerance on comparable inputs. Run the gate before any
model change — version bump, provider switch, prompt edit, scorer fix —
keeping the accepted baseline immutable in its own directory and writing
each candidate to a fresh dated one. When the gate passes, the candidate
becomes a defensible new baseline whose manifest explains why.
