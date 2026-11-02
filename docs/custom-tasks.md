# Add your own task

A task is **one TOML file plus a labelled data file**. You do not write Python. The
benchmark treats your task like the four built-in ones: every model gets the same
prompt, answers through the same structured-output path, and is scored the same way.

Five steps, about ten minutes after you have the data. Run every command from the
repository root after `uv sync`. The complete worked example is
[`tasks/defect.toml`](../tasks/defect.toml), its prompt
[`tasks/defect.prompt.txt`](../tasks/defect.prompt.txt) and its data under
[`data/defect/`](../data/defect/); read those beside this page.

## 1. Write the spec and its prompt

```bash
uv run wam task new my-task
```

The id is lower case, digits and single hyphens — `defect` or `my-task`, not
`MyTask`. The command writes two files:

- `tasks/my-task.toml` — the spec, with a comment on every setting;
- `tasks/my-task.prompt.txt` — the prompt template.

The template has two placeholders. `{labels}` becomes one `- id: description` line per
label, in the order of the labels file, and `{text}` becomes the item's text. Leave
every other brace alone: a JSON example in the prompt is fine exactly as written.

Edit the title, the description, the field names and the prompt. If your task has a
yes/no output as well as a label, uncomment the `[[outputs]]` block with `kind = "bool"`.

## 2. Add the data and the labels

**Data** — one JSON object per line, at the path in `[data]`. Every item needs a unique
id, the text field and a value for every output's `truth_field`:

```json
{"id": "0001", "text": "The invoice is wrong.", "queue": "billing", "urgent": true}
```

**Labels** — for every `kind = "label"` output, a small TOML file listed in
`labels_file`. Each group needs an id and a one-line description:

```toml
[[groups]]
id = "billing"
description = "Invoices, payments, refunds and plan changes"

[[groups]]
id = "technical"
description = "Errors, outages, setup and anything that needs a fix"
```

A `[labels]` table of `id = "description"` works too. The id `unclear` is reserved: set
`allow_unclear = true` on the output to let a model abstain.

Keep the data out of the commit if it is large or restricted. The spec, the template and
the labels file are what make the task reproducible.

## 3. Check the spec

```bash
uv run wam task check tasks/my-task.toml
```

It reads every file the spec names and prints each problem in plain language: a missing
data file, a duplicate id, a truth label that is not in the labels file, a prompt
template without `{labels}` or `{text}`, a wrong type in a yes/no field. Fix them and run
it again; `OK: 'my-task' is valid` means the task is ready to run.

The check does not call a model and does not need a key.

## 4. Record a run

```bash
uv run wam bench --mode record --tasks my-task --run-id my-task-1
```

This is a **paid** run: it needs your key in `.env` and a budget. It prints a cost
estimate before the first call; `--cap-usd 2` stops the run at a hard USD 2. Every model
in `config/models.lock.toml` answers every item once. To try one model first, point
`--models` at a lock file that contains only that model. Answers are saved under
`recordings/`, so a replay scores them again without a call or a key:

```bash
uv run wam bench --mode replay --tasks my-task --run-id my-task-1
```

Scoring is deterministic. For a label output you get accuracy, macro-F1 (an `unclear`
answer or an invalid answer counts as wrong), coverage and accuracy on the answered
items; for a yes/no output, precision, recall and F1; always the schema-valid rate, with
95% confidence intervals. The reported primary metric is the one named in
`[metrics] primary`. Nothing is judged by another model or by a person.

## 5. Build the card

```bash
make cards
```

`results/cards/my-task.md` and `.json` hold the job, the data and its licence, the
per-model table with intervals and cost per 1,000 items, the zero-cost baseline, the
recommendation, the gate rule, when to re-check, the cost at volume and the known
limits. Every number comes from the committed run files under `results/`; nothing is
retyped and no model is called. The same card is exported as a ready-to-use prompt by
`make prompt-library` (`results/prompt-library/my-task.md` and `.json`).

## Changing the rules later

The prompt version, the scorer version and the sample hash are recorded with every run
and the gate refuses to compare runs that differ. Bump `prompt_version` when the prompt
changes and `scorer_version` when the truth or the metric changes, then record a fresh
run and rebuild the card. `[gate]` sets the thresholds a new run must hold: the primary
metric may not drop by more than `primary_drop_points`, the schema-valid rate may not
fall below `schema_valid_floor`, and each `[[gate.floors]]` entry is a metric that may
never go below `min` or drop by more than `max_drop_points`. Compare a new run with the
old one using `uv run wam gate --baseline results/runs/<old> --candidate
results/runs/<new>`, or use `wam recheck` when the model id stayed the same; see
[`recheck.md`](recheck.md).
