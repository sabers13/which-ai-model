# Re-checking a model

A benchmark result is a statement about **one model, at one moment, with one prompt and
one set of sampling settings**. When any of those changes, the old numbers are not wrong —
they are just about the old thing. `wam recheck` records fresh answers, scores them the
same way, and gates them against the earlier run so you can see whether the change cost
you quality.

Re-checking is deliberately separate from the normal benchmark:

- it **never reuses a saved answer**;
- it writes its answers to `recordings/fresh/<run_id>/`, so the committed evidence under
  `recordings/` is never overwritten or mixed with a new run;
- it records the sampling settings it used in the run manifest, so a later gate can tell
  whether two runs are actually comparable.

Run the commands from the repository root after `uv sync`. The examples use the seven
models in `config/models.lock.toml` and the committed v1 run, `results/runs/live`; swap in
your own lock file and run id. A re-check makes paid calls, so it needs the key in `.env`
and a budget; the commands below print a cost estimate first.

## Three questions, three commands

### 1. "A new model came out. How does it do on my task?"

A new model has no earlier answers to compare with, so the normal record run is enough —
only the new model's calls are made. Point `--models` at a lock file that contains just
the new model; the runner makes calls only for the models in that file.

```bash
# config/new-model.lock.toml holds one [[model]] block for the new model
uv run wam bench --mode record --models config/new-model.lock.toml \
  --tasks T-SCOPE --run-id new-model
```

Then compare it with the committed run in the usual way:

```bash
uv run wam gate --baseline results/runs/live --candidate results/runs/new-model \
  --model-a <same-task-model> --model-b new-model-id
```

**What it costs.** One task × one model × the number of items in that task's sample.
The runner prints an estimate before the first call; the measured cost lands in
`results/runs/<run_id>/budget.json`. There is no per-run fixed fee.

### 2. "The same model id behaves differently now."

This is what `recheck` is for. It records fresh answers for one model, scores them, and
gates them against the earlier run with the **same model id** on both sides.

```bash
uv run wam recheck run \
  --models config/models.lock.toml \
  --tasks T-SCOPE \
  --against results/runs/live \
  --candidate gpt-6-luna \
  --run-id recheck-gpt-6-luna
```

The run writes:

- fresh answers: `recordings/fresh/recheck-gpt-6-luna/`;
- the scored run: `results/runs/recheck-gpt-6-luna/` (`manifest.json`, `cells.json`,
  `predictions/`, `calls.jsonl`, `budget.json`);
- the verdict: `results/runs/recheck-gpt-6-luna/recheck.json`.

**What it costs.** The same as one model × the selected tasks × the sample sizes. A
re-check of one small task on one model is normally cents, not dollars. If the model has
no price entry and the gateway sends no measured cost, the recorded cost is `unknown`,
never a guess.

### 3. "I changed a setting (temperature, reasoning, output allowance)."

Use the same `recheck` command. The manifest now records, per model,
`temperature`, `reasoning` and `max_output_tokens`, plus the sha256 of the lock file
that was used. The gate compares those settings with the earlier run:

- **no difference** → the normal gate verdict;
- **a difference** → **REFUSED**, with one reason per difference. A refusal is not a
  pass;
- **a difference you meant** → add `--accept-settings-mismatch`. The gate then returns
  its normal verdict plus a `WARN` line naming every difference, so the recorded result
  says plainly that the settings changed.

```bash
uv run wam recheck run \
  --models config/models.lock.toml \
  --tasks T-SCOPE \
  --against results/runs/live \
  --candidate gpt-6-luna \
  --accept-settings-mismatch \
  --run-id recheck-gpt-6-luna-warm
```

A v1 run recorded no temperature. The gate reads such a run as
`temperature = "provider-default"`; a new run that states `0.0` is therefore a real
difference, not a bug in the comparison.

## What a re-check does not do

- It does not change the four built-in task prompts, samples or scorer versions. The
  ordinary comparability checks (`sample_hash`, `prompt_version`, `scorer_version`)
  still come first; a mismatch there is refused even with
  `--accept-settings-mismatch`.
- It does not merge fresh answers into the normal recordings. Replaying the committed
  benchmark still reads `recordings/<task>/`, never `recordings/fresh/`.
- It does not retry an empty, cut-off or unparseable answer to fish for a better one.
  Those count as wrong and schema-invalid, exactly as in the benchmark.
- It does not price a call from a guess. If the model has no entry in
  `config/prices.toml` and the gateway sends no `list_cost`, the cost is `unknown`.

## The committed demonstration

The repository contains a complete example, so you can see both outcomes without
spending anything: GPT 6 Luna on T-SCOPE, recorded fresh at temperature 0 and gated
against the committed v1 `live` run.

```bash
make recheck-demo        # replay the committed demo recordings; no key, no network
```

`make recheck-demo` rebuilds two reports:

1. without the flag — **REFUSED**, because the one recorded difference is real
   (`temperature 0.0` vs `"provider-default"`); the v1 run recorded no reasoning value,
   so it reads as `not-recorded` and is not reported as a difference;
2. with `--accept-settings-mismatch` — **PASS** plus one `WARN` line naming that
   difference.

The reports are `results/runs/recheck-demo/recheck.json` and
`results/runs/recheck-demo-accepted/recheck.json`. In replay mode the command reads the
committed demo recordings, so it makes no network call and costs nothing. With
`WAM_MODE=record` it makes the 300 calls once (about one cent), requires the key in
`.env`, and refreshes `recordings/fresh/recheck-demo/`.

## Fresh runs without the gate

`uv run wam bench --mode record --fresh --run-id <id>` uses the same fresh root,
`recordings/fresh/<id>/`, and the same "never reuse a saved answer" rule. It is useful
when you want clean answers now and will score or gate them later. It does not write a
`recheck.json`.

## Exit codes

`wam recheck run` uses the same vocabulary as `wam gate`:

| Exit | Outcome | Meaning |
| --- | --- | --- |
| 0 | `PASS` | no rule failed |
| 1 | `FAIL` | a quality rule failed (see `reasons`) |
| 2 | `REFUSED` | the runs are not comparable, or the input is invalid; never a pass |

`make recheck-demo` records both outcomes and exits 0 when both reports were written,
whatever the gate said. The verdicts are in the JSON files.
