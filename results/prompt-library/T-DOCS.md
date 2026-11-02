# Answer from the guide

**Task:** `T-DOCS`

## Task and when to use

Answer a question about this stack from retrieved guide passages and cite the passages used.

## Recommended model

- **Recommended:** `gpt-6-luna` — $0.15 per 1,000 items.
- **Fallback:** `muse-spark-1.3-contributor` — $0.16 per 1,000 items.

The rule that chose them: within 3.0 points of the best model on the primary metric; a model below any gate floor is not good enough; ties go to lower cost.

## Prompt (version `v1`)

```
You answer a question about this project using only the passages supplied with the
question. Cite the passage ids you used. If the passages do not contain the answer,
abstain instead of guessing.

Reply with one JSON object and nothing else:
{"answer": "<answer>", "cited_ids": ["<passage id>", ...], "abstained": false}
Set "abstained" to true and "cited_ids" to [] when the passages do not answer it.

Question:
{text}
```

## Output schema

```json
{
  "additionalProperties": false,
  "description": "T-DOCS: an answer with the passage ids it cites, or an explicit abstention.",
  "properties": {
    "abstained": {
      "default": false,
      "title": "Abstained",
      "type": "boolean"
    },
    "answer": {
      "title": "Answer",
      "type": "string"
    },
    "cited_ids": {
      "items": {
        "type": "string"
      },
      "title": "Cited Ids",
      "type": "array"
    }
  },
  "required": [
    "answer"
  ],
  "title": "DocsOut",
  "type": "object"
}
```

## Guardrails

- An `unclear` or abstained answer goes to a human; it is scored as wrong, so abstaining is never free.
- The schema-valid rate must stay at or above 98%.
- Never auto-close, auto-send or auto-approve a model answer; a person makes the final call.

## Re-check

`wam recheck run --tasks T-DOCS --against results/runs/live` when a new model is released, when a model changes under the same id, or after a prompt change (that needs a new baseline run first).
