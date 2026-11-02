# Check whether a message is in scope

**Task:** `T-SCOPE`

## Task and when to use

Decide whether a support message is inside the supported banking scope or outside it.

## Recommended model

- **Recommended:** `gpt-6-luna` — $0.02 per 1,000 items.
- **Fallback:** `glm-5.3-flash` — $0.02 per 1,000 items.

The rule that chose them: within 3.0 points of the best model on the primary metric; a model below any gate floor is not good enough; ties go to lower cost.

## Prompt (version `v1`)

```
You decide whether a customer-service message is outside the supported banking scope.
A message is out of scope when it asks for something this bank does not handle (for
example restaurant booking, weather, or travel advice). A message about cards,
accounts, transfers, payments, top-ups, identity checks or other retail banking topics
is in scope.

Reply with one JSON object and nothing else: {"out_of_scope": true} or
{"out_of_scope": false}.

Customer message:
{text}
```

## Output schema

```json
{
  "additionalProperties": false,
  "description": "T-SCOPE: is the message outside the supported scope?",
  "properties": {
    "out_of_scope": {
      "title": "Out Of Scope",
      "type": "boolean"
    }
  },
  "required": [
    "out_of_scope"
  ],
  "title": "ScopeOut",
  "type": "object"
}
```

## Guardrails

- An `unclear` or abstained answer goes to a human; it is scored as wrong, so abstaining is never free.
- The schema-valid rate must stay at or above 98%.
- Never auto-close, auto-send or auto-approve a model answer; a person makes the final call.

## Re-check

`wam recheck run --tasks T-SCOPE --against results/runs/live` when a new model is released, when a model changes under the same id, or after a prompt change (that needs a new baseline run first).
