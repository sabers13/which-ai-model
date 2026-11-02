# Draft a structured reply

**Task:** `T-REPLY`

## Task and when to use

Turn a customer message into one typed object: intent, urgency and reply text.

## Recommended model

- **Recommended:** `gemini-4-argon` — $1.26 per 1,000 items.
- **Fallback:** none yet; only one model is good enough at this sample size.

The rule that chose them: within 3.0 points of the best model on the primary metric; a model below any gate floor is not good enough; ties go to lower cost.

## Prompt (version `v1`)

```
You draft a structured customer-service reply for a retail bank. Read the message and
return exactly one JSON object with these fields:
- "intent": one Banking77 intent label in snake_case;
- "urgency": one of "low", "medium", "high";
- "reply": a short, polite, specific answer that does not promise anything the bank
  cannot do.

Reply with the JSON object and nothing else.

Customer message:
{text}
```

## Output schema

```json
{
  "additionalProperties": false,
  "description": "T-REPLY: a structured customer-service reply.",
  "properties": {
    "intent": {
      "minLength": 1,
      "title": "Intent",
      "type": "string"
    },
    "reply": {
      "minLength": 1,
      "title": "Reply",
      "type": "string"
    },
    "urgency": {
      "enum": [
        "low",
        "medium",
        "high"
      ],
      "title": "Urgency",
      "type": "string"
    }
  },
  "required": [
    "intent",
    "urgency",
    "reply"
  ],
  "title": "ReplyOut",
  "type": "object"
}
```

## Guardrails

- An `unclear` or abstained answer goes to a human; it is scored as wrong, so abstaining is never free.
- The schema-valid rate must stay at or above 98%.
- Never auto-close, auto-send or auto-approve a model answer; a person makes the final call.

## Re-check

`wam recheck run --tasks T-REPLY --against results/runs/live` when a new model is released, when a model changes under the same id, or after a prompt change (that needs a new baseline run first).
