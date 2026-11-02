# Sort electrical complaints and flag safety

**Task:** `defect`

## Task and when to use

Sort a vehicle owner's electrical complaint into a component group and say whether it describes a crash, a fire or an injury.

## Recommended model

- **Recommended:** `muse-spark-1.3-contributor` — $0.09 per 1,000 items.
- **Fallback:** `glm-5.3-flash` — $0.09 per 1,000 items.

The rule that chose them: within 3.0 points of the best model on the primary metric; a model below any gate floor is not good enough; ties go to lower cost.

## Prompt (version `d1`)

```
A vehicle owner has filed a complaint about an electrical or electronic part of a
vehicle. Read the complaint and answer two questions about it.

Question 1: which component group does the complaint describe?
Use exactly one of these group ids. Each line is the id, then what the group covers.

{labels}

If the complaint does not clearly fit one group, answer "unclear". Do not guess.

Question 2: is the complaint safety-relevant?
Answer true when the complaint says the vehicle crashed or caught fire, or that someone
was injured or killed. Answer false when it does not. Judge only what the text says.

Complaint:
{text}

Reply with one JSON object and nothing else, in exactly this shape:
{"component_group": "<one group id, or unclear>", "safety_relevant": true or false}
```

## Output schema

```json
{
  "additionalProperties": false,
  "properties": {
    "component_group": {
      "enum": [
        "battery_charging_12v",
        "ev_high_voltage",
        "instrument_panel_displays",
        "exterior_lighting",
        "wipers_washers",
        "power_windows_locks_switches",
        "driver_assist_sensors",
        "airbags",
        "unclear"
      ],
      "title": "Component Group",
      "type": "string"
    },
    "safety_relevant": {
      "title": "Safety Relevant",
      "type": "boolean"
    }
  },
  "required": [
    "component_group",
    "safety_relevant"
  ],
  "title": "DefectOut",
  "type": "object"
}
```

## Guardrails

- An `unclear` or abstained answer goes to a human; it is scored as wrong, so abstaining is never free.
- Gate floor: `safety_relevant.recall` must stay at or above 0.90 and may not drop by more than 2 points against the baseline.
- The schema-valid rate must stay at or above 98%.
- Never auto-close, auto-send or auto-approve a model answer; a person makes the final call.

## Re-check

`wam recheck run --tasks defect --against results/runs/live-v1.1` when a new model is released, when a model changes under the same id, or after a prompt change (that needs a new baseline run first).
