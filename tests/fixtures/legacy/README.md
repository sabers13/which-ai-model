# Legacy parity fixtures

Shared by the offline migration parity tests. Nothing here is dataset text: the 50
messages are hand-written Banking77-style examples and every model response is invented.

## `parity_items.jsonl`

One JSON object per line, `legacy-01` … `legacy-50`:

| Field | Meaning |
| --- | --- |
| `item_id` | Stable id used in parity reports. |
| `message` | Hand-written customer message. |
| `intent` | Gold Banking77-style label (also the label the fake L1 returns). |
| `docs_question` | Present on 10 items: a project-guide question for the docs helper. |
| `responses.l1.content` | Raw content a fake router returns; first line is the label. |
| `responses.l2.content` | Raw content a fake drafter returns; one JSON object in free text. |
| `responses.l3.content` | Raw yes/no content a fake scope checker returns. |
| `responses.l4.content` | Raw docs answer. |
| `responses.l5.content` | Final agent answer. |
| `responses.l5.tool_calls` | Tool-call sequence (`name`, `arguments`); empty for 40 items, multi-step for 10. |

`fees.json` and `policies.json` are clearly fictional example-bank data. The L5 fixture
answers are consistent with them; change both together.
