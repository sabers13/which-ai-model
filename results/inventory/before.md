# LLM call-site inventory

Root: `src/which_ai_model/legacy`

Modules scanned: **9**

Call sites found: **5**

## Summary

| Dimension | Value | Count |
| --- | --- | ---: |
| call_style | hand-written tool loop | 1 |
| call_style | raw HTTP | 1 |
| call_style | vendor SDK | 3 |
| provider | anthropic | 1 |
| provider | gemini | 1 |
| provider | openai-compatible | 3 |
| structured_output | none | 3 |
| structured_output | regex | 1 |
| structured_output | substring | 1 |

## Call sites

| File:line | Tool | Provider | Call style | Model | Structured output | Inputs | Credentials |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `src/which_ai_model/legacy/l1_router.py:57` | route_message | openai-compatible | raw HTTP | env:ROUTER_MODEL | none | message | ROUTER_KEY |
| `src/which_ai_model/legacy/l2_drafter.py:72` | draft_reply | anthropic | vendor SDK | env:DRAFTER_MODEL | regex | message | ANTHROPIC_API_KEY |
| `src/which_ai_model/legacy/l3_scope.py:56` | check_scope | gemini | vendor SDK | env:SCOPE_MODEL | substring | message | GEMINI_API_KEY |
| `src/which_ai_model/legacy/l4_docs.py:72` | docs_answer | openai-compatible | vendor SDK | env:DOCS_MODEL | none | question, guide_path | DOCS_KEY |
| `src/which_ai_model/legacy/l5_agent.py:73` | run_agent | openai-compatible | hand-written tool loop | env:AGENT_MODEL | none | question | AGENT_KEY |
