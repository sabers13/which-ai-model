# LLM call-site inventory

Root: `src/which_ai_model/tools`

Modules scanned: **8**

Call sites found: **2**

## Summary

| Dimension | Value | Count |
| --- | --- | ---: |
| call_style | LangChain | 2 |
| provider | langchain | 2 |
| structured_output | none | 2 |

## Call sites

| File:line | Tool | Provider | Call style | Model | Structured output | Inputs | Credentials |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `src/which_ai_model/tools/agent.py:212` | arun_agent | langchain | LangChain | — | none | question, max_turns | — |
| `src/which_ai_model/tools/scope.py:129` | check_scope | langchain | LangChain | — | none | message | — |

## Notes

**Known blind spot.** The scanner counts an `invoke` it can see in the scanned module. Four of the five tools reach the model through a shared helper — `_template.ask` → `core.structured` (router, scope, drafter) and `rag.answer_question_with_metadata` (docs) — and those calls live in `core/` and `rag/`, outside this root, so only the two graph-level invokes above (agent, scope) are listed. The after picture is five tools on one shared structured call path, with `LangChain structured` output for all five: the row count is a lower bound, not a call total.
