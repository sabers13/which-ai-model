# Add a tool

A tool in this project is a plain Python function with a typed signature and
a docstring, wrapped so the model can call it during a run. The docstring
matters more than you might expect: it becomes the tool description the
model reads when deciding what to call, so write it as an instruction that
says when to use the tool and what each argument means. Keep the argument
types simple — strings, numbers, booleans, and small pydantic models — and
return either a scalar or a small dictionary that the agent can quote back.
Fallback and retry are never coded inside the tool; they are attached once
at agent construction as middleware, which keeps every tool honest about
what it cost and what it returned. The template in the later guide page
shows the exact module layout to copy, so a new tool starts consistent
with the five that came before it.

## From a function to a callable tool

The thinnest path is a decorated function handed to the agent builder with a
system prompt and an expected response shape. The builder used here is
`create_agent` from the installed `langchain` package, which returns a
compiled graph you invoke with a message list:

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[lookup_fee, check_policy],
    system_prompt="Answer the fee question using the tools. Quote their results.",
)
result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
```

Use the asynchronous `ainvoke` form. Some tool sources in this stack are
async-only and raise on synchronous calls, so awaiting everywhere keeps every
tool working through the same path. The result dictionary carries the message
history and, when a response format is declared, a `structured_response`
field with the parsed object. Tool-call arguments that fail schema
validation must surface as errors for scoring; do not enable silent repair,
because the benchmark counts a schema-invalid answer as wrong.

## Fallback and retry belong to middleware, not the tool

A tool function itself contains no retry loop and no fallback model. Those
concerns are attached once, at agent construction, with two middleware
pieces: a fallback that switches to a second model instance on transport
failures, and a retry policy with a small retry count, exponential backoff,
and an error outcome that raises instead of inventing an answer. Pass model
instances built from `Connection` objects so the fallback keeps its own base
URL, key reference, and headers; passing plain model name strings would
resolve through provider defaults and lose the gateway address.

Fallback and retry trigger only on transport problems such as rate limits
and unreachable endpoints. A slow or wrong tool result is returned to the
model as an observation, and a schema-invalid final answer is scored, never
re-requested. This keeps the cost accounting honest: every request the runner
paid for appears in the trace, and no hidden retry fishes for a better
answer off the books.
