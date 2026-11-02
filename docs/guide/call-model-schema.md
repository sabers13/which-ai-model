# Call a model with a schema

Every tool in this project asks the model for structured data, never for
free text that is parsed with a prompt or a regular expression. The shape of
each answer is a pydantic model in `contracts.py`: `RouteOut` carries one
intent label, `ScopeOut` carries an out-of-scope flag, `ReplyOut` carries an
intent plus an urgency level plus the reply text, and `DocsOut` carries an
answer plus the passage ids it cites plus an abstention flag. Because the
schema is a real type, a malformed answer is a validation error you can
count, not a formatting surprise you discover on the page. The model layer
builds the right provider class from the connection's protocol, and one
shared helper wraps any of those classes with your schema, so tool code
never branches on providers and every benchmark call stays comparable.

## Building the chat model from a connection

You never construct a provider class by hand in tool code. The model layer
builds it from a `Connection` plus a model id, choosing the class from the
protocol: `ChatOpenAI` for `OPENAI_COMPATIBLE`, `ChatOpenAI` in Responses
mode for `OPENAI_RESPONSES`, `ChatAnthropic` for `ANTHROPIC`, and
`ChatGoogleGenerativeAI` for `GEMINI`. Custom base URLs, the identifying
`User-Agent` and session headers, the lowest reasoning level each API allows,
and at least 1,024 output tokens are all set inside this one function, so
every benchmark call is comparable:

```python
from which_ai_model.connect import get_connection, load_connections
from which_ai_model.core import make_chat_model

connection = get_connection("opencode-go-openai", load_connections())
model = make_chat_model(connection, "deepseek-v4.1-flash")
```

Retries happen in middleware, so the model is built with retries disabled;
only transport errors are ever retried, and a bad answer is scored as wrong
rather than re-requested. An empty answer, a cut-off answer, or an
unparseable answer counts as wrong with the reason recorded. It never crashes
the run.

## Requesting a schema with `with_structured_output`

Wrap the model with the schema and call it with `include_raw=True`, which
keeps the raw provider message beside the parsed object for the trace:

```python
from which_ai_model.contracts import ReplyOut
from which_ai_model.core import make_structured_model

structured = make_structured_model(model, ReplyOut)
result = structured.invoke("Customer: my card never arrived. Draft the reply.")
```

The layer uses the `function_calling` strategy for every benchmark model so
the comparison stays identical across providers. With `include_raw=True` a
parse failure arrives as data — a `parsing_error` value next to a `None`
parse — and the runner records `schema_valid=False` while keeping the raw
content for diagnosis. Inside agent loops the response format is declared
with an explicit strategy object and error handling switched off, because a
schema-invalid answer must be scored, not silently repaired by the framework.
Token usage for the call comes from the message `usage_metadata`, where
thinking tokens appear under output details when the provider reports them.
