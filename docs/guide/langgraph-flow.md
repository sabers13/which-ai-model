# Build a LangGraph flow

Three of the migrated tools are graphs: the scope checker branches to an
escalation step, the docs helper chains retrieval into answering into
citing, and the multi-step agent loops until the question is resolved. All
three are built with `StateGraph` from the installed `langgraph` package
using the same small pattern: declare the state shape, add one node per
step, wire the edges, compile, and invoke with the initial state. The
`create_agent` builder itself returns a compiled graph, so plain flows and
agent flows share one runtime and one tracing path. If you can build the
tiny classifier below, you can build all three production flows: they add
retrieval, models, and branches around this exact skeleton.

## The minimal node-and-edge pattern

State is a `TypedDict`; each node maps state to a dictionary of updates.
Plain edges fix the order, and conditional edges branch on the state.
Keep node functions pure — they read the state and return updates without
touching the network — and put model calls in designated nodes that go
through the shared model layer, so tracing and recording see every call in
the same format. Only add a checkpointer when the flow must genuinely
pause and resume; benchmark flows run start to finish:

```python
from typing import TypedDict
from langgraph.graph import END, START, StateGraph


class ScopeState(TypedDict):
    message: str
    out_of_scope: bool


def classify(state: ScopeState) -> dict[str, object]:
    return {"out_of_scope": "refund" in state["message"].lower()}


builder = StateGraph(ScopeState)
builder.add_node("classify", classify)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", lambda s: "escalate" if s["out_of_scope"] else END)
```

Compiling catches wiring mistakes — unknown node names, missing edges —
before the first message flows, so read the compile error before
suspecting your node logic.

## Composing agents and retrieval into the graph

When a flow needs an agent inside it, build the agent first with
`create_agent` and add the compiled result as a single node instead of
rebuilding the loop from primitives. The docs flow follows retrieve, then
answer, then cite: retrieval queries the passage store built by `make
rag-index`, answering calls the model with those passages in context, and
citing attaches the passage ids the answer actually used. Citations are
plain strings of the form `guide/<page>#<n>` carried through the state,
and the final node checks that every cited id was really retrieved.

The retrieval-augmented answer comes from `answer_question`, which takes
the question with the store and the model and returns a `DocsOut` object;
`build_rag_graph` wires the same steps into a reusable compiled graph for
the benchmark. For questions the guide cannot answer, the flow sets
`abstained=True` with an empty citation list instead of guessing. The gold
set holds five such unanswerable questions, and the correct response to
each is an explicit refusal rather than a fabricated answer.
