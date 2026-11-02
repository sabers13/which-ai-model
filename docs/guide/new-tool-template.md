# Adding a new tool from the template

A new tool starts from the template: a module with a typed entry function,
a pydantic output schema, a system prompt, and a parity test. Copy the
closest existing tool rather than a blank file — the router for
single-call classification, the drafter for structured multi-field output,
the scope checker for a branching graph, or the agent for multi-step tool
use. Rename the module, the schema, and the prompt, but keep the frame:
connection in, parsed contract out, trace row written, recording consulted
before any live call is even considered. Because every tool shares this
frame, reading one tool teaches you all five, and reviewing a new one
means checking the four pieces rather than learning a new architecture.

## The four pieces to fill in

First, the schema. Contracts are frozen and shared, so adding a model to
`contracts.py` goes through the approved change process; usually one of
the four existing shapes already fits and you reuse it. The schema is the
tool's real interface — the page, the scorer, and the parity test all read
it — so name fields plainly and require exactly what downstream code
needs, no more. Second, the entry function: it takes the end-user input
plus an already-built chat model and returns the parsed contract, such as
`route_message(message, model) -> RouteOut`, building the structured model
with the `function_calling` strategy and returning the parsed object
without post-processing its text. Resist the urge to add fallback logic
here: retries and second models attach at the agent level, where they
cover every tool uniformly instead of being reimplemented per tool.

## Prompts, servers, and parity tests

Third, the prompt: one versioned string shared by all models, temperature
0 where allowed, with the schema enforced by the framework instead of
described in prose. Fourth, the MCP server entry when the tool needs
outside data: add the function behind `server.tool()`, expose it through
the builder, and load it with `MCPAdapter`, remembering MCP tools are
async-only and return content blocks rather than scalars.

Then prove the tool behaves. A migrated tool must match its legacy twin on
a fixed 50-item fixture set in replay mode, with a fake model returning
the legacy fixture's raw content: same label for classification, same
parsed fields for the drafter, same tool-call sequence and final answer
for the agent. The docs helper is the deliberate exception — retrieval
changes its answers by design — so its check is answering all 10 fixture
questions while citing at least one passage each. Write the parity test
first and keep the fixtures fixed and committed; when it holds with zero
network calls the tool is ready for the live run and the gate behind it.
