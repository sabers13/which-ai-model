# Load MCP tools

The fee-table and policy tools used by the multi-step agent are served by a
small local MCP server built with the official `mcp` Python SDK, and the
agent loads them through the `langchain.mcp` client adapter. This split
keeps the tools reusable: the same server can serve this project's agent,
an inspector, or any other MCP client without changing the tool code. Do
not add the archived `langchain-mcp-adapters` package; the maintained path
is `langchain[mcp]`, whose import raises a one-per-process beta warning
that is expected and left alone in both source and tests. The server
defines the tools once; every consumer lists them at startup, so adding a
tool means editing one file and restarting the consumers. The next two
sections walk through the server half first and the client half second,
in the order you would build them.

## Writing the server with the official SDK

In `mcp` version 2 the server class is `MCPServer` imported from
`mcp.server.mcpserver`; the older `FastMCP` name no longer exists. You
create a named server, decorate plain functions as tools, and choose the
transport when you run it, not when you construct it:

```python
from mcp.server.mcpserver import MCPServer

server = MCPServer("fee-tools")


@server.tool()
def fee(amount: float) -> float:
    """Return the fee for an amount."""
    return amount * 0.01


if __name__ == "__main__":
    server.run("stdio")
```

The builder for this project's tools lives in `tools/mcp_server.py` and is
exposed as `build_mcp_server()`. The server runs over standard input and
output as a subprocess, so it needs no socket, no port, and no network —
which is also why the tool path works unchanged under the blocked-network
test suite.

## Listing the tools on the LangChain side

The client side is `MCPAdapter`, used as an async context manager. The
target selects the transport: pass a `Path` to the server script for a
stdio subprocess, an `http(s)` URL for a remote server, or a configuration
dictionary with an `mcpServers` key. A plain string must be an `http(s)`
URL, so always use a `Path` object for a local script:

```python
from pathlib import Path
from langchain.mcp import MCPAdapter

async with MCPAdapter(Path("src/which_ai_model/tools/mcp_server.py")) as adapter:
    tools = await adapter.list_tools()
    result = await tools[0].ainvoke({"amount": 100.0})
```

The adapter closes the subprocess when the block exits, so tools never
leak processes between questions in a long benchmark run.

Two details catch people out. First, MCP tools are async-only: calling
`invoke` raises `NotImplementedError`, so use `ainvoke` and drive the agent
with `ainvoke` as well. Second, a tool call returns a
list of MCP content blocks — dictionaries with `type`, `text`, and an id —
rather than a plain Python value, so parse the text block to recover the
scalar.
