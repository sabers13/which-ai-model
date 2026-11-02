"""Versioned benchmark prompts (``PLAN.md`` §7).

One prompt per task, **identical for every model**. Nothing in this module takes a
model, a provider or a connection, so a per-model special case cannot creep in. The
prompt version is part of the recording key and of :class:`~which_ai_model.contracts.RunManifest`;
the gate refuses runs scored with different prompts.

The task predictors in T07a/T09 may build richer messages (T-DOCS adds retrieved
passages). They still report :data:`PROMPT_VERSION`, and the runner records the exact
messages they send, so the comparability rule holds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from which_ai_model.bench.data import BANKING77_LABELS
from which_ai_model.contracts import DocsOut, ReplyOut, RouteOut, ScopeOut, TaskName

__all__ = [
    "PROMPTS",
    "PROMPT_VERSION",
    "Message",
    "Prompt",
    "intent_names",
    "item_text",
    "prompt_for",
]

#: Bumped only when a prompt changes. Recorded in the manifest and the recording key.
PROMPT_VERSION = "v1"

Message = dict[str, str]

_ROUTE_INTENTS = "\n".join(f"- {intent}" for intent in BANKING77_LABELS)

_ROUTE_SYSTEM = f"""\
You route a customer-service message for a retail bank to exactly one Banking77 intent.
Choose one label from this fixed list, copied exactly:
{_ROUTE_INTENTS}

Reply with one JSON object and nothing else: {{"intent": "<one label from the list>"}}.
Never invent a label. Never return more than one label."""

_SCOPE_SYSTEM = """\
You decide whether a customer-service message is outside the supported banking scope.
A message is out of scope when it asks for something this bank does not handle (for
example restaurant booking, weather, or travel advice). A message about cards,
accounts, transfers, payments, top-ups, identity checks or other retail banking topics
is in scope.

Reply with one JSON object and nothing else: {"out_of_scope": true} or
{"out_of_scope": false}."""

_REPLY_SYSTEM = """\
You draft a structured customer-service reply for a retail bank. Read the message and
return exactly one JSON object with these fields:
- "intent": one Banking77 intent label in snake_case;
- "urgency": one of "low", "medium", "high";
- "reply": a short, polite, specific answer that does not promise anything the bank
  cannot do.

Reply with the JSON object and nothing else."""

_DOCS_SYSTEM = """\
You answer a question about this project using only the passages supplied with the
question. Cite the passage ids you used. If the passages do not contain the answer,
abstain instead of guessing.

Reply with one JSON object and nothing else:
{"answer": "<answer>", "cited_ids": ["<passage id>", ...], "abstained": false}
Set "abstained" to true and "cited_ids" to [] when the passages do not answer it."""


@dataclass(frozen=True)
class Prompt:
    """A task prompt: a fixed system message, a user template and the output schema."""

    task: TaskName
    system: str
    user_template: str
    schema: type[BaseModel]

    def messages(self, item: Mapping[str, Any]) -> list[Message]:
        """The exact messages sent for ``item``. No model input, so all models match."""
        user = self.user_template.format(text=item_text(item))
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user},
        ]


def item_text(item: Mapping[str, Any]) -> str:
    """The customer text of an item (``text`` for messages, ``question`` for docs)."""
    value = item.get("text")
    if value is None:
        value = item.get("question", "")
    return str(value)


PROMPTS: dict[TaskName, Prompt] = {
    TaskName.ROUTE: Prompt(
        task=TaskName.ROUTE,
        system=_ROUTE_SYSTEM,
        user_template="Customer message:\n{text}",
        schema=RouteOut,
    ),
    TaskName.SCOPE: Prompt(
        task=TaskName.SCOPE,
        system=_SCOPE_SYSTEM,
        user_template="Customer message:\n{text}",
        schema=ScopeOut,
    ),
    TaskName.REPLY: Prompt(
        task=TaskName.REPLY,
        system=_REPLY_SYSTEM,
        user_template="Customer message:\n{text}",
        schema=ReplyOut,
    ),
    TaskName.DOCS: Prompt(
        task=TaskName.DOCS,
        system=_DOCS_SYSTEM,
        user_template="Question:\n{text}",
        schema=DocsOut,
    ),
}


def prompt_for(task: TaskName | str) -> Prompt:
    """The prompt for ``task`` (accepts the enum value or the ``T-ROUTE`` string)."""
    return PROMPTS[TaskName(task)]


def intent_names() -> Sequence[str]:
    """The 77 Banking77 intent names the routing prompt lists."""
    return BANKING77_LABELS
