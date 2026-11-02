"""L2-style Anthropic SDK call whose JSON answer is pulled out with a regex."""

from __future__ import annotations

import os
import re

from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def draft_reply(message: str) -> dict[str, str]:
    response = client.messages.create(
        model=os.environ["ANTHROPIC_MODEL"],
        max_tokens=512,
        messages=[{"role": "user", "content": message}],
    )
    text = response.content[0].text
    match = re.search(r"\{.*\}", text)
    intent = match.group(0) if match else ""
    return {"intent": intent, "raw": text}
