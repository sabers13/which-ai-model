"""L3-style Google Gemini SDK call whose answer is a yes/no substring check."""

from __future__ import annotations

import os

from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_ID = "gemini-2.5-flash"


def check_scope(message: str) -> bool:
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=message,
    )
    answer = response.text.strip().lower()
    return "yes" in answer
