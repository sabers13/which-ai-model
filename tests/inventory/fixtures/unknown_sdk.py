"""A vendor SDK the scanner cannot identify: provider must fall back to `unknown`."""

from __future__ import annotations

import os

import mystery_sdk

client = mystery_sdk.Client(api_key=os.environ["MYSTERY_API_KEY"])


def complete(message: str) -> str:
    response = client.complete(model="mystery-1", prompt=message)
    return str(response.text)
