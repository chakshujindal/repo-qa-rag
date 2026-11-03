"""Test helpers: fixture paths and SDK clients wired to an in-memory transport."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import anthropic
import httpx2
import openai

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_REPO = FIXTURES / "sample_repo"

Handler = Callable[[httpx2.Request], httpx2.Response]


def mock_openai(handler: Handler) -> openai.OpenAI:
    transport = httpx2.MockTransport(handler)
    return openai.OpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=openai.DefaultHttpxClient(transport=transport),
    )


def mock_anthropic(handler: Handler) -> anthropic.Anthropic:
    transport = httpx2.MockTransport(handler)
    return anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=transport),
    )


def sequence(*responses: httpx2.Response) -> tuple[Handler, list[httpx2.Request]]:
    """A handler that replays `responses` in order and records each request."""
    seen: list[httpx2.Request] = []
    queue = list(responses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return queue.pop(0)

    return handler, seen
