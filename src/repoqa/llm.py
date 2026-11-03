"""LLM clients behind one small protocol: OpenAI, Anthropic, and a deterministic FakeLLM."""

from __future__ import annotations

import json
import re
from collections import deque
from typing import Literal, Protocol, TypedDict

import anthropic
import openai

from repoqa.clients import ANTHROPIC_ERRORS, OPENAI_ERRORS, anthropic_client, openai_client
from repoqa.config import Settings
from repoqa.errors import ProviderError
from repoqa.resilience import RateLimiter, RetryPolicy, call_with_retry, classify_sdk_errors

MAX_OUTPUT_TOKENS = 16_000


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class LLMClient(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, system: str, messages: list[ChatMessage]) -> str:
        """Return the model's text reply to `messages` under the `system` prompt."""
        ...


class OpenAIChat:
    def __init__(
        self, model: str, *, client: openai.OpenAI, policy: RetryPolicy, limiter: RateLimiter
    ) -> None:
        self.model = model
        self._client = client
        self._policy = policy
        self._limiter = limiter

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def complete(self, system: str, messages: list[ChatMessage]) -> str:
        return call_with_retry(
            lambda: self._request(system, messages), self._policy, what=f"OpenAI {self.model}"
        )

    def _request(self, system: str, messages: list[ChatMessage]) -> str:
        with self._limiter, classify_sdk_errors("OpenAI", OPENAI_ERRORS):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, *messages],
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise ProviderError(f"OpenAI {self.model} returned an empty reply")
        return content


class AnthropicChat:
    def __init__(
        self,
        model: str,
        *,
        client: anthropic.Anthropic,
        policy: RetryPolicy,
        limiter: RateLimiter,
    ) -> None:
        self.model = model
        self._client = client
        self._policy = policy
        self._limiter = limiter

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def complete(self, system: str, messages: list[ChatMessage]) -> str:
        return call_with_retry(
            lambda: self._request(system, messages), self._policy, what=f"Anthropic {self.model}"
        )

    def _request(self, system: str, messages: list[ChatMessage]) -> str:
        with self._limiter, classify_sdk_errors("Anthropic", ANTHROPIC_ERRORS):
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system,
                messages=list(messages),
                # Grounded Q&A over short sources does not need deep thinking.
                output_config={"effort": "medium"},
                # If a safety classifier declines, let the API retry on its recommended model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        if response.stop_reason == "refusal":
            raise ProviderError(f"Anthropic {self.model} declined to answer")
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise ProviderError(f"Anthropic {self.model} returned no text")
        return text


# Matches the source format written by `answer.build_prompt`.
_SOURCE_BLOCK = re.compile(
    r'<source id="(\d+)" cite="([^"]+)" symbol="([^"]*)">\n(.*?)\n</source>', re.DOTALL
)


class FakeLLM:
    """Deterministic stand-in for tests and the no-API-key demo.

    It reads the numbered sources from the prompt and writes an extractive answer that
    cites the top ones, so the whole pipeline (citations, post-check) runs offline.
    `scripted` replies, if given, are returned first, in order.
    """

    name = "fake"

    def __init__(self, scripted: list[str] | None = None, max_sources: int = 3) -> None:
        self._scripted = list(scripted or [])
        self._max_sources = max_sources
        self.calls: deque[list[ChatMessage]] = deque(maxlen=20)  # recent calls, for tests

    def complete(self, system: str, messages: list[ChatMessage]) -> str:
        self.calls.append(list(messages))
        if self._scripted:
            return self._scripted.pop(0)
        sources = _SOURCE_BLOCK.findall(messages[0]["content"])
        if not sources:
            return json.dumps({"answer": "", "insufficient_context": True})
        lines = ["The most relevant code for this question:"]
        for _, cite, symbol, body in sources[: self._max_sources]:
            label = f"`{symbol}`" if symbol else cite.split(":")[0]
            lines.append(f"- {label}: {_summary(body)} [{cite}]")
        return json.dumps({"answer": "\n".join(lines), "insufficient_context": False})


def _summary(body: str) -> str:
    """First docstring/comment line if present, else the first non-blank line."""
    stripped = [line.strip() for line in body.splitlines() if line.strip()]
    for line in stripped:
        text = line.strip("\"'#/* ").strip()
        if line.startswith(('"""', "'''", "#", "//")) and text:
            return text
    return stripped[0]


def build_llm(settings: Settings, limiter: RateLimiter) -> LLMClient:
    policy = RetryPolicy(max_retries=settings.max_retries)
    if settings.llm == "openai":
        return OpenAIChat(
            settings.openai_model,
            client=openai_client(settings.timeout_s),
            policy=policy,
            limiter=limiter,
        )
    if settings.llm == "anthropic":
        return AnthropicChat(
            settings.anthropic_model,
            client=anthropic_client(settings.timeout_s),
            policy=policy,
            limiter=limiter,
        )
    return FakeLLM()
