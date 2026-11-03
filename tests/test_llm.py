import json

import httpx2
import pytest

from repoqa.config import Settings
from repoqa.errors import ProviderError
from repoqa.llm import AnthropicChat, FakeLLM, OpenAIChat, build_llm
from repoqa.resilience import RateLimiter, RetryPolicy
from tests.helpers import mock_anthropic, mock_openai, sequence

PROMPT = (
    '<source id="1" cite="app/auth.py:10-20" symbol="hash_password">\n'
    'def hash_password(p):\n    """Hash a password with PBKDF2."""\n</source>\n\n'
    '<source id="2" cite="README.md:1-3" symbol="">\n# Shop\nA tiny shop.\n</source>\n\n'
    '<source id="3" cite="app/cart.py:1-2" symbol="total">\ndef total(c):\n    return 1\n'
    "</source>\n\n"
    "Question: how are passwords hashed?"
)


def anthropic_message(text: str, stop_reason: str = "end_turn") -> httpx2.Response:
    content = [{"type": "text", "text": text}] if text else []
    return httpx2.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })  # fmt: skip


def openai_completion(content: str | None) -> httpx2.Response:
    return httpx2.Response(200, json={
        "id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-6.1-sol",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
    })  # fmt: skip


def test_fake_llm_cites_the_sources_it_was_given() -> None:
    reply = json.loads(FakeLLM().complete("system", [{"role": "user", "content": PROMPT}]))
    assert reply["insufficient_context"] is False
    assert "`hash_password`: Hash a password with PBKDF2. [app/auth.py:10-20]" in reply["answer"]
    assert "README.md: Shop [README.md:1-3]" in reply["answer"]
    assert "`total`: def total(c): [app/cart.py:1-2]" in reply["answer"]  # no docstring


def test_fake_llm_without_sources_reports_insufficient_context() -> None:
    reply = json.loads(FakeLLM().complete("s", [{"role": "user", "content": "no sources"}]))
    assert reply == {"answer": "", "insufficient_context": True}


def test_fake_llm_scripted_replies_come_first() -> None:
    fake = FakeLLM(scripted=["one"])
    assert fake.complete("s", [{"role": "user", "content": PROMPT}]) == "one"
    assert fake.complete("s", [{"role": "user", "content": PROMPT}]).startswith("{")
    assert len(fake.calls) == 2


def test_anthropic_request_shape_and_text_extraction(
    no_wait_policy: RetryPolicy, limiter: RateLimiter
) -> None:
    handler, seen = sequence(anthropic_message('{"answer": "hi"}'))
    chat = AnthropicChat(
        "claude-sonnet-5-5", client=mock_anthropic(handler), policy=no_wait_policy,
        limiter=limiter,
    )  # fmt: skip
    assert chat.complete("be brief", [{"role": "user", "content": "q"}]) == '{"answer": "hi"}'
    body = json.loads(seen[0].content)
    assert body["model"] == "claude-sonnet-5-5"
    assert body["system"] == "be brief"
    assert body["fallbacks"] == "default"
    assert "temperature" not in body
    assert seen[0].headers["anthropic-beta"] == "server-side-fallback-2026-07-01"
    assert chat.name == "anthropic:claude-sonnet-5-5"


def test_anthropic_retries_overload_then_succeeds(
    no_wait_policy: RetryPolicy, limiter: RateLimiter
) -> None:
    handler, seen = sequence(
        httpx2.Response(529, json={"type": "error", "error": {"type": "overloaded_error"}}),
        anthropic_message("ok"),
    )
    chat = AnthropicChat(
        "m", client=mock_anthropic(handler), policy=no_wait_policy, limiter=limiter
    )
    assert chat.complete("s", [{"role": "user", "content": "q"}]) == "ok"
    assert len(seen) == 2


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (anthropic_message("", stop_reason="refusal"), "declined"),
        (anthropic_message(""), "no text"),
        (httpx2.Response(400, json={"type": "error", "error": {"type": "invalid"}}), "HTTP 400"),
    ],
)
def test_anthropic_failures_are_provider_errors(
    response: httpx2.Response, message: str, no_wait_policy: RetryPolicy, limiter: RateLimiter
) -> None:
    handler, seen = sequence(response)
    chat = AnthropicChat(
        "m", client=mock_anthropic(handler), policy=no_wait_policy, limiter=limiter
    )
    with pytest.raises(ProviderError, match=message):
        chat.complete("s", [{"role": "user", "content": "q"}])
    assert len(seen) == 1  # not retried


def test_openai_chat_request_shape(no_wait_policy: RetryPolicy, limiter: RateLimiter) -> None:
    handler, seen = sequence(openai_completion('{"answer": "x"}'))
    chat = OpenAIChat(
        "gpt-6.1-sol", client=mock_openai(handler), policy=no_wait_policy, limiter=limiter
    )
    assert chat.complete("sys", [{"role": "user", "content": "q"}]) == '{"answer": "x"}'
    body = json.loads(seen[0].content)
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["response_format"] == {"type": "json_object"}
    assert chat.name == "openai:gpt-6.1-sol"


def test_openai_chat_gives_up_after_retries(
    no_wait_policy: RetryPolicy, limiter: RateLimiter
) -> None:
    handler, seen = sequence(*[httpx2.Response(503, json={"error": {}}) for _ in range(3)])
    chat = OpenAIChat("m", client=mock_openai(handler), policy=no_wait_policy, limiter=limiter)
    with pytest.raises(ProviderError, match="failed after 3 attempts"):
        chat.complete("s", [{"role": "user", "content": "q"}])
    assert len(seen) == 3


def test_openai_chat_empty_reply(no_wait_policy: RetryPolicy, limiter: RateLimiter) -> None:
    handler, _ = sequence(openai_completion(None))
    chat = OpenAIChat("m", client=mock_openai(handler), policy=no_wait_policy, limiter=limiter)
    with pytest.raises(ProviderError, match="empty reply"):
        chat.complete("s", [{"role": "user", "content": "q"}])


def test_build_llm(monkeypatch: pytest.MonkeyPatch, limiter: RateLimiter) -> None:
    assert isinstance(build_llm(Settings(), limiter), FakeLLM)
    assert isinstance(build_llm(Settings(llm="anthropic"), limiter), AnthropicChat)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert isinstance(build_llm(Settings(llm="openai"), limiter), OpenAIChat)
