import random
import threading

import httpx2
import openai
import pytest

from repoqa.clients import OPENAI_ERRORS
from repoqa.errors import ProviderError
from repoqa.resilience import (
    RateLimiter,
    RetryableError,
    RetryPolicy,
    backoff_delay,
    call_with_retry,
    classify_sdk_errors,
    parse_retry_after,
)


def test_backoff_is_exponential_with_full_jitter_and_capped() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=5.0)
    rng = random.Random(0)
    for attempt, ceiling in [(0, 1.0), (1, 2.0), (2, 4.0), (5, 5.0)]:
        delays = [backoff_delay(attempt, policy, rng) for _ in range(200)]
        assert all(0 <= d <= ceiling for d in delays)
        assert max(delays) > ceiling * 0.8  # jitter spreads across the whole range


def test_retries_transient_errors_then_succeeds() -> None:
    sleeps: list[float] = []
    outcomes: list[object] = [RetryableError("429"), RetryableError("503", retry_after=2.0), "ok"]

    def flaky() -> str:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome)

    policy = RetryPolicy(max_retries=3, base_delay_s=0.1, max_delay_s=10.0)
    assert call_with_retry(flaky, policy, what="test", sleep=sleeps.append) == "ok"
    assert len(sleeps) == 2
    assert 0 <= sleeps[0] <= 0.1  # jittered backoff
    assert sleeps[1] == 2.0  # server's Retry-After wins


def test_gives_up_after_max_retries() -> None:
    def always_busy() -> str:
        raise RetryableError("HTTP 503")

    with pytest.raises(ProviderError, match="failed after 3 attempts: HTTP 503"):
        call_with_retry(always_busy, RetryPolicy(max_retries=2), what="svc", sleep=lambda _: None)


def test_non_retryable_errors_propagate_immediately() -> None:
    calls = 0

    def bad() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("bug")

    with pytest.raises(ValueError):
        call_with_retry(bad, RetryPolicy(), what="svc", sleep=lambda _: None)
    assert calls == 1


def test_parse_retry_after() -> None:
    assert parse_retry_after({"retry-after": "1.5"}) == 1.5
    assert parse_retry_after({"retry-after": "-3"}) == 0.0
    assert parse_retry_after({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}) is None
    assert parse_retry_after({}) is None


def status_error(status: int, headers: dict[str, str] | None = None) -> openai.APIStatusError:
    request = httpx2.Request("POST", "https://api.openai.com/v1/x")
    response = httpx2.Response(status, headers=headers, request=request)
    return openai.APIStatusError("boom", response=response, body=None)


def test_classify_maps_sdk_errors() -> None:
    with pytest.raises(RetryableError) as retryable, classify_sdk_errors("OpenAI", OPENAI_ERRORS):
        raise status_error(429, {"retry-after": "7"})
    assert retryable.value.retry_after == 7.0

    with pytest.raises(RetryableError), classify_sdk_errors("OpenAI", OPENAI_ERRORS):
        raise status_error(500)

    request = httpx2.Request("POST", "https://api.openai.com/v1/x")
    with (
        pytest.raises(RetryableError, match="connection"),
        classify_sdk_errors("OpenAI", OPENAI_ERRORS),
    ):
        raise openai.APITimeoutError(request=request)

    with (
        pytest.raises(ProviderError, match="HTTP 400"),
        classify_sdk_errors("OpenAI", OPENAI_ERRORS),
    ):
        raise status_error(400)

    with (
        pytest.raises(ProviderError, match="request failed"),
        classify_sdk_errors("OpenAI", OPENAI_ERRORS),
    ):
        raise openai.OpenAIError("no key")


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_rate_limiter_spaces_call_starts() -> None:
    clock = FakeClock()
    limiter = RateLimiter(max_concurrent=4, per_minute=60, clock=clock, sleep=clock.sleep)
    starts = []
    for _ in range(3):
        with limiter:
            starts.append(clock.now)
    assert starts == [100.0, 101.0, 102.0]


def test_rate_limiter_caps_concurrency() -> None:
    limiter = RateLimiter(max_concurrent=1, per_minute=60_000, sleep=lambda _: None)
    holding, release, second_in = threading.Event(), threading.Event(), threading.Event()

    def first() -> None:
        with limiter:
            holding.set()
            release.wait(5)

    def second() -> None:
        with limiter:
            second_in.set()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    threads[0].start()
    assert holding.wait(5)
    threads[1].start()
    assert not second_in.wait(0.1)  # blocked while the only slot is held
    release.set()
    assert second_in.wait(5)
    for thread in threads:
        thread.join()
