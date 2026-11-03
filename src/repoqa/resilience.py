"""Retries with exponential backoff + jitter, and a client-side rate limiter for provider calls."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import TracebackType

from repoqa.errors import ProviderError

logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """A transient failure (HTTP 429/5xx, timeout, connection reset) worth retrying."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 4
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0


def backoff_delay(attempt: int, policy: RetryPolicy, rng: random.Random) -> float:
    """ "Full jitter" backoff: uniform in [0, min(max_delay, base * 2**attempt)]."""
    ceiling = min(policy.max_delay_s, policy.base_delay_s * 2**attempt)
    return rng.uniform(0, ceiling)


def call_with_retry[T](
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    what: str,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Call `fn`, retrying on `RetryableError`. Other exceptions propagate immediately."""
    rng = rng or random.Random()
    for attempt in range(policy.max_retries + 1):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == policy.max_retries:
                raise ProviderError(f"{what} failed after {attempt + 1} attempts: {exc}") from exc
            # Prefer the server's Retry-After hint; otherwise back off with jitter.
            delay = exc.retry_after
            if delay is None:
                delay = backoff_delay(attempt, policy, rng)
            delay = min(delay, policy.max_delay_s)
            logger.info("%s: %s; retry %d in %.2fs", what, exc, attempt + 1, delay)
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def is_retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds from a numeric `Retry-After` header (HTTP-date values are ignored)."""
    raw = headers.get("retry-after")
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except ValueError:
        return None


@dataclass(frozen=True)
class SdkErrors:
    """The exception classes of one provider SDK."""

    base: type[Exception]
    status: type[Exception]
    connection: type[Exception]


@contextmanager
def classify_sdk_errors(provider: str, errors: SdkErrors) -> Iterator[None]:
    """Map an SDK's exceptions onto ours: 429/5xx/network -> `RetryableError`,
    anything else (bad request, auth, ...) -> `ProviderError`, which is not retried.

    The OpenAI and Anthropic SDKs share this shape: `APIStatusError` carries `status_code`
    and `response`; `APIConnectionError` (including timeouts) means there was no response.
    """
    try:
        yield
    except errors.connection as exc:
        raise RetryableError(f"connection error: {exc}") from exc
    except errors.status as exc:
        status = int(getattr(exc, "status_code", 0))
        if not is_retryable_status(status):
            raise ProviderError(f"{provider} returned HTTP {status}: {exc}") from exc
        response = getattr(exc, "response", None)
        headers = response.headers if response is not None else {}
        raise RetryableError(f"HTTP {status}", parse_retry_after(headers)) from exc
    except errors.base as exc:
        raise ProviderError(f"{provider} request failed: {exc}") from exc


class RateLimiter:
    """Client-side limiter: at most `max_concurrent` calls in flight, and call starts spaced
    so we never exceed `per_minute` requests per minute. Thread-safe (the API server is
    multi-threaded)."""

    def __init__(
        self,
        max_concurrent: int,
        per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._interval = 60.0 / per_minute
        self._lock = threading.Lock()
        self._next_start = 0.0
        self._clock = clock
        self._sleep = sleep

    def __enter__(self) -> RateLimiter:
        self._slots.acquire()
        with self._lock:
            now = self._clock()
            start = max(now, self._next_start)
            self._next_start = start + self._interval
        if start > now:
            self._sleep(start - now)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._slots.release()
