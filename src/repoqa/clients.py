"""SDK client construction. Our own retry loop replaces the SDKs' built-in retries."""

from __future__ import annotations

import anthropic
import openai

from repoqa.errors import ConfigError
from repoqa.resilience import SdkErrors

OPENAI_ERRORS = SdkErrors(openai.OpenAIError, openai.APIStatusError, openai.APIConnectionError)
ANTHROPIC_ERRORS = SdkErrors(
    anthropic.AnthropicError, anthropic.APIStatusError, anthropic.APIConnectionError
)


def openai_client(timeout_s: float) -> openai.OpenAI:
    try:
        return openai.OpenAI(timeout=timeout_s, max_retries=0)
    except openai.OpenAIError as exc:  # raised when OPENAI_API_KEY is missing
        raise ConfigError(f"Cannot create the OpenAI client: {exc}") from exc


def anthropic_client(timeout_s: float) -> anthropic.Anthropic:
    # Credentials resolve from ANTHROPIC_API_KEY (or an `ant auth login` profile) at call time.
    return anthropic.Anthropic(timeout=timeout_s, max_retries=0)
