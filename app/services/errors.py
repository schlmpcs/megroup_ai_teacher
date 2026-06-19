"""Service-layer exception types shared by the LLM, retrieval and ingestion
layers. Routes map these to HTTP status codes (LLMTimeoutError -> 504, other
LLMError -> 502). Kept in their own module so embeddings/vectorstore can raise
them without importing llm.py (which would create an import cycle)."""

import openai


class LLMError(Exception):
    """Base class for upstream LLM / retrieval failures."""


class LLMTimeoutError(LLMError):
    """Upstream timed out or could not be reached."""


class LLMUpstreamError(LLMError):
    """Upstream returned a 5xx / server-side error."""


class LLMMalformedResponseError(LLMError):
    """The upstream response could not be parsed into a usable result."""


def _map_openai_error(exc: Exception) -> LLMError:
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError(f"OpenAI request timed out: {exc}")
    if isinstance(exc, openai.APIConnectionError):
        return LLMTimeoutError(f"Could not connect to OpenAI: {exc}")
    if isinstance(exc, openai.RateLimitError):
        return LLMUpstreamError(f"OpenAI rate limit: {exc}")
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code >= 500:
            return LLMUpstreamError(f"OpenAI upstream error {exc.status_code}: {exc}")
        return LLMMalformedResponseError(f"OpenAI returned {exc.status_code}: {exc}")
    if isinstance(exc, openai.APIError):
        return LLMMalformedResponseError(f"OpenAI API error: {exc}")
    return LLMMalformedResponseError(f"Unexpected LLM error: {exc}")
