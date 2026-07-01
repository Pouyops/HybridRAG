import logging

from langchain_core.rate_limiters import InMemoryRateLimiter
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True for 429 / quota-exhausted errors that are worth retrying."""
    msg = str(exc).lower()
    return (
        "429" in msg
        or "resource exhausted" in msg
        or "rate limit" in msg
        or "quota exceeded" in msg
    )


# Applied to every .invoke() call: up to 6 attempts, 5 s → 10 s → 20 s → 40 s → 60 s → 60 s
llm_retry = retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def create_llm_rate_limiter(requests_per_minute: int = 8) -> InMemoryRateLimiter:
    """
    Create an InMemoryRateLimiter sized for the Gemini free tier.

    Gemini 2.5 Flash free tier: 10 RPM, 250 K TPM, 500 RPD.
    The default of 8 RPM leaves a small headroom below the 10 RPM ceiling.
    Override by passing a value or by setting GEMINI_RPM_LIMIT in your .env.
    """
    return InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60,
        check_every_n_seconds=0.5,
        max_bucket_size=requests_per_minute,
    )


class RateLimitedLLM:
    """
    Transparent wrapper around any LangChain LLM / Runnable that adds
    exponential-backoff retry on rate-limit errors.

    The InMemoryRateLimiter (passed to the LLM constructor) acts as the first
    line of defence by throttling outgoing requests.  This wrapper is the
    second line: if a 429 still slips through it will sleep and retry
    automatically before propagating the error.

    Usage:
        limiter = create_llm_rate_limiter()
        llm = RateLimitedLLM(
            ChatGoogleGenerativeAI(..., rate_limiter=limiter)
        )
        response = llm.invoke("Hello")

        # with_structured_output is also wrapped automatically
        structured = llm.with_structured_output(MySchema)
        result = structured.invoke("Hello")
    """

    def __init__(self, runnable):
        self._runnable = runnable

    @llm_retry
    def invoke(self, *args, **kwargs):
        return self._runnable.invoke(*args, **kwargs)

    def with_structured_output(self, schema, **kwargs) -> "RateLimitedLLM":
        return RateLimitedLLM(self._runnable.with_structured_output(schema, **kwargs))

    def __getattr__(self, name: str):
        # Proxy everything else (e.g. .content, model_name, …) to the inner runnable
        return getattr(self._runnable, name)
