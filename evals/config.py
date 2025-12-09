"""Centralized configuration for TutorBench evaluation."""

from dataclasses import dataclass

# Model defaults by provider
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-20250514"
TOGETHER_DEFAULT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
OPENAI_DEFAULT_MODEL = "gpt-4"
JUDGE_MODEL = "claude-sonnet-4-20250514"

# Token limits
TUTOR_MAX_TOKENS = 1000
JUDGE_MAX_TOKENS = 500

# Concurrency defaults
MAX_CONCURRENT = 50  # Lower for parallel workers with different rate limits
SAMPLE_BATCH_SIZE = 10
RUBRIC_BATCH_SIZE = 5
MAX_REQUESTS_PER_SECOND = 200
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0

# API key environment variables by provider
API_KEY_VARS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "together": ("TOGETHER_API_KEY", "TOGETHER_AI_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
}

# Retryable error keywords
RETRYABLE_ERRORS = frozenset([
    "rate limit", "timeout", "connection", "429", "503", "502", "500", "529",
    "overloaded", "not ready", "cloudflare", "server error"
])


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrent API calls and rate limiting."""
    max_concurrent: int = MAX_CONCURRENT
    sample_batch_size: int = SAMPLE_BATCH_SIZE
    rubric_batch_size: int = RUBRIC_BATCH_SIZE
    max_requests_per_second: int = MAX_REQUESTS_PER_SECOND
    retry_attempts: int = RETRY_ATTEMPTS
    retry_base_delay: float = RETRY_BASE_DELAY
    enable_async: bool = True
