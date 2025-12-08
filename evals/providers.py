"""
Async provider implementations for TutorBench evaluation.

Supports concurrent API calls with:
- Rate limiting and concurrency control
- Retry logic with exponential backoff
- Async versions of Together AI, Anthropic, and OpenAI providers
"""

import os
import asyncio
import time
from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass
from importlib import import_module


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrent execution."""

    # PRIMARY CONCURRENCY CONTROLS
    max_concurrent: int = 50  # Max concurrent API calls
    sample_batch_size: int = 10  # Process N samples concurrently
    rubric_batch_size: int = 5  # Evaluate N rubrics per sample concurrently

    # RATE LIMITING
    max_requests_per_second: int = 50  # Global rate limit

    # RETRY LOGIC
    retry_attempts: int = 3
    retry_base_delay: float = 1.0

    # FEATURES
    enable_async: bool = True  # Master switch


class AsyncRateLimiter:
    """
    Rate limiter for async operations.

    Controls both concurrent requests and requests per second.

    Uses lazy initialization to ensure semaphore and lock are created
    in the same event loop that will use them.
    """

    def __init__(self, max_concurrent: int = 50, max_requests_per_second: int = 50):
        """
        Args:
            max_concurrent: Maximum number of concurrent operations
            max_requests_per_second: Maximum requests per second
        """
        self.max_concurrent = max_concurrent
        self.max_rps = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second if max_requests_per_second > 0 else 0
        self.last_request_time = 0

        # Lazy initialization - created on first use
        self._semaphore = None
        self._lock = None
        self._event_loop = None  # Track which event loop we're in

    def _ensure_initialized(self):
        """Ensure semaphore and lock are created in the current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop
            return

        # Re-initialize if we're in a different event loop
        if self._event_loop is not current_loop:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
            self._lock = asyncio.Lock()
            self._event_loop = current_loop
            self.last_request_time = 0  # Reset rate limiting for new loop

    async def __aenter__(self):
        """Enter rate limiter context."""
        self._ensure_initialized()

        await self._semaphore.acquire()

        # Enforce rate limit
        async with self._lock:
            if self.min_interval > 0:
                now = time.time()
                time_since_last = now - self.last_request_time
                if time_since_last < self.min_interval:
                    await asyncio.sleep(self.min_interval - time_since_last)
                self.last_request_time = time.time()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit rate limiter context."""
        self._semaphore.release()


async def async_retry_with_backoff(
    coro_fn: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    verbose: bool = False,
):
    """
    Retry async function with exponential backoff.

    Args:
        coro_fn: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        verbose: Print retry information

    Returns:
        Result from coro_fn

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except Exception as e:
            last_exception = e

            # Check if we should retry
            error_str = str(e).lower()
            should_retry = any(
                keyword in error_str
                for keyword in ["rate limit", "timeout", "connection", "429", "503", "502"]
            )

            if not should_retry or attempt == max_retries - 1:
                raise

            # Calculate delay with exponential backoff
            delay = min(base_delay * (2 ** attempt), max_delay)

            if verbose:
                print(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s due to: {e}")

            await asyncio.sleep(delay)

    # Should not reach here, but raise last exception if we do
    raise last_exception


def _init_provider(
    library_name: str,
    client_class_path: str,
    api_key_vars: List[str],
) -> Any:
    """
    Initialize sync provider client with API key validation.

    Args:
        library_name: Library name for error messages
        client_class_path: Import path (e.g., "anthropic.Anthropic")
        api_key_vars: List of possible API key environment variables

    Returns:
        Initialized client instance

    Raises:
        ImportError: If library not installed
        ValueError: If API key not found
    """
    # Import library
    try:
        module_name, class_name = client_class_path.rsplit(".", 1)
        module = import_module(module_name)
        client_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"{library_name} package required. "
            f"Install with: pip install {library_name}"
        ) from e

    # Get API key from environment
    api_key = None
    for var in api_key_vars:
        api_key = os.getenv(var)
        if api_key:
            break

    if not api_key:
        raise ValueError(f"API key not found. Set one of: {', '.join(api_key_vars)}")

    # Initialize client
    return client_class(api_key=api_key)


def _init_async_provider(
    library_name: str,
    client_class_path: str,
    api_key_vars: List[str],
) -> Any:
    """
    Initialize async provider client with API key validation.

    Args:
        library_name: Library name for error messages
        client_class_path: Import path (e.g., "anthropic.AsyncAnthropic")
        api_key_vars: List of possible API key environment variables

    Returns:
        Initialized async client instance

    Raises:
        ImportError: If library not installed
        ValueError: If API key not found
    """
    # Import library
    try:
        module_name, class_name = client_class_path.rsplit(".", 1)
        module = import_module(module_name)
        client_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"{library_name} package required. "
            f"Install with: pip install {library_name}"
        ) from e

    # Get API key from environment
    api_key = None
    for var in api_key_vars:
        api_key = os.getenv(var)
        if api_key:
            break

    if not api_key:
        raise ValueError(f"API key not found. Set one of: {', '.join(api_key_vars)}")

    # Initialize client
    return client_class(api_key=api_key)


def _format_messages_anthropic(messages: List[Dict]) -> List[Dict]:
    """
    Format messages for Anthropic (system is separate parameter).

    Args:
        messages: User messages

    Returns:
        Formatted messages
    """
    return [
        {"role": msg.get("role", "user"), "content": msg["content"]}
        for msg in messages
    ]


def _format_messages_with_system(system_prompt: str, messages: List[Dict]) -> List[Dict]:
    """
    Format messages with system prompt in messages array (Together AI, OpenAI style).

    Args:
        system_prompt: System prompt
        messages: User messages

    Returns:
        Formatted messages with system prompt prepended
    """
    formatted = []
    if system_prompt:
        formatted.append({"role": "system", "content": system_prompt})

    for msg in messages:
        formatted.append({
            "role": msg.get("role", "user"),
            "content": msg["content"]
        })

    return formatted


def get_anthropic_async_provider(
    model: str = "claude-sonnet-4-20250514",
    rate_limiter: Optional[AsyncRateLimiter] = None,
    config: Optional[ConcurrencyConfig] = None,
) -> Callable:
    """
    Get async Anthropic provider function.

    Args:
        model: Anthropic model identifier
        rate_limiter: Rate limiter instance
        config: Concurrency configuration

    Returns:
        Async model function
    """
    # Use lazy initialization to ensure client is created in the same event loop
    client = None
    client_loop = None

    config = config or ConcurrencyConfig()
    limiter = rate_limiter or AsyncRateLimiter(
        max_concurrent=config.max_concurrent,
        max_requests_per_second=config.max_requests_per_second,
    )

    async def model_fn(system_prompt: str, messages: List[Dict]) -> str:
        """Call Anthropic API asynchronously."""
        nonlocal client, client_loop

        # Re-initialize if in a different event loop
        current_loop = asyncio.get_running_loop()
        if client is None or client_loop is not current_loop:
            client = _init_async_provider(
                library_name="anthropic",
                client_class_path="anthropic.AsyncAnthropic",
                api_key_vars=["ANTHROPIC_API_KEY"],
            )
            client_loop = current_loop

        formatted = _format_messages_anthropic(messages)

        async def api_call():
            async with limiter:
                response = await client.messages.create(
                    model=model,
                    max_tokens=1000,
                    system=system_prompt,
                    messages=formatted,
                )
                return response.content[0].text

        try:
            return await async_retry_with_backoff(
                api_call,
                max_retries=config.retry_attempts,
                base_delay=config.retry_base_delay,
            )
        except Exception as e:
            return f"Error calling Anthropic: {e}"

    return model_fn


def get_together_async_provider(
    model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo",
    rate_limiter: Optional[AsyncRateLimiter] = None,
    config: Optional[ConcurrencyConfig] = None,
) -> Callable:
    """
    Get async Together AI provider function.

    Args:
        model: Together AI model identifier
        rate_limiter: Rate limiter instance
        config: Concurrency configuration

    Returns:
        Async model function
    """
    # Use lazy initialization to ensure client is created in the same event loop
    client = None
    client_loop = None

    config = config or ConcurrencyConfig()
    limiter = rate_limiter or AsyncRateLimiter(
        max_concurrent=config.max_concurrent,
        max_requests_per_second=config.max_requests_per_second,
    )

    async def model_fn(system_prompt: str, messages: List[Dict]) -> str:
        """Call Together AI API asynchronously."""
        nonlocal client, client_loop

        # Re-initialize if in a different event loop
        current_loop = asyncio.get_running_loop()
        if client is None or client_loop is not current_loop:
            client = _init_async_provider(
                library_name="together",
                client_class_path="together.AsyncTogether",
                api_key_vars=["TOGETHER_API_KEY", "TOGETHER_AI_API_KEY"],
            )
            client_loop = current_loop

        formatted = _format_messages_with_system(system_prompt, messages)

        async def api_call():
            async with limiter:
                response = await client.chat.completions.create(
                    model=model,
                    messages=formatted,
                )
                return response.choices[0].message.content

        try:
            return await async_retry_with_backoff(
                api_call,
                max_retries=config.retry_attempts,
                base_delay=config.retry_base_delay,
            )
        except Exception as e:
            return f"Error calling Together AI: {e}"

    return model_fn


def get_openai_async_provider(
    model: str = "gpt-4",
    rate_limiter: Optional[AsyncRateLimiter] = None,
    config: Optional[ConcurrencyConfig] = None,
) -> Callable:
    """
    Get async OpenAI provider function.

    Args:
        model: OpenAI model identifier
        rate_limiter: Rate limiter instance
        config: Concurrency configuration

    Returns:
        Async model function
    """
    # Use lazy initialization to ensure client is created in the same event loop
    client = None
    client_loop = None

    config = config or ConcurrencyConfig()
    limiter = rate_limiter or AsyncRateLimiter(
        max_concurrent=config.max_concurrent,
        max_requests_per_second=config.max_requests_per_second,
    )

    async def model_fn(system_prompt: str, messages: List[Dict]) -> str:
        """Call OpenAI API asynchronously."""
        nonlocal client, client_loop

        # Re-initialize if in a different event loop
        current_loop = asyncio.get_running_loop()
        if client is None or client_loop is not current_loop:
            client = _init_async_provider(
                library_name="openai",
                client_class_path="openai.AsyncOpenAI",
                api_key_vars=["OPENAI_API_KEY"],
            )
            client_loop = current_loop

        formatted = _format_messages_with_system(system_prompt, messages)

        async def api_call():
            async with limiter:
                response = await client.chat.completions.create(
                    model=model,
                    messages=formatted,
                )
                return response.choices[0].message.content

        try:
            return await async_retry_with_backoff(
                api_call,
                max_retries=config.retry_attempts,
                base_delay=config.retry_base_delay,
            )
        except Exception as e:
            return f"Error calling OpenAI: {e}"

    return model_fn


def get_async_provider(
    provider_name: str,
    model: Optional[str] = None,
    rate_limiter: Optional[AsyncRateLimiter] = None,
    config: Optional[ConcurrencyConfig] = None,
) -> Callable[[str, List[Dict]], str]:
    """
    Get an async model function for the specified provider.

    Args:
        provider_name: Provider name ("together", "anthropic", "openai")
        model: Model identifier (uses default if not specified)
        rate_limiter: Shared rate limiter instance
        config: Concurrency configuration

    Returns:
        Async model function with signature: async (system_prompt, messages) -> response

    Raises:
        ValueError: If provider_name is unknown
    """
    provider_name = provider_name.lower()

    if provider_name == "together":
        return get_together_async_provider(
            model or "Qwen/Qwen2.5-7B-Instruct-Turbo",
            rate_limiter,
            config,
        )
    elif provider_name == "anthropic":
        return get_anthropic_async_provider(
            model or "claude-sonnet-4-20250514",
            rate_limiter,
            config,
        )
    elif provider_name == "openai":
        return get_openai_async_provider(
            model or "gpt-4",
            rate_limiter,
            config,
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}")



# Alias for backwards compatibility
def get_provider(
    provider_name: str,
    model: Optional[str] = None,
) -> Callable[[str, List[Dict]], str]:
    """
    Get a model function for the specified provider.
    
    This is an alias for get_async_provider for backwards compatibility.
    
    Args:
        provider_name: Provider name ("together", "anthropic", "openai")
        model: Model identifier (uses default if not specified)
    
    Returns:
        Async model function with signature: async (system_prompt, messages) -> response
    """
    return get_async_provider(provider_name, model)

