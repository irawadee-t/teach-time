"""Base classes and utilities for async LLM providers."""

import os
import asyncio
import time
from typing import List, Callable, Any
from importlib import import_module

from ..config import RETRYABLE_ERRORS


class AsyncRateLimiter:
    """
    Rate limiter for async operations with lazy initialization.

    Controls both concurrent requests and requests per second.
    Automatically reinitializes when used in a different event loop.
    """

    def __init__(self, max_concurrent: int = 50, max_requests_per_second: int = 50):
        self.max_concurrent = max_concurrent
        self.max_rps = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second if max_requests_per_second > 0 else 0
        self.last_request_time = 0
        self._semaphore = None
        self._lock = None
        self._event_loop = None

    def _ensure_initialized(self):
        """Ensure semaphore and lock are created in the current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        if self._event_loop is not current_loop:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
            self._lock = asyncio.Lock()
            self._event_loop = current_loop
            self.last_request_time = 0

    async def __aenter__(self):
        self._ensure_initialized()
        await self._semaphore.acquire()

        async with self._lock:
            if self.min_interval > 0:
                now = time.time()
                time_since_last = now - self.last_request_time
                if time_since_last < self.min_interval:
                    await asyncio.sleep(self.min_interval - time_since_last)
                self.last_request_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._semaphore.release()


async def async_retry_with_backoff(
    coro_fn: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    verbose: bool = False,
):
    """Retry async function with exponential backoff on retryable errors."""
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            should_retry = any(kw in error_str for kw in RETRYABLE_ERRORS)

            if not should_retry or attempt == max_retries - 1:
                raise

            delay = min(base_delay * (2 ** attempt), max_delay)
            if verbose:
                print(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
            await asyncio.sleep(delay)

    raise last_exception


def init_client(library_name: str, client_class_path: str, api_key_vars: List[str]) -> Any:
    """
    Initialize a provider client with API key validation.

    Args:
        library_name: Library name for error messages
        client_class_path: Import path (e.g., "anthropic.AsyncAnthropic")
        api_key_vars: List of possible API key environment variables

    Returns:
        Initialized client instance
    """
    try:
        module_name, class_name = client_class_path.rsplit(".", 1)
        module = import_module(module_name)
        client_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"{library_name} package required. Install with: pip install {library_name}") from e

    api_key = None
    for var in api_key_vars:
        api_key = os.getenv(var)
        if api_key:
            break

    if not api_key:
        raise ValueError(f"API key not found. Set one of: {', '.join(api_key_vars)}")

    return client_class(api_key=api_key)
