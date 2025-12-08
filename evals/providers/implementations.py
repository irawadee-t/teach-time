"""Concrete provider implementations for Anthropic, Together AI, and OpenAI."""

import asyncio
from typing import List, Dict, Callable, Optional

from ..config import (
    ConcurrencyConfig,
    API_KEY_VARS,
    ANTHROPIC_DEFAULT_MODEL,
    TOGETHER_DEFAULT_MODEL,
    OPENAI_DEFAULT_MODEL,
    TUTOR_MAX_TOKENS,
)
from .base import AsyncRateLimiter, async_retry_with_backoff, init_client


def _create_async_provider(
    provider_name: str,
    client_class_path: str,
    model: str,
    rate_limiter: Optional[AsyncRateLimiter],
    config: Optional[ConcurrencyConfig],
    api_call_fn: Callable,
) -> Callable:
    """
    Factory for creating async provider functions with lazy client initialization.

    Args:
        provider_name: Name for error messages
        client_class_path: Import path for async client
        model: Model identifier
        rate_limiter: Optional rate limiter
        config: Concurrency configuration
        api_call_fn: Function that takes (client, model, system_prompt, messages) and returns response
    """
    client = None
    client_loop = None

    config = config or ConcurrencyConfig()
    limiter = rate_limiter or AsyncRateLimiter(
        max_concurrent=config.max_concurrent,
        max_requests_per_second=config.max_requests_per_second,
    )

    async def model_fn(system_prompt: str, messages: List[Dict]) -> str:
        nonlocal client, client_loop

        current_loop = asyncio.get_running_loop()
        if client is None or client_loop is not current_loop:
            client = init_client(provider_name, client_class_path, API_KEY_VARS[provider_name])
            client_loop = current_loop

        async def call():
            async with limiter:
                return await api_call_fn(client, model, system_prompt, messages)

        try:
            return await async_retry_with_backoff(
                call,
                max_retries=config.retry_attempts,
                base_delay=config.retry_base_delay,
            )
        except Exception as e:
            return f"Error calling {provider_name}: {e}"

    return model_fn


async def _anthropic_call(client, model: str, system_prompt: str, messages: List[Dict]) -> str:
    """Anthropic API call (system prompt is separate parameter)."""
    formatted = [{"role": m.get("role", "user"), "content": m["content"]} for m in messages]
    response = await client.messages.create(
        model=model,
        max_tokens=TUTOR_MAX_TOKENS,
        system=system_prompt,
        messages=formatted,
    )
    return response.content[0].text


async def _openai_style_call(client, model: str, system_prompt: str, messages: List[Dict]) -> str:
    """OpenAI-style API call (system prompt in messages array)."""
    formatted = [{"role": "system", "content": system_prompt}] if system_prompt else []
    formatted.extend({"role": m.get("role", "user"), "content": m["content"]} for m in messages)
    response = await client.chat.completions.create(model=model, messages=formatted)
    return response.choices[0].message.content


def get_anthropic_async_provider(
    model: str = ANTHROPIC_DEFAULT_MODEL,
    rate_limiter: Optional[AsyncRateLimiter] = None,
    config: Optional[ConcurrencyConfig] = None,
) -> Callable:
    """Get async Anthropic provider function."""
    return _create_async_provider(
        "anthropic", "anthropic.AsyncAnthropic", model, rate_limiter, config, _anthropic_call
    )


def get_together_async_provider(
    model: str = TOGETHER_DEFAULT_MODEL,
    rate_limiter: Optional[AsyncRateLimiter] = None,
    config: Optional[ConcurrencyConfig] = None,
) -> Callable:
    """Get async Together AI provider function."""
    return _create_async_provider(
        "together", "together.AsyncTogether", model, rate_limiter, config, _openai_style_call
    )


def get_openai_async_provider(
    model: str = OPENAI_DEFAULT_MODEL,
    rate_limiter: Optional[AsyncRateLimiter] = None,
    config: Optional[ConcurrencyConfig] = None,
) -> Callable:
    """Get async OpenAI provider function."""
    return _create_async_provider(
        "openai", "openai.AsyncOpenAI", model, rate_limiter, config, _openai_style_call
    )


def get_async_provider(
    provider_name: str,
    model: Optional[str] = None,
    rate_limiter: Optional[AsyncRateLimiter] = None,
    config: Optional[ConcurrencyConfig] = None,
) -> Callable[[str, List[Dict]], str]:
    """
    Get an async model function for the specified provider.

    Args:
        provider_name: "together", "anthropic", or "openai"
        model: Model identifier (uses provider default if not specified)
        rate_limiter: Shared rate limiter instance
        config: Concurrency configuration

    Returns:
        Async model function: async (system_prompt, messages) -> response
    """
    provider_name = provider_name.lower()
    providers = {
        "together": (get_together_async_provider, TOGETHER_DEFAULT_MODEL),
        "anthropic": (get_anthropic_async_provider, ANTHROPIC_DEFAULT_MODEL),
        "openai": (get_openai_async_provider, OPENAI_DEFAULT_MODEL),
    }

    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Choose from: {list(providers.keys())}")

    factory, default_model = providers[provider_name]
    return factory(model or default_model, rate_limiter, config)


# Alias for backwards compatibility
get_provider = get_async_provider
