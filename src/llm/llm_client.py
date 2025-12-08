"""
LLM client with Together AI integration and caching.

Provides a clean abstraction over Together AI API with SQLite-based caching
for reproducibility and cost savings.
"""

import os
import json
import hashlib
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import time

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    print("Warning: 'together' package not installed. Install with: pip install together")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: 'anthropic' package not installed. Install with: pip install anthropic")


@dataclass
class LLMResponse:
    """Structured LLM response."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached: bool = False


class LLMCache:
    """SQLite-based cache for LLM responses."""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "llm_cache.db"
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                model TEXT NOT NULL,
                temperature REAL NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _compute_cache_key(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Compute cache key from request parameters."""
        # Serialize messages to stable JSON
        messages_str = json.dumps(messages, sort_keys=True)
        key_input = f"{messages_str}|{model}|{temperature}|{max_tokens}"
        return hashlib.sha256(key_input.encode()).hexdigest()

    def get(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Get cached response if it exists."""
        cache_key = self._compute_cache_key(messages, model, temperature, max_tokens)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT response FROM llm_cache WHERE cache_key = ?",
            (cache_key,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return row[0]
        return None

    def set(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
        response: str
    ):
        """Store response in cache."""
        cache_key = self._compute_cache_key(messages, model, temperature, max_tokens)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO llm_cache
            (cache_key, response, model, temperature, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (cache_key, response, model, temperature, time.time()))
        conn.commit()
        conn.close()

    def clear(self):
        """Clear all cached responses."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM llm_cache")
        conn.commit()
        conn.close()

    def size(self) -> int:
        """Get number of cached responses."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM llm_cache")
        count = cursor.fetchone()[0]
        conn.close()
        return count


class LLMClient:
    """
    Client for calling LLMs via Together AI or Anthropic with caching.

    Supports:
    - Together AI API
    - Anthropic API (Claude models)
    - SQLite caching for reproducibility
    - Token counting and usage tracking
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        default_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        enable_cache: bool = True,
        cache_dir: str = ".cache",
        verbose: bool = False,
    ):
        """
        Args:
            api_key: Together AI API key (reads from TOGETHER_API_KEY if not provided)
            anthropic_api_key: Anthropic API key (reads from ANTHROPIC_API_KEY if not provided)
            default_model: Default model to use
            enable_cache: Whether to use caching
            cache_dir: Directory for cache database
            verbose: Whether to print debug information
        """
        self.together_api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

        self.default_model = default_model
        self.enable_cache = enable_cache
        self.verbose = verbose

        # Initialize cache
        if self.enable_cache:
            self.cache = LLMCache(cache_dir)
        else:
            self.cache = None

        # Initialize Together client
        if TOGETHER_AVAILABLE and self.together_api_key:
            self.together_client = Together(api_key=self.together_api_key)
        else:
            self.together_client = None
            if self.verbose and not self.together_api_key:
                print("Together client not initialized (API key missing or package not installed)")

        # Initialize Anthropic client
        if ANTHROPIC_AVAILABLE and self.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        else:
            self.anthropic_client = None
            if self.verbose and not self.anthropic_api_key:
                print("Anthropic client not initialized (API key missing or package not installed)")

        # Usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_hits = 0
        self.total_api_calls = 0

    def _detect_provider(self, model: str) -> str:
        """
        Detect which provider to use based on model name.

        Args:
            model: Model name

        Returns:
            Provider name: 'anthropic' or 'together'
        """
        # Claude models use Anthropic
        if model.startswith("claude-"):
            return "anthropic"
        # Default to Together for all other models
        return "together"

    def call(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Call LLM with messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (uses default if not specified)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Optional stop sequences

        Returns:
            Generated text content
        """
        model = model or self.default_model

        # Check cache first
        if self.enable_cache and self.cache:
            cached_response = self.cache.get(messages, model, temperature, max_tokens)
            if cached_response:
                if self.verbose:
                    print(f"[Cache hit] {model}")
                self.total_cached_hits += 1
                return cached_response

        # Detect provider
        provider = self._detect_provider(model)

        # Route to appropriate provider
        if provider == "anthropic":
            response_text = self._call_anthropic(messages, model, temperature, max_tokens, stop)
        else:
            response_text = self._call_together(messages, model, temperature, max_tokens, stop)

        # Cache the response
        if self.enable_cache and self.cache and not response_text.startswith("[Error"):
            self.cache.set(messages, model, temperature, max_tokens, response_text)

        return response_text

    def _call_together(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> str:
        """Call Together AI API."""
        if not self.together_client:
            if self.verbose:
                print(f"Warning: Using mock response (Together client not available)")
            return "[Mock response - Together API key not configured]"

        try:
            if self.verbose:
                print(f"[Together API call] {model} (temp={temperature}, max_tokens={max_tokens})")

            response = self.together_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )

            response_text = response.choices[0].message.content

            # Track usage
            if hasattr(response, 'usage'):
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens

            self.total_api_calls += 1
            return response_text

        except Exception as e:
            print(f"Error calling Together API: {e}")
            return f"[Error: {str(e)}]"

    def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> str:
        """Call Anthropic API."""
        if not self.anthropic_client:
            if self.verbose:
                print(f"Warning: Using mock response (Anthropic client not available)")
            return "[Mock response - Anthropic API key not configured]"

        try:
            if self.verbose:
                print(f"[Anthropic API call] {model} (temp={temperature}, max_tokens={max_tokens})")

            # Anthropic requires system message to be separate
            system_message = None
            api_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    api_messages.append(msg)

            # Build request params
            params = {
                "model": model,
                "messages": api_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if system_message:
                params["system"] = system_message

            if stop:
                params["stop_sequences"] = stop

            response = self.anthropic_client.messages.create(**params)

            response_text = response.content[0].text

            # Track usage
            if hasattr(response, 'usage'):
                self.total_prompt_tokens += response.usage.input_tokens
                self.total_completion_tokens += response.usage.output_tokens

            self.total_api_calls += 1
            return response_text

        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return f"[Error: {str(e)}]"

    def call_with_response_object(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Call LLM and return structured response object with metadata.

        Args:
            messages: List of message dicts
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Optional stop sequences

        Returns:
            LLMResponse object with content and metadata
        """
        model = model or self.default_model

        # Check cache
        cached = False
        if self.enable_cache and self.cache:
            cached_response = self.cache.get(messages, model, temperature, max_tokens)
            if cached_response:
                cached = True
                content = cached_response
                # Estimate token counts (rough approximation)
                prompt_tokens = sum(len(m['content'].split()) for m in messages)
                completion_tokens = len(content.split())
            else:
                content = self.call(messages, model, temperature, max_tokens, stop)
                # These will be updated from actual API response
                prompt_tokens = 0
                completion_tokens = 0
        else:
            content = self.call(messages, model, temperature, max_tokens, stop)
            prompt_tokens = 0
            completion_tokens = 0

        return LLMResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cached=cached,
        )

    def get_usage_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            "total_api_calls": self.total_api_calls,
            "total_cached_hits": self.total_cached_hits,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "cache_size": self.cache.size() if self.cache else 0,
        }

    def clear_cache(self):
        """Clear the LLM response cache."""
        if self.cache:
            self.cache.clear()
            if self.verbose:
                print("Cache cleared")


# Convenience function for quick usage
def create_llm_client(
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    enable_cache: bool = True,
    verbose: bool = False,
) -> LLMClient:
    """
    Create an LLM client with default settings.

    Args:
        model: Model name to use
        enable_cache: Whether to enable caching
        verbose: Whether to print debug info

    Returns:
        Configured LLMClient instance
    """
    return LLMClient(
        default_model=model,
        enable_cache=enable_cache,
        verbose=verbose,
    )
