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
    Client for calling LLMs via Together AI with caching.

    Supports:
    - Together AI API
    - SQLite caching for reproducibility
    - Token counting and usage tracking
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        enable_cache: bool = True,
        cache_dir: str = ".cache",
        verbose: bool = False,
    ):
        """
        Args:
            api_key: Together AI API key (reads from env var if not provided)
            default_model: Default model to use
            enable_cache: Whether to use caching
            cache_dir: Directory for cache database
            verbose: Whether to print debug information
        """
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key and TOGETHER_AVAILABLE:
            print("Warning: TOGETHER_API_KEY not found in environment")

        self.default_model = default_model
        self.enable_cache = enable_cache
        self.verbose = verbose

        # Initialize cache
        if self.enable_cache:
            self.cache = LLMCache(cache_dir)
        else:
            self.cache = None

        # Initialize Together client
        if TOGETHER_AVAILABLE and self.api_key:
            self.client = Together(api_key=self.api_key)
        else:
            self.client = None
            if self.verbose:
                print("Together client not initialized (API key missing or package not installed)")

        # Usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_hits = 0
        self.total_api_calls = 0

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

        # Make API call
        if not self.client:
            # Fallback: return a placeholder for testing without API key
            response_text = f"[Mock response - API key not configured]"
            if self.verbose:
                print(f"Warning: Using mock response (no API client available)")
            return response_text

        try:
            if self.verbose:
                print(f"[API call] {model} (temp={temperature}, max_tokens={max_tokens})")

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )

            # Extract response text
            response_text = response.choices[0].message.content

            # Track usage
            if hasattr(response, 'usage'):
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens

            self.total_api_calls += 1

            # Cache the response
            if self.enable_cache and self.cache:
                self.cache.set(messages, model, temperature, max_tokens, response_text)

            return response_text

        except Exception as e:
            print(f"Error calling LLM: {e}")
            # Return error placeholder
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
