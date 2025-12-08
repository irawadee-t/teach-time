"""
LLM client with multi-provider support and caching.

Supports Together AI, Anthropic, and HuggingFace Transformers with SQLite-based caching
for reproducibility and efficiency.
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

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


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
    Client for calling LLMs with multi-provider support and caching.

    Supports:
    - Together AI API (Llama, Mistral, etc.)
    - Anthropic API (Claude models)
    - HuggingFace Transformers (SocraticLM, local models)
    - SQLite caching for reproducibility
    - Token counting and usage tracking
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        default_model: str = "CogBase-USTC/SocraticLM",
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

        # Initialize Anthropic client
        if ANTHROPIC_AVAILABLE and self.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        else:
            self.anthropic_client = None

        # Initialize HuggingFace model and tokenizer
        self.tokenizer = None
        self.model = None
        if TRANSFORMERS_AVAILABLE:
            try:
                if self.verbose:
                    print(f"Loading model: {default_model}")

                # Get HuggingFace token from environment or use cached token
                hf_token = os.getenv("HF_TOKEN") or None

                self.tokenizer = AutoTokenizer.from_pretrained(
                    default_model,
                    token=hf_token,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    default_model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token
                )
                if self.verbose:
                    print(f"Model loaded successfully on device: {self.model.device}")
            except Exception as e:
                if self.verbose:
                    print(f"Could not load HuggingFace model: {e}")
                self.tokenizer = None
                self.model = None

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
            Provider name: 'anthropic', 'together', or 'huggingface'
        """
        # Claude models use Anthropic
        if model.startswith("claude-"):
            return "anthropic"
        # HuggingFace models (contain / or are known HF models)
        if "/" in model or model == self.default_model:
            if self.model and self.tokenizer:
                return "huggingface"
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
        Call LLM with messages using the appropriate provider.

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

        if self.verbose:
            print(f"[{provider.upper()}] {model} (temp={temperature}, max_tokens={max_tokens})")

        # Route to appropriate provider
        if provider == "anthropic":
            response_text = self._call_anthropic(messages, model, temperature, max_tokens, stop)
        elif provider == "huggingface":
            response_text = self._call_huggingface(messages, model, temperature, max_tokens)
        else:  # together
            response_text = self._call_together(messages, model, temperature, max_tokens, stop)

        # Cache the response
        if self.enable_cache and self.cache:
            self.cache.set(messages, model, temperature, max_tokens, response_text)

        return response_text

    def _call_together(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]]
    ) -> str:
        """Call Together AI API."""
        if not self.together_client:
            if self.verbose:
                print("Warning: Together client not available")
            return "[Mock response - Together AI not configured]"

        try:
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
            print(f"Error calling Together AI: {e}")
            return f"[Error: {str(e)}]"

    def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]]
    ) -> str:
        """Call Anthropic API."""
        if not self.anthropic_client:
            if self.verbose:
                print("Warning: Anthropic client not available")
            return "[Mock response - Anthropic not configured]"

        try:
            # Extract system message if present
            system_message = None
            chat_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    chat_messages.append(msg)

            # Call Anthropic API
            if system_message:
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_message,
                    messages=chat_messages,
                    stop_sequences=stop if stop else anthropic.NOT_GIVEN,
                )
            else:
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=chat_messages,
                    stop_sequences=stop if stop else anthropic.NOT_GIVEN,
                )

            response_text = response.content[0].text

            # Track usage
            self.total_prompt_tokens += response.usage.input_tokens
            self.total_completion_tokens += response.usage.output_tokens
            self.total_api_calls += 1

            return response_text

        except Exception as e:
            print(f"Error calling Anthropic: {e}")
            return f"[Error: {str(e)}]"

    def _call_huggingface(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call local HuggingFace model."""
        if not self.model or not self.tokenizer:
            if self.verbose:
                print("Warning: HuggingFace model not loaded")
            return "[Mock response - Model not loaded]"

        try:
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            input_length = inputs.shape[1]

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.to(self.model.device),
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response (only the generated part, not the prompt)
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the assistant's response by removing the prompt
            prompt_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
            if full_response.startswith(prompt_text):
                response_text = full_response[len(prompt_text):].strip()
            else:
                response_text = full_response.strip()

            # Track usage (approximate token counts)
            output_length = outputs.shape[1]
            prompt_tokens = input_length
            completion_tokens = output_length - input_length

            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_api_calls += 1

            return response_text

        except Exception as e:
            print(f"Error calling HuggingFace model: {e}")
            import traceback
            traceback.print_exc()
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
    model: str = "CogBase-USTC/SocraticLM",
    enable_cache: bool = True,
    verbose: bool = False,
) -> LLMClient:
    """
    Create an LLM client with default settings.

    Args:
        model: HuggingFace model name to use
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
