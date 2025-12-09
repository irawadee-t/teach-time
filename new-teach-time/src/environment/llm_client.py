"""
Minimal LLM client for Together API.

Simple, synchronous client for calling language models via Together's OpenAI-compatible API.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from .config import TOGETHER_API_BASE_URL, TOGETHER_API_KEY_ENV


def _load_dotenv():
    """Load .env file from project root if it exists."""
    # Try to find .env in current dir or parent dirs
    current = Path.cwd()
    for _ in range(5):  # Look up to 5 levels
        env_file = current / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
            return True
        current = current.parent
    return False


# Auto-load .env on module import
_load_dotenv()


@dataclass
class LLMResponse:
    """Response from LLM API call."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"


class LLMClient:
    """
    Synchronous client for Together API.
    
    Uses the OpenAI-compatible chat completions endpoint.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = TOGETHER_API_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            api_key: Together API key (reads from env if not provided)
            base_url: API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Initial delay between retries (doubles each retry)
        """
        self.api_key = api_key or os.environ.get(TOGETHER_API_KEY_ENV)
        if not self.api_key:
            raise ValueError(
                f"API key required. Set {TOGETHER_API_KEY_ENV} environment variable "
                "or pass api_key parameter."
            )
        
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
    
    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        """
        Send chat completion request.
        
        Args:
            model: Model ID (e.g., "Qwen/Qwen2.5-7B-Instruct")
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Optional stop sequences
            **kwargs: Additional parameters passed to the API
            
        Returns:
            Generated text content
        """
        response = self.chat_with_metadata(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs,
        )
        return response.content
    
    def chat_with_metadata(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send chat completion request and return full response with metadata.
        
        Args:
            model: Model ID
            messages: List of message dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Optional stop sequences
            **kwargs: Additional API parameters
            
        Returns:
            LLMResponse with content and metadata
        """
        # PRE-FLIGHT: Validate message alternation
        self._validate_message_alternation(messages, model)
        
        url = f"{self.base_url}/chat/completions"
        
        # DEBUG: Print messages being sent (uncomment to inspect format)
        # import json
        # print(f"\n{'='*70}\nDEBUG LLM MESSAGES ({model[:40]}):\n{'='*70}")
        # for i, m in enumerate(messages):
        #     role = m.get('role', '?')
        #     content = m.get('content', '')[:300].replace('\n', '\\n')
        #     print(f"[{i}] {role.upper():10} | {content}...")
        # print(f"{'='*70}\n")
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if stop:
            payload["stop"] = stop
        
        payload.update(kwargs)
        
        # Retry loop
        last_error = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                
                data = response.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})
                
                return LLMResponse(
                    content=choice["message"]["content"],
                    model=data.get("model", model),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    finish_reason=choice.get("finish_reason", "stop"),
                )
                
            except requests.exceptions.RequestException as e:
                last_error = e
                # Try to get more error details
                error_details = ""
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_details = f" Response: {e.response.text}"
                    except:
                        pass
                
                if attempt < self.max_retries - 1:
                    print(f"  [Retry {attempt + 1}/{self.max_retries}] {e}{error_details[:200]}")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
        
        # Include error details in final exception
        error_msg = f"API call failed after {self.max_retries} attempts: {last_error}"
        if hasattr(last_error, 'response') and last_error.response is not None:
            try:
                error_msg += f"\nResponse: {last_error.response.text[:500]}"
            except:
                pass
        raise RuntimeError(error_msg)
    
    def _validate_message_alternation(self, messages: list[dict], model: str) -> None:
        """
        Validate that messages follow proper alternation pattern.
        
        Expected: system (optional) → user → assistant → user → assistant → ...
        
        Rules:
        1. First non-system message must be 'user'
        2. After 'user' must come 'assistant' (or end)
        3. After 'assistant' must come 'user'
        4. No consecutive same roles (except system at start)
        """
        if not messages:
            return
        
        # Find where content messages start (skip system)
        start_idx = 0
        for i, msg in enumerate(messages):
            if msg.get("role") != "system":
                start_idx = i
                break
        else:
            return  # Only system messages, that's fine
        
        content_messages = messages[start_idx:]
        if not content_messages:
            return
        
        # First content message must be 'user'
        if content_messages[0].get("role") != "user":
            print(f"[WARN] Message validation: First content message should be 'user', got '{content_messages[0].get('role')}'")
        
        # Check alternation
        prev_role = None
        for i, msg in enumerate(content_messages):
            role = msg.get("role")
            
            if prev_role is not None and role == prev_role:
                # Log warning but don't fail - the API might still work
                print(f"[WARN] Message validation failed for {model[:30]}:")
                print(f"       Consecutive '{role}' messages at positions {start_idx + i - 1} and {start_idx + i}")
                print(f"       Messages should strictly alternate user/assistant")
                # Show the offending messages
                if i > 0:
                    prev_content = content_messages[i-1].get('content', '')[:100]
                    print(f"       Prev: {prev_content}...")
                curr_content = msg.get('content', '')[:100]
                print(f"       Curr: {curr_content}...")
                break
            
            prev_role = role
    
    def close(self):
        """Close the session."""
        self._session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# Factory function for easy client creation
def create_llm_client(
    api_key: Optional[str] = None,
    base_url: str = TOGETHER_API_BASE_URL,
) -> LLMClient:
    """Create an LLM client with default settings."""
    return LLMClient(api_key=api_key, base_url=base_url)

