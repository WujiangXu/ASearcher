#!/usr/bin/env python3
"""
OpenAI API Client for ASearcher Data Generation

This module provides an OpenAI-compatible client that can replace
the SGLangAPIClient in qa_synthesis_agent.py, enabling the use of
GPT models (GPT-4, GPT-4-turbo, GPT-3.5) instead of local Qwen models.

Usage:
    from openai_client import OpenAIAPIClient

    # Replace SGLangAPIClient with OpenAIAPIClient
    reasoning_client = OpenAIAPIClient(model="gpt-4-turbo-preview")
    instruct_client = OpenAIAPIClient(model="gpt-4")

    # Use exactly like SGLangAPIClient
    agent = ConstructQAAgent(all_links, pages, search_client)
    asyncio.run(main(agent, reasoning_client, instruct_client, save_path))

Author: ASearcher Team
"""

import os
import time
import asyncio
import json
from typing import Dict, Any, Optional, List

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: OpenAI library not installed")
    print("Install with: pip install openai")
    import sys
    sys.exit(1)


class OpenAIAPIClient:
    """
    OpenAI API client that mimics SGLangAPIClient interface.

    Compatible with qa_synthesis_agent.py without code changes.
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 600.0
    ):
        """
        Initialize OpenAI API client.

        Args:
            model: OpenAI model name
                - "gpt-4-turbo-preview" (recommended for reasoning, like QwQ-32B)
                - "gpt-4" (high quality, expensive)
                - "gpt-3.5-turbo" (fast, cheap, like Qwen2.5-72B-Instruct)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            base_url: Optional custom API base URL
            organization: Optional OpenAI organization ID
            max_retries: Number of retries on failure
            timeout: Request timeout in seconds
        """
        self.model = model

        # Get API key from env if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenAI API key not found. Either:\n"
                    "1. Pass api_key parameter\n"
                    "2. Set OPENAI_API_KEY environment variable"
                )

        # Initialize async client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            max_retries=max_retries,
            timeout=timeout
        )

        print(f"✓ OpenAI client initialized with model: {model}")

        # Track usage stats
        self.total_tokens = 0
        self.total_requests = 0

    async def __aenter__(self):
        """Context manager entry (compatibility with SGLangAPIClient)."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (compatibility with SGLangAPIClient)."""
        await self.client.close()

    async def async_generate(
        self,
        prompt: str,
        sampling_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate text using OpenAI API (compatible with SGLangAPIClient).

        Args:
            prompt: Input prompt (can include chat template formatting)
            sampling_kwargs: Sampling parameters
                - temperature: float (0-2)
                - top_p: float (0-1)
                - max_new_tokens: int (mapped to max_tokens)
                - n: int (number of completions)
                - stop: list of stop sequences

        Returns:
            Dict with "text" key containing generated text(s)
        """
        # Extract parameters
        temperature = sampling_kwargs.get("temperature", 0.8)
        top_p = sampling_kwargs.get("top_p", 0.95)
        max_tokens = sampling_kwargs.get("max_new_tokens", 4096)
        n = sampling_kwargs.get("n", 1)
        stop = sampling_kwargs.get("stop", None)

        # Handle stop sequences
        if stop:
            # Filter out token IDs (OpenAI uses strings)
            stop = [s for s in stop if isinstance(s, str)]
            if len(stop) == 0:
                stop = None

        # Parse prompt if it contains chat template
        messages = self._parse_prompt_to_messages(prompt)

        # Call OpenAI API
        try:
            start_time = time.time()

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=n,
                stop=stop
            )

            latency = time.time() - start_time

            # Track stats
            self.total_tokens += response.usage.total_tokens
            self.total_requests += 1

            # Extract generated text(s)
            if n == 1:
                text = response.choices[0].message.content
            else:
                text = [choice.message.content for choice in response.choices]

            # Log stats
            print(f"[OpenAI] Generated {len(text) if isinstance(text, list) else 1} completion(s)")
            print(f"  Tokens: {response.usage.total_tokens} "
                  f"(prompt: {response.usage.prompt_tokens}, "
                  f"completion: {response.usage.completion_tokens})")
            print(f"  Latency: {latency:.2f}s")
            print(f"  Total usage: {self.total_tokens} tokens, {self.total_requests} requests")

            return {"text": text}

        except Exception as e:
            print(f"ERROR: OpenAI API call failed: {e}")
            # Return empty to avoid crashing the generation loop
            if n == 1:
                return {"text": ""}
            else:
                return {"text": ["" for _ in range(n)]}

    def _parse_prompt_to_messages(self, prompt: str) -> List[Dict[str, str]]:
        """
        Parse prompt into OpenAI messages format.

        Handles prompts from Qwen chat template:
        - Extracts content after <|im_start|>user and before <|im_start|>assistant
        - Converts to OpenAI format: [{"role": "user", "content": "..."}]

        Args:
            prompt: Raw prompt string (may contain chat template markers)

        Returns:
            List of message dicts for OpenAI API
        """
        # Check if prompt contains Qwen chat template markers
        if "<|im_start|>" in prompt and "<|im_end|>" in prompt:
            # Extract user content
            if "<|im_start|>user\n" in prompt:
                user_content = prompt.split("<|im_start|>user\n")[-1]
                if "<|im_end|>" in user_content:
                    user_content = user_content.split("<|im_end|>")[0]

                return [{"role": "user", "content": user_content.strip()}]

        # Otherwise, treat entire prompt as user message
        return [{"role": "user", "content": prompt}]

    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics."""
        return {
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests
        }

    def estimate_cost(self, pricing: Optional[Dict[str, float]] = None) -> float:
        """
        Estimate API cost based on token usage.

        Args:
            pricing: Optional pricing dict with "input" and "output" per 1K tokens
                Default pricing for gpt-4-turbo-preview:
                - Input: $0.01 per 1K tokens
                - Output: $0.03 per 1K tokens

        Returns:
            Estimated cost in USD
        """
        if pricing is None:
            # Default pricing (approximate, may vary)
            if "gpt-4-turbo" in self.model:
                pricing = {"input": 0.01, "output": 0.03}  # per 1K tokens
            elif "gpt-4" in self.model:
                pricing = {"input": 0.03, "output": 0.06}
            elif "gpt-3.5-turbo" in self.model:
                pricing = {"input": 0.0005, "output": 0.0015}
            else:
                print(f"WARNING: Unknown model {self.model}, using gpt-4-turbo pricing")
                pricing = {"input": 0.01, "output": 0.03}

        # Rough estimate (assume 2:1 output:input ratio)
        input_tokens = self.total_tokens * 0.33
        output_tokens = self.total_tokens * 0.67

        cost = (
            (input_tokens / 1000) * pricing["input"] +
            (output_tokens / 1000) * pricing["output"]
        )

        return cost


# Example usage and testing
async def test_openai_client():
    """Test OpenAI client with sample prompts."""
    print("="*60)
    print("Testing OpenAI API Client")
    print("="*60)

    # Initialize client
    client = OpenAIAPIClient(model="gpt-3.5-turbo")  # Use cheap model for testing

    async with client:
        # Test 1: Simple generation
        print("\nTest 1: Simple generation")
        prompt = "What is the capital of France? Answer in one word."
        result = await client.async_generate(
            prompt=prompt,
            sampling_kwargs={"temperature": 0.7, "max_new_tokens": 50, "n": 1}
        )
        print(f"Result: {result['text']}")

        # Test 2: Multiple completions
        print("\nTest 2: Multiple completions (n=3)")
        result = await client.async_generate(
            prompt=prompt,
            sampling_kwargs={"temperature": 0.8, "max_new_tokens": 50, "n": 3}
        )
        print(f"Results:")
        for i, text in enumerate(result['text'], 1):
            print(f"  {i}. {text}")

        # Test 3: With Qwen chat template format
        print("\nTest 3: Qwen chat template format")
        qwen_prompt = "<|im_start|>user\nExplain quantum computing in one sentence.<|im_end|>\n<|im_start|>assistant\n"
        result = await client.async_generate(
            prompt=qwen_prompt,
            sampling_kwargs={"temperature": 0.7, "max_new_tokens": 100, "n": 1}
        )
        print(f"Result: {result['text']}")

    # Print stats
    print("\n" + "="*60)
    print("Usage Statistics")
    print("="*60)
    stats = client.get_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Estimated cost: ${client.estimate_cost():.4f}")


if __name__ == "__main__":
    print("""
OpenAI API Client for ASearcher Data Generation

Usage:
    1. Set your OpenAI API key:
       export OPENAI_API_KEY='your-api-key-here'

    2. In qa_synthesis_agent.py, replace:
       from llm_utils import SGLangAPIClient

       with:
       from openai_client import OpenAIAPIClient as SGLangAPIClient

    3. Run as normal:
       reasoning_client = SGLangAPIClient(model="gpt-4-turbo-preview")
       instruct_client = SGLangAPIClient(model="gpt-3.5-turbo")

Recommended model mapping:
    - QwQ-32B (reasoning) → gpt-4-turbo-preview or gpt-4
    - Qwen2.5-72B-Instruct → gpt-3.5-turbo (faster, cheaper)

Cost estimates (per 1000 questions generated):
    - gpt-4-turbo: ~$50-100 (high quality)
    - gpt-3.5-turbo: ~$5-10 (good quality, 10× cheaper)
    """)

    # Run test if API key is available
    if os.getenv("OPENAI_API_KEY"):
        print("\nRunning tests...")
        asyncio.run(test_openai_client())
    else:
        print("\nTo run tests, set OPENAI_API_KEY environment variable")
        print("export OPENAI_API_KEY='your-key-here'")
