"""LLM Engine with resilience and failover support."""

import base64
import logging
import time
from pathlib import Path
from typing import Any, Optional

from ..core.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Error from LLM provider."""
    pass


class AllProvidersFailedError(Exception):
    """All LLM providers failed."""
    pass


class LLMEngine:
    """Resilient LLM engine with retry and failover support.

    Integrates with UnifiedLLMClient when available, falls back to direct API calls.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the LLM engine.

        Args:
            config: LLM configuration. Uses defaults if not provided.
        """
        self.config = config or LLMConfig()
        self._clients: dict[str, Any] = {}
        self._unified_client = None

        # Try to load UnifiedLLMClient
        self._init_unified_client()

    def _init_unified_client(self) -> None:
        """Initialize UnifiedLLMClient if available."""
        try:
            # Try to import from the UnifiedLLMClient package
            import sys
            sys.path.insert(0, str(Path.home() / "Development/Scripts/UnifiedLLMClient"))
            from llm_client import LLMClientFactory
            self._unified_client = LLMClientFactory
            logger.info("UnifiedLLMClient loaded successfully")
        except ImportError:
            logger.warning("UnifiedLLMClient not available, using direct API calls")
            self._unified_client = None

    def _get_client(self, provider: str) -> Any:
        """Get or create a client for a provider.

        Args:
            provider: Provider name (claude, openai, etc.)

        Returns:
            Client instance.
        """
        if provider not in self._clients:
            if self._unified_client:
                self._clients[provider] = self._unified_client.create(provider=provider)
            else:
                self._clients[provider] = self._create_direct_client(provider)

        return self._clients[provider]

    def _create_direct_client(self, provider: str) -> Any:
        """Create a direct API client for a provider.

        Args:
            provider: Provider name.

        Returns:
            Client wrapper.
        """
        import os

        if provider == "claude":
            try:
                import anthropic
                return AnthropicWrapper(
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                    model=self.config.model
                )
            except ImportError:
                raise LLMProviderError("anthropic package not installed")

        elif provider == "openai":
            try:
                import openai
                return OpenAIWrapper(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model="gpt-4-turbo"
                )
            except ImportError:
                raise LLMProviderError("openai package not installed")

        else:
            raise LLMProviderError(f"Unknown provider: {provider}")

    def complete(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Complete a prompt with retry and failover.

        Args:
            prompt: The prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text.

        Raises:
            AllProvidersFailedError: If all providers fail.
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        for provider in self.config.failover_chain:
            for attempt in range(self.config.max_retries):
                try:
                    client = self._get_client(provider)

                    if self._unified_client:
                        return client.generate(
                            prompt,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                    else:
                        return client.complete(
                            prompt,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )

                except Exception as e:
                    logger.warning(f"{provider} attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"{provider} failed after {self.config.max_retries} retries")
                        break

        raise AllProvidersFailedError("All LLM providers failed")

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Send chat messages with retry and failover.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated response.

        Raises:
            AllProvidersFailedError: If all providers fail.
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        for provider in self.config.failover_chain:
            for attempt in range(self.config.max_retries):
                try:
                    client = self._get_client(provider)

                    if self._unified_client:
                        return client.chat(
                            messages,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                    else:
                        return client.chat(
                            messages,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )

                except Exception as e:
                    logger.warning(f"{provider} attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        break

        raise AllProvidersFailedError("All LLM providers failed")

    def analyze_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 1024
    ) -> str:
        """Analyze an image with a vision model.

        Args:
            image_path: Path to the image file.
            prompt: Prompt describing what to analyze.
            max_tokens: Maximum tokens to generate.

        Returns:
            Analysis result.
        """
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        # Determine media type
        suffix = Path(image_path).suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        media_type = media_types.get(suffix, "image/png")

        # Use Claude for vision (best support)
        try:
            import anthropic
            import os

            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            return message.content[0].text

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise LLMProviderError(f"Image analysis failed: {e}")


class AnthropicWrapper:
    """Wrapper for direct Anthropic API calls."""

    def __init__(self, api_key: str, model: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, max_tokens: int, temperature: float) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def chat(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages
        )
        return message.content[0].text


class OpenAIWrapper:
    """Wrapper for direct OpenAI API calls."""

    def __init__(self, api_key: str, model: str):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, max_tokens: int, temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def chat(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages
        )
        return response.choices[0].message.content
