"""
Unified interface for multiple LLM providers
"""

from typing import Any, Dict, List, Optional, Union


class LLMModel:
    """
    Unified interface for interacting with different LLM providers.
    This class wraps provider-specific models and exposes a consistent API for text generation and batch generation.
    """

    def __init__(self, model_name: str, *, provider: Optional[str] = None, **kwargs):
        """
        Initialize the LLMModel with the specified model name and optional provider.

        Args:
            model_name (str): The name of the model to use (e.g., "gpt-5").
            provider (Optional[str]): The name of the LLM provider. If None, will auto-detect based on model_name.
        """
        self.model_name = model_name
        self.provider = self._detect_provider(model_name, provider)
        self.model = self._init_model(**kwargs)

    def _detect_provider(self, model_name: str, provider: Optional[str]) -> str:
        """
        Auto-detect the provider based on the model name if provider is not specified.

        Args:
            model_name (str): Model name to use for auto-detection
            provider (Optional[str]): Explicitly specified provider

        Returns:
            str: The detected or specified provider name
        """
        if provider is not None:
            return provider.lower()

        model_name_lower = model_name.lower()

        # Auto-detect based on model name patterns
        if "gpt" in model_name_lower:
            return "openai"
        elif "gemini" in model_name_lower or "vertex-gemini" in model_name_lower:
            return "gemini"
        elif "qwen3" in model_name_lower or "qwen25" in model_name_lower:
            return "qwen3vl"
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _init_model(self, **kwargs):
        if self.provider == "openai":
            from .openai_gpt import OpenAIModelFast

            return OpenAIModelFast(model_name=self.model_name, **kwargs)
        elif self.provider == "gemini":
            from .google_gemini import GeminiModel

            return GeminiModel(model_name=self.model_name, **kwargs)
        elif self.provider == "qwen3vl":
            from .qwen3vl import Qwen3VLModel

            return Qwen3VLModel(model_name=self.model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def generate_async(
        self, prompt: Union[str, List[Dict[str, Any]]], **kwargs
    ) -> Any:
        """
        Asynchronous generate method for OpenAI and Gemini models.
        """
        if self.provider in ("openai", "gemini"):
            return await self.model.generate_async(prompt, **kwargs)
        else:
            # Fallback to synchronous generate in a thread if not implemented
            import asyncio

            return await asyncio.to_thread(self.generate, prompt, **kwargs)

    def generate(self, prompt: Union[str, List[Dict[str, Any]]], **kwargs) -> Any:
        """
        Unified generate method for all LLMs.
        """
        if self.provider in ("openai", "gemini", "qwen3vl"):
            return self.model.generate(prompt, **kwargs)
        else:
            raise NotImplementedError(
                f"Model {self.provider} does not support text generation."
            )

    def generate_batch(
        self, batch_prompts: List[Union[str, List[Dict[str, Any]]]], **kwargs
    ) -> List[Any]:
        """
        Unified batch generate method for all LLMs.
        """
        if self.provider in ("openai", "gemini", "qwen3vl"):
            return self.model.generate_batch(batch_prompts, **kwargs)
        else:
            raise NotImplementedError(
                f"Model {self.provider} does not support batch generation."
            )

    def __repr__(self):
        return f"LLMModel(provider={self.provider}, model_name={self.model_name}, kwargs={self.model.kwargs})"
