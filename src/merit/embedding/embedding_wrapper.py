from typing import List, Optional, Union

import numpy as np


class EmbeddingModel:
    """Text embedding wrapper using Qwen3-Embedding for key-based retrieval."""

    def __init__(
        self,
        text_model_name: str = "Qwen/Qwen3-Embedding-4B",
        device: str = "cuda",
    ):
        self.device = device
        self._text_model = None
        self.text_model_name = text_model_name

    @property
    def text_model(self):
        """Lazy loading of text model"""
        if self._text_model is None:
            from .qwen3_embedding import Qwen3EmbeddingModel

            self._text_model = Qwen3EmbeddingModel(
                model_name=self.text_model_name, device=self.device
            )
        return self._text_model

    def load_model(self, model_type: Optional[str] = None):
        """Load embedding models based on specified type"""
        if model_type is None or model_type == "text":
            _ = self.text_model
        else:
            raise ValueError(
                f"Invalid model_type: {model_type}. Choose from None or 'text'"
            )

    def encode_text(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode text using Qwen3 model"""
        return self.text_model.encode_text(texts, **kwargs)

    def encode(
        self,
        content: Union[str, List[str]],
        modality: str = "text",
        **kwargs,
    ) -> np.ndarray:
        if modality == "text":
            return self.encode_text(content, **kwargs)
        else:
            raise ValueError(f"Unsupported modality: {modality}.")
