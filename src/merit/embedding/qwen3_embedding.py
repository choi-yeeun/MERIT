from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class Qwen3EmbeddingModel:
    """Wrapper for Qwen3 Embedding Model"""

    def __init__(
        self, model_name: str = "Qwen/Qwen3-Embedding-4B", device: str = "auto"
    ):
        self.model_name = model_name
        self.device = device

        # flash_attention_2 only works on CUDA; fall back to sdpa on CPU
        attn_impl = "sdpa" if device == "cpu" else "flash_attention_2"
        model_kwargs = {
            "attn_implementation": attn_impl,
            "dtype": "auto",
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"
            self.model = SentenceTransformer(
                model_name,
                model_kwargs=model_kwargs,
                tokenizer_kwargs={"padding_side": "left"},
            )
        else:
            # If a specific device is provided, don't use device_map in model_kwargs.
            # SentenceTransformer will handle moving the model to the specified device.
            self.model = SentenceTransformer(
                model_name,
                model_kwargs=model_kwargs,
                tokenizer_kwargs={"padding_side": "left"},
                device=device,
            )

    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 256,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode text into embeddings"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=show_progress_bar
        )
        return embeddings

    def encode(self, content: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Universal encode method for text"""
        return self.encode_text(content, **kwargs)
