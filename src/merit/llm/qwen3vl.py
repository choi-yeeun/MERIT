"""
Qwen3VL Model Wrapper with comprehensive video and image processing capabilities.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from decord import cpu, VideoReader
from PIL import Image
from pydantic import ValidationError
from qwen_vl_utils import process_vision_info

# Patch: PytorchGELUTanh was renamed to GELUTanh in transformers 4.57+
# AWQ models still reference the old name during from_pretrained
import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh") and hasattr(_act, "GELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from .utils import dynamic_retry_decorator

# Configure logging
logger = logging.getLogger(__name__)

# Model configuration
MODEL_DICT = {
    # Qwen3-VL series
    "qwen3vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3vl-30b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "qwen3vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
    # Qwen2.5-VL series
    "qwen25vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen25vl-32b-awq": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    "qwen25vl-72b-awq": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
}


class Qwen3VLModelError(Exception):
    """Custom exception for Qwen3VL model operations."""

    pass


class Qwen3VLModel:
    """
    Qwen3VL model wrapper with video and image processing capabilities.
    """

    def __init__(
        self,
        model_name: str,
        max_retries: int = 3,
        max_size: Tuple[int, int] = (512, 512),
        max_size_video: Tuple[int, int] = (256, 256),
        quality: int = 85,
        fps: Optional[int] = None,
        nframes: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Qwen3VL model wrapper.

        Args:
            model_name: Name of the Qwen3VL model to use
            max_retries: Maximum number of retry attempts
            max_size: Maximum size for image thumbnails
            max_size_video: Maximum size for video frames
            quality: JPEG quality for encoding (1-100)
            fps: Frames per second for video sampling
            nframes: Number of frames to sample from video
            **kwargs: Additional arguments for generation

        Raises:
            Qwen3VLModelError: If model initialization fails
            ValueError: If both fps and nframes are provided
        """
        # Validate parameters
        if fps is not None and nframes is not None:
            raise ValueError(
                "Cannot provide both 'fps' and 'nframes'. Please choose one for video sampling."
            )

        if model_name not in MODEL_DICT:
            raise ValueError(
                f"Unsupported model: {model_name}. Available: {list(MODEL_DICT.keys())}"
            )

        # Set instance attributes
        self.model_name = MODEL_DICT[model_name]
        self.max_retries = max(1, max_retries)
        self.max_size = max_size
        self.max_size_video = max_size_video
        self.quality = max(1, min(100, quality))  # Clamp quality between 1-100
        self.fps = fps
        self.nframes = nframes

        # Extract device configuration so it is not passed to generate
        self.device_map = kwargs.pop("device_map", kwargs.pop("device", "cuda:0"))
        
        # Filter out GPT-specific arguments that might be passed from LLMModel
        kwargs.pop("service_tier", None)
        kwargs.pop("timeout", None)
        
        self.kwargs = kwargs
        if "seed" not in self.kwargs:
            self.kwargs["seed"] = 42
        if "temperature" not in self.kwargs:
            self.kwargs["temperature"] = 0.0
            
        # Handle greedy decoding (temperature=0.0)
        if self.kwargs.get("temperature") == 0.0:
            self.kwargs["do_sample"] = False
            self.kwargs.pop("temperature")
        elif "temperature" in self.kwargs:
            self.kwargs["do_sample"] = True

        # Set seed and deterministic settings for reproducibility
        self._setup_reproducibility(self.kwargs["seed"])

        # Determine image_patch_size from model architecture
        # Qwen2.5-VL uses patch_size=14, Qwen3-VL uses patch_size=16
        if "Qwen2.5-VL" in self.model_name:
            self.image_patch_size = 14
        else:
            self.image_patch_size = 16

        # Initialize model
        try:
            self._init_model()
            logger.info(
                f"Initialized Qwen3VLModel with {self.model_name} "
                f"(image_patch_size={self.image_patch_size})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Qwen3VL model: {e}")
            raise Qwen3VLModelError("Failed to initialize Qwen3VL model") from e

    def _setup_reproducibility(self, seed: int) -> None:
        """Setup reproducibility settings for deterministic behavior."""
        # Set CUBLAS workspace config for deterministic cuBLAS operations
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Set seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Set cuDNN to deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Enable deterministic algorithms (warn_only=True to allow Flash Attention)
        # When Flash Attention is disabled, this ensures full determinism
        torch.use_deterministic_algorithms(True, warn_only=True)

        logger.info(f"Reproducibility settings applied (seed={seed})")

    def _init_model(self) -> None:
        """Initialize the model and processor."""
        # Select model class based on model name
        if "Qwen2.5-VL" in self.model_name:
            model_cls = Qwen2_5_VLForConditionalGeneration
        elif "MoE" in self.model_name or "A3B" in self.model_name:
            model_cls = Qwen3VLMoeForConditionalGeneration
        else:
            model_cls = Qwen3VLForConditionalGeneration

        # AWQ quantized models are quantized in fp16; using "auto" can mix
        # fp16 weights with bf16 activations, causing triton dtype mismatches.
        is_awq = "AWQ" in self.model_name
        dtype = torch.float16 if is_awq else torch.bfloat16

        # Use flash_attention_2 if available, fall back to sdpa
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            print("Using flash_attention_2")
        except ImportError:
            attn_impl = "sdpa"
            print("Using sdpa")
        # Load model
        logger.info(f"Loading {model_cls.__name__} with device_map={self.device_map}, dtype={dtype}, attn={attn_impl}")

        load_kwargs = dict(
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            device_map=self.device_map,
        )

        # When distributing across multiple GPUs, limit GPU 0 so only the
        # vision encoder + input embedding fit there. This keeps the bulk of
        # transformer layers on GPUs 1-N, while GPU 0 retains enough free
        # memory for input pixel values, embedding model, and activations.
        if self.device_map == "auto" and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                full_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # GPU 0: small limit (vision encoder ~3-5GB only)
                # Other GPUs: large limit for transformer layers
                gpu0_limit = f"{int(full_mem * 0.25)}GiB"
                other_limit = f"{int(full_mem * 0.9)}GiB"
                max_memory = {0: gpu0_limit}
                for i in range(1, num_gpus):
                    max_memory[i] = other_limit
                load_kwargs["max_memory"] = max_memory
                logger.info(f"Multi-GPU max_memory: {max_memory}")

        self.model = model_cls.from_pretrained(self.model_name, **load_kwargs)

        # Log device distribution
        if hasattr(self.model, "hf_device_map"):
            unique_devices = set(self.model.hf_device_map.values())
            logger.info(f"Model distributed across devices: {sorted(unique_devices)}")
        else:
            logger.info(f"Model on device: {self.model.device}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def _validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate and convert file path to Path object."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return path

    def _calculate_sample_indices(
        self, vr: VideoReader, total_frames: int
    ) -> List[int]:
        """Calculate which frames to sample from the video."""
        sample_indices = []

        if self.fps is not None:
            # Sample at specified FPS
            video_fps = vr.get_avg_fps()
            if video_fps <= 0:
                raise Qwen3VLModelError("Cannot determine video FPS")

            frame_interval = max(1, int(video_fps / self.fps))
            sample_indices = list(range(0, total_frames, frame_interval))

        elif self.nframes is not None:
            # Sample fixed number of frames
            if self.nframes <= 0:
                raise ValueError("nframes must be a positive integer")

            if self.nframes >= total_frames:
                sample_indices = list(range(total_frames))
            else:
                # Evenly distribute frames across video duration
                indices = [
                    int(i * (total_frames - 1) / (self.nframes - 1))
                    for i in range(self.nframes)
                ]
                sample_indices = sorted(list(set(indices)))

        else:
            # Default: sample at 1 FPS
            logger.warning("No fps or nframes specified, defaulting to 1 FPS sampling")
            video_fps = vr.get_avg_fps()
            frame_interval = max(1, int(video_fps)) if video_fps > 0 else 30
            sample_indices = list(range(0, total_frames, frame_interval))

        # Ensure we have at least the first and last frame
        if sample_indices and sample_indices[0] != 0:
            sample_indices.insert(0, 0)
        if sample_indices and sample_indices[-1] != total_frames - 1:
            sample_indices.append(total_frames - 1)

        # Remove duplicates and sort
        sample_indices = sorted(list(set(sample_indices)))

        logger.debug(
            f"Sampling {len(sample_indices)} frames from {total_frames} total frames"
        )
        return sample_indices

    def _process_image(self, image_input: Any) -> Image.Image:
        """Process a single image input with resizing."""
        # Load image
        if isinstance(image_input, (str, Path)):
            path = self._validate_file_path(image_input)
            img = Image.open(path)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        # Ensure RGB mode
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize maintaining aspect ratio
        img.thumbnail(self.max_size, Image.Resampling.LANCZOS)

        return img

    def _process_video(self, video_path: Union[str, Path]) -> List[Image.Image]:
        """Extract video frames as PIL Image objects with resizing."""
        video_path = self._validate_file_path(video_path)

        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)

            if total_frames == 0:
                raise Qwen3VLModelError(
                    f"Video file appears to be empty or corrupted: {video_path}"
                )

            # Determine sampling strategy
            sample_indices = self._calculate_sample_indices(vr, total_frames)

            # Extract frames as PIL Images
            frames = []
            for idx in sample_indices:
                try:
                    frame = vr[idx].asnumpy()
                    img = Image.fromarray(frame)

                    # Ensure RGB mode
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    # Resize maintaining aspect ratio
                    img.thumbnail(self.max_size_video, Image.Resampling.LANCZOS)

                    frames.append(img)
                except Exception as e:
                    logger.warning(f"Failed to process frame {idx}: {e}")
                    continue

            return frames

        except Exception as e:
            logger.error(f"Failed to extract video frames {video_path}: {e}")
            raise Qwen3VLModelError(f"Failed to extract video frames: {e}") from e

    def _preprocess_prompt(self, prompt: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess prompt to convert it to Qwen3VL message format.

        Args:
            prompt: Conversation prompt in our internal format

        Returns:
            Prompt in Qwen3VL message format
        """
        prompt_copy = copy.deepcopy(prompt)

        for item in prompt_copy:
            if isinstance(item["content"], str):
                item["content"] = [{"type": "text", "text": item["content"]}]

        # Collect items with content to process in parallel
        content_items = [(i, item) for i, item in enumerate(prompt_copy)]

        # Process content items in parallel using ThreadPoolExecutor
        max_workers = min(len(content_items), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._process_content, item["content"]): i
                for i, item in content_items
            }

            # Wait for all content processing to complete
            for future in as_completed(future_to_index):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing content: {e}")

        return prompt_copy

    def _process_content(
        self, content: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    ) -> None:
        """Process content by converting image and video references."""
        if isinstance(content, str):
            return

        if isinstance(content, dict):
            content = [content]

        if not isinstance(content, list):
            raise ValueError("content must be a str, dict, or list")

        # Collect image/video tasks
        media_tasks = []
        for i, item in enumerate(content):
            if not isinstance(item, dict):
                raise ValueError(f"Content item must be a dict, got {type(item)}")

            t = item.get("type")
            if t == "text" and "text" in item:
                # Keep text items as-is
                continue
            elif t == "image" and "image" in item:
                media_tasks.append((i, "image", item["image"]))
            elif t == "video" and "video" in item:
                media_tasks.append((i, "video", item["video"]))
            else:
                raise ValueError(f"Unsupported media item at index {i}: {item}")

        if not media_tasks:
            return

        # Process image/video tasks in parallel
        max_workers = min(len(media_tasks), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            for task in media_tasks:
                i, media_type, file = task
                if media_type == "image":
                    fut = executor.submit(self._process_image, file)
                elif media_type == "video":
                    fut = executor.submit(self._process_video, file)
                else:
                    raise ValueError(
                        f"Unsupported media type at index {i}: {media_type}"
                    )
                future_to_task[fut] = task

            # Collect results
            results = []
            for fut in as_completed(future_to_task):
                i, media_type, _ = future_to_task[fut]
                try:
                    if media_type == "image":
                        processed_image = fut.result()
                        results.append((i, "image", processed_image))
                    elif media_type == "video":
                        video_frames = fut.result()
                        results.append((i, "video", video_frames))
                except Exception as e:
                    logger.error(
                        f"Failed to process {media_type} content at index {i}: {e}"
                    )
                    results.append((i, "error", None))

            # Sort results by original index to maintain order
            results.sort(key=lambda x: x[0])

            # Apply results to content with proper shifting
            shift = 0
            for i, media_type, result in results:
                if media_type == "error":
                    continue
                elif media_type == "image":
                    content[i + shift] = {"type": "image", "image": result}
                elif media_type == "video":
                    # For Qwen3VL, we convert video frames to image items
                    image_items = [
                        {"type": "image", "image": frame} for frame in result
                    ]
                    content[i + shift : i + shift + 1] = image_items
                    shift += len(image_items) - 1

    def _normalize_prompt(
        self, prompt: Union[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Coerce string prompts into a single user message."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    @dynamic_retry_decorator
    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        text_format: Optional[type] = None,
        **kwargs,
    ) -> Any:
        """
        Generate completion for a single prompt.

        Args:
            prompt: Conversation prompt in our internal format or raw string
            text_format (optional): Pydantic model or other structure for parsing the response
            **kwargs: Additional arguments for the generation call

        Returns:
            Generated response string or parsed object if text_format is provided

        Raises:
            Qwen3VLModelError: If generation fails
        """
        prompt_copy = copy.deepcopy(self._normalize_prompt(prompt))
        messages = self._preprocess_prompt(prompt_copy)

        try:
            # Prepare text for Qwen3VL processor
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision info (images and videos)
            # process_vision_info returns (image_inputs, video_inputs, video_kwargs)
            vision_outputs = process_vision_info(
                messages, image_patch_size=self.image_patch_size
            )
            image_inputs = vision_outputs[0] if len(vision_outputs) > 0 else None
            video_inputs = vision_outputs[1] if len(vision_outputs) > 1 else None

            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # Free vision data after tokenization
            del messages, vision_outputs, image_inputs, video_inputs, text

            # Merge kwargs with instance kwargs (call-specific kwargs take precedence)
            gen_kwargs = {**self.kwargs, **kwargs}

            # Filter out GPT-specific arguments that might be passed from LLMModel
            gen_kwargs.pop("service_tier", None)
            gen_kwargs.pop("timeout", None)

            # Set seed for reproducibility if present
            if "seed" in gen_kwargs:
                seed = gen_kwargs.pop("seed")
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            # Set default generation parameters if not specified
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 2048

            # Handle greedy decoding (temperature=0.0)
            if gen_kwargs.get("temperature") == 0.0:
                gen_kwargs["do_sample"] = False
                gen_kwargs.pop("temperature")
            elif "temperature" in gen_kwargs:
                gen_kwargs["do_sample"] = True

            # Generate with no_grad to prevent autograd from caching computation graphs
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

            # Decode output
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            result = output_text[0].strip() if output_text else ""

            # Free GPU tensors and release cached memory
            del inputs, generated_ids, generated_ids_trimmed
            torch.cuda.empty_cache()

            # Parse response using text_format if provided
            if text_format is not None:
                result = self._parse_structured_response(result, text_format)

            return result

        except Exception as e:
            logger.error(f"Qwen3VL generation error: {e}")
            # Cleanup on error too
            torch.cuda.empty_cache()
            raise Qwen3VLModelError(f"Failed to generate completion: {e}") from e

    def generate_batch(
        self,
        batch_prompts: List[Union[str, List[Dict[str, Any]]]],
        text_format: Optional[type] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Process multiple prompts in batch.

        Args:
            batch_prompts: List of conversation prompts
            text_format (optional): Pydantic model or other structure for parsing the response
            **kwargs: Additional arguments for the generation call

        Returns:
            List of response strings or parsed objects if text_format is provided

        Raises:
            Qwen3VLModelError: If batch generation fails
        """
        if not batch_prompts:
            return []

        # Process prompts in parallel using ThreadPoolExecutor
        max_workers = min(len(batch_prompts), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    lambda p: self.generate(
                        self._normalize_prompt(p), text_format=text_format, **kwargs
                    ),
                    batch_prompts,
                )
            )

        return results

    def generate_batch_parallel(
        self,
        batch_prompts: List[Union[str, List[Dict[str, Any]]]],
        max_workers: Optional[int] = None,
        text_format: Optional[type] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Process multiple prompts in parallel using ThreadPoolExecutor.

        This is an alias for generate_batch with explicit max_workers parameter.

        Args:
            batch_prompts: List of conversation prompts
            max_workers: Maximum number of parallel workers (defaults to min(32, len(batch_prompts)))
            text_format (optional): Pydantic model or other structure for parsing the response
            **kwargs: Additional arguments for the generation call

        Returns:
            List of response strings or parsed objects if text_format is provided, in the same order as input

        Raises:
            Qwen3VLModelError: If batch generation fails
        """
        if not batch_prompts:
            return []

        if max_workers is None:
            max_workers = min(32, len(batch_prompts))

        results: List[Any] = [""] * len(batch_prompts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.generate, prompt, text_format, **kwargs): i
                for i, prompt in enumerate(batch_prompts)
            }

            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Failed to process prompt at index {index}: {e}")
                    raise Qwen3VLModelError(
                        f"Batch generation failed at index {index}"
                    ) from e

        return results

    def _parse_structured_response(self, response: str, text_format: type) -> Any:
        """
        Parse the response string using the provided text_format (Pydantic model).

        Args:
            response: The raw response string from the model
            text_format: Pydantic model class for parsing

        Returns:
            Parsed object of type text_format

        Raises:
            Qwen3VLModelError: If parsing fails
        """
        try:
            # Clean the response by removing markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # Remove C-style comments (// ...)
            import re

            cleaned_response = re.sub(
                r"//.*$", "", cleaned_response, flags=re.MULTILINE
            )

            # Try to parse as JSON first
            json_data = json.loads(cleaned_response)

            # If it's a Pydantic model, use model_validate (Pydantic v2)
            if hasattr(text_format, "model_validate"):
                return text_format.model_validate(json_data)
            # Fallback for older Pydantic versions
            elif hasattr(text_format, "parse_obj"):
                return text_format.parse_obj(json_data)
            else:
                logger.warning(f"Unsupported text_format type: {text_format}")
                raise ValueError("text_format must be a Pydantic model class")
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(
                f"Failed to parse response as JSON or Validation failed: {e}"
            )
            logger.error(f"Raw response causing failure: {response}")

            # Fallback: Try to extract data using regex if standard JSON parsing fails
            try:
                import re

                if "NerRawOutput" in str(text_format):
                    # Extract named_entities list
                    match = re.search(
                        r'"named_entities"\s*:\s*\[(.*?)\]', cleaned_response, re.DOTALL
                    )
                    if match:
                        content = match.group(1)
                        # Extract individual strings
                        entities = re.findall(r'"((?:[^"\\]|\\.)*)"', content)
                        return text_format(named_entities=entities)

                elif "TripleRawOutput" in str(text_format):
                    # Improved fallback for TripleRawOutput
                    # Find all innermost lists (triples) using a regex that handles quoted brackets
                    # This regex matches [...] containing no unquoted brackets
                    list_regex = r"\[((?:[^\[\]\"]|\"[^\"\\]*(?:\\.[^\"\\]*)*\")*)\]"
                    inner_lists = re.findall(list_regex, cleaned_response, re.DOTALL)

                    triples = []
                    for inner in inner_lists:
                        # Extract items: quoted strings or null
                        # Group 1: quoted string content
                        # Group 2: null
                        item_regex = r'(?:"((?:[^"\\]|\\.)*)"|\b(null)\b)'
                        items_matches = re.findall(item_regex, inner)

                        current_triple = []
                        has_null = False
                        for match in items_matches:
                            if match[1]:  # null found
                                has_null = True
                                break  # Skip this triple entirely if it has null
                            else:  # String found
                                current_triple.append(match[0])

                        if not has_null and current_triple:
                            triples.append(current_triple)

                    return text_format(triples=triples)
            except Exception as fallback_error:
                logger.error(f"Fallback parsing also failed: {fallback_error}")

            raise Qwen3VLModelError(f"Failed to parse response as JSON: {e}") from e
        except Exception as e:
            logger.error(f"Failed to parse structured response: {e}")
            raise Qwen3VLModelError(f"Failed to parse structured response: {e}") from e

    def __repr__(self) -> str:
        """String representation of the model instance."""
        return f"Qwen3VLModel(model_name='{self.model_name}', " f"kwargs={self.kwargs})"


# Convenience function
def create_qwen3vl_model(model_name: str, **kwargs) -> Qwen3VLModel:
    """Create a Qwen3VL model instance with convenient defaults."""
    return Qwen3VLModel(model_name=model_name, **kwargs)
