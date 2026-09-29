"""Gemini wrapper: generate / generate_batch / generate_async with SQLite caching."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import io
import json
import logging
import os
import sqlite3
import threading
from collections import OrderedDict
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
from decord import cpu, VideoReader
from filelock import FileLock
from google import genai
from google.genai import types
from PIL import Image

from .utils import dynamic_retry_decorator

logger = logging.getLogger(__name__)

# Model configuration — aliases map to canonical Gemini model IDs.
# Unknown names pass through as-is.
# "vertex-gemini-*" prefix → Vertex AI, stripped to "gemini-*" for API call.
MODEL_DICT = {
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "vertex-gemini-2.5-pro": "gemini-2.5-pro",
    "vertex-gemini-2.5-flash": "gemini-2.5-flash",
    "vertex-gemini-2.0-flash": "gemini-2.0-flash",
}

# Global in-memory cache for encoded images/videos
_CACHE: OrderedDict[Tuple, Any] = OrderedDict()
_MAX_CACHE_SIZE = 500
_CACHE_LOCK = threading.Lock()


class GeminiModelError(Exception):
    """Custom exception for Gemini model operations."""

    pass


class GeminiModel:
    """
    Google Gemini model wrapper with video and image processing capabilities.

    Accepts prompts in OpenAI chat format and converts them to the Gemini API format.
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
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        service_tier: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if fps is not None and nframes is not None:
            raise ValueError(
                "Cannot provide both 'fps' and 'nframes'. Please choose one."
            )

        # Detect Vertex AI mode from "vertex-" prefix
        self._use_vertex = model_name.lower().startswith("vertex-")

        # Resolve model name (alias → canonical, or pass through)
        self.model_name = MODEL_DICT.get(model_name, model_name)
        # Also handle unknown vertex- names not in MODEL_DICT
        if self.model_name.startswith("vertex-"):
            self.model_name = self.model_name[len("vertex-"):]

        # Initialize Gemini client
        try:
            if self._use_vertex:
                vertex_api_key = (
                    api_key
                    or os.getenv("VERTEX_API_KEY")
                    or os.getenv("GOOGLE_API_KEY")
                    or os.getenv("GEMINI_API_KEY")
                )
                if not vertex_api_key:
                    raise GeminiModelError(
                        "Vertex AI requires an API key. "
                        "Set VERTEX_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY."
                    )
                self.client = genai.Client(
                    vertexai=True,
                    api_key=vertex_api_key,
                    location="global",
                )
                logger.info("Using Vertex AI (api_key, global endpoint)")
            else:
                api_key = (
                    api_key
                    or os.getenv("GOOGLE_API_KEY")
                    or os.getenv("GEMINI_API_KEY")
                )
                if not api_key:
                    raise GeminiModelError(
                        "Google API key not found. "
                        "Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
                    )
                self.client = genai.Client(api_key=api_key)
                logger.info("Using AI Studio (API key)")
        except GeminiModelError:
            raise
        except Exception as e:
            raise GeminiModelError(f"Failed to initialize Gemini client: {e}") from e

        # Instance attributes
        self.max_retries = max(1, max_retries)
        self.max_size = max_size
        self.max_size_video = max_size_video
        self.quality = max(1, min(100, quality))
        self.fps = fps
        self.nframes = nframes
        self.timeout = timeout

        # Filter out irrelevant kwargs (from OpenAI / local model callers)
        kwargs.pop("device_map", None)
        kwargs.pop("device", None)

        self.kwargs = kwargs
        if "seed" not in self.kwargs:
            self.kwargs["seed"] = 42
        if "temperature" not in self.kwargs:
            self.kwargs["temperature"] = 0.0

        # SQLite cache
        cache_base = cache_dir or ".cache"
        os.makedirs(cache_base, exist_ok=True)
        safe_name = model_name.replace("-", "_").replace(".", "_")
        self.cache_file_name = os.path.join(cache_base, f"gemini_cache_{safe_name}.db")

        logger.info(f"Initialized GeminiModel with {self.model_name}")

    # ------------------------------------------------------------------
    # Image / video encoding (mirrors OpenAI wrapper)
    # ------------------------------------------------------------------

    def _validate_file_path(self, file_path: Union[str, Path]) -> Path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return path

    def _image_cache_identifier(self, img: Image.Image) -> str:
        filename = getattr(img, "filename", None)
        if filename:
            return str(filename)
        try:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG", optimize=True)
            return hashlib.md5(buf.getvalue()).hexdigest()
        except Exception:
            return f"pil-id-{id(img)}"

    def _manage_cache(self, key: Tuple, value: Any) -> None:
        with _CACHE_LOCK:
            _CACHE[key] = value
            _CACHE.move_to_end(key)
            if len(_CACHE) > _MAX_CACHE_SIZE:
                _CACHE.popitem(last=False)

    def encode_image(self, image: Union[str, Path, Image.Image]) -> str:
        """Encode image to base64 JPEG string with caching."""
        try:
            if isinstance(image, Image.Image):
                img = image
                identifier = self._image_cache_identifier(img)
            else:
                path = self._validate_file_path(image)
                identifier = str(path)
                img = Image.open(path)

            cache_key = (identifier, self.max_size, self.quality)
            with _CACHE_LOCK:
                if cache_key in _CACHE:
                    _CACHE.move_to_end(cache_key)
                    return _CACHE[cache_key]

            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail(self.max_size, Image.Resampling.LANCZOS)

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=self.quality, optimize=True)
            encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

            self._manage_cache(cache_key, encoded)

            if not isinstance(image, Image.Image):
                try:
                    img.close()
                except Exception:
                    pass
            return encoded

        except Exception as e:
            raise GeminiModelError(f"Failed to encode image: {e}") from e

    def encode_video(self, video_path: Union[str, Path]) -> List[str]:
        """Encode video frames to base64 JPEG strings with caching."""
        video_path = self._validate_file_path(video_path)
        cache_key = (
            str(video_path),
            self.max_size_video,
            self.quality,
            self.fps,
            self.nframes,
        )
        with _CACHE_LOCK:
            if cache_key in _CACHE:
                _CACHE.move_to_end(cache_key)
                return _CACHE[cache_key]

        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)
            if total_frames == 0:
                raise GeminiModelError(f"Video appears empty: {video_path}")

            sample_indices = self._calculate_sample_indices(vr, total_frames)
            base64_frames: List[str] = []
            for idx in sample_indices:
                try:
                    frame = vr[idx].asnumpy()
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_resized = cv2.resize(
                        frame_bgr, self.max_size_video, interpolation=cv2.INTER_LANCZOS4
                    )
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                    success, buffer = cv2.imencode(".jpg", frame_resized, encode_param)
                    if success:
                        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
                except Exception as e:
                    logger.warning(f"Failed to process frame {idx}: {e}")

            if not base64_frames:
                raise GeminiModelError(f"No frames extracted from: {video_path}")

            self._manage_cache(cache_key, base64_frames)
            return base64_frames

        except Exception as e:
            raise GeminiModelError(f"Failed to encode video: {e}") from e

    def _calculate_sample_indices(
        self, vr: VideoReader, total_frames: int
    ) -> List[int]:
        if self.fps is not None:
            video_fps = vr.get_avg_fps()
            if video_fps <= 0:
                raise GeminiModelError("Cannot determine video FPS")
            frame_interval = max(1, int(video_fps / self.fps))
            sample_indices = list(range(0, total_frames, frame_interval))
        elif self.nframes is not None:
            if self.nframes <= 0:
                raise ValueError("nframes must be positive")
            if self.nframes >= total_frames:
                sample_indices = list(range(total_frames))
            else:
                indices = [
                    int(i * (total_frames - 1) / (self.nframes - 1))
                    for i in range(self.nframes)
                ]
                sample_indices = sorted(set(indices))
        else:
            video_fps = vr.get_avg_fps()
            frame_interval = max(1, int(video_fps)) if video_fps > 0 else 30
            sample_indices = list(range(0, total_frames, frame_interval))

        if sample_indices and sample_indices[0] != 0:
            sample_indices.insert(0, 0)
        if sample_indices and sample_indices[-1] != total_frames - 1:
            sample_indices.append(total_frames - 1)
        return sorted(set(sample_indices))

    # ------------------------------------------------------------------
    # Prompt conversion: OpenAI chat format → Gemini format
    # ------------------------------------------------------------------

    def _convert_prompt(
        self, prompt: Union[str, List[Dict[str, Any]]]
    ) -> Tuple[List[types.Content], Optional[str]]:
        """Convert OpenAI-style chat messages to Gemini (contents, system_instruction)."""
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        system_parts: List[str] = []
        contents: List[types.Content] = []

        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Extract system messages → system_instruction
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            system_parts.append(item["text"])
                        elif isinstance(item, str):
                            system_parts.append(item)
                continue

            gemini_role = "model" if role == "assistant" else "user"
            parts = self._convert_content_to_parts(content)
            if not parts:
                continue

            # Merge consecutive messages with the same role
            if contents and contents[-1].role == gemini_role:
                contents[-1].parts.extend(parts)
            else:
                contents.append(types.Content(role=gemini_role, parts=parts))

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return contents, system_instruction

    def _convert_content_to_parts(
        self, content: Union[str, Dict, List]
    ) -> List[types.Part]:
        """Convert a single message's content to a list of Gemini Parts."""
        if isinstance(content, str):
            return [types.Part.from_text(text=content)]

        if isinstance(content, dict):
            content = [content]

        if not isinstance(content, list):
            return [types.Part.from_text(text=str(content))]

        parts: List[types.Part] = []
        # Collect media tasks for parallel encoding
        media_tasks: List[Tuple[int, str, Any]] = []

        for i, item in enumerate(content):
            if isinstance(item, str):
                parts.append(("text", item))
                continue
            if not isinstance(item, dict):
                parts.append(("text", str(item)))
                continue

            item_type = item.get("type", "")
            if item_type == "text":
                parts.append(("text", item.get("text", "")))
            elif item_type == "image":
                img = item.get("image")
                if img is not None:
                    media_tasks.append((len(parts), "image", img))
                    parts.append(("placeholder", None))
            elif item_type == "video":
                vid = item.get("video")
                if vid is not None:
                    media_tasks.append((len(parts), "video", vid))
                    parts.append(("placeholder", None))
            else:
                # Unknown type — treat as text
                text = item.get("text", str(item))
                parts.append(("text", text))

        # Process media in parallel
        if media_tasks:
            max_workers = min(len(media_tasks), (os.cpu_count() or 1) + 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {}
                for idx, media_type, media_src in media_tasks:
                    if media_type == "image":
                        fut = executor.submit(self.encode_image, media_src)
                    else:
                        fut = executor.submit(self.encode_video, media_src)
                    future_to_task[fut] = (idx, media_type)

                for fut in as_completed(future_to_task):
                    idx, media_type = future_to_task[fut]
                    try:
                        result = fut.result()
                        if media_type == "image":
                            parts[idx] = ("image_b64", result)
                        else:
                            parts[idx] = ("video_b64s", result)
                    except Exception as e:
                        logger.error(f"Failed to process {media_type}: {e}")
                        parts[idx] = ("text", f"[{media_type} encoding failed]")

        # Convert tagged tuples to actual Parts
        gemini_parts: List[types.Part] = []
        for tag, data in parts:
            if tag == "text":
                gemini_parts.append(types.Part.from_text(text=data))
            elif tag == "image_b64":
                gemini_parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=base64.b64decode(data),
                        )
                    )
                )
            elif tag == "video_b64s":
                for frame_b64 in data:
                    gemini_parts.append(
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/jpeg",
                                data=base64.b64decode(frame_b64),
                            )
                        )
                    )
            elif tag == "placeholder":
                pass  # encoding failed silently

        return gemini_parts

    # ------------------------------------------------------------------
    # SQLite response caching
    # ------------------------------------------------------------------

    def _cache_key_hash(
        self, prompt: Any, text_format: Optional[type] = None
    ) -> str:
        key_data = {
            "prompt": prompt,
            "model": self.model_name,
            "text_format": str(text_format) if text_format else None,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def _read_cache(
        self, key_hash: str, text_format: Optional[type] = None
    ) -> Tuple[bool, Any]:
        try:
            if not os.path.exists(self.cache_file_name):
                return False, None
            conn = sqlite3.connect(self.cache_file_name, timeout=1.0)
            c = conn.cursor()
            c.execute("SELECT message FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message_dict = json.loads(row[0])
                if text_format and isinstance(message_dict, dict):
                    return True, text_format(**message_dict)
                return True, message_dict
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            pass
        return False, None

    def _write_cache(self, key_hash: str, message: Any) -> None:
        lock_file = self.cache_file_name + ".lock"
        os.makedirs(os.path.dirname(self.cache_file_name) or ".", exist_ok=True)
        try:
            with FileLock(lock_file):
                conn = sqlite3.connect(self.cache_file_name)
                c = conn.cursor()
                c.execute(
                    "CREATE TABLE IF NOT EXISTS cache "
                    "(key TEXT PRIMARY KEY, message TEXT)"
                )
                message_str = json.dumps(
                    message,
                    default=lambda o: (
                        o.model_dump() if hasattr(o, "model_dump") else str(o)
                    ),
                )
                c.execute(
                    "INSERT OR REPLACE INTO cache (key, message) VALUES (?, ?)",
                    (key_hash, message_str),
                )
                conn.commit()
                conn.close()
        except Exception as e:
            logger.warning(f"Failed to write to cache: {e}")

    # ------------------------------------------------------------------
    # Build GenerateContentConfig
    # ------------------------------------------------------------------

    def _build_config(
        self,
        system_instruction: Optional[str],
        text_format: Optional[type] = None,
    ) -> types.GenerateContentConfig:
        cfg_kwargs: Dict[str, Any] = {
            "temperature": self.kwargs.get("temperature", 0.0),
            "seed": self.kwargs.get("seed", 42),
        }
        if system_instruction:
            cfg_kwargs["system_instruction"] = system_instruction
        if text_format is not None:
            cfg_kwargs["response_mime_type"] = "application/json"
            if hasattr(text_format, "model_json_schema"):
                cfg_kwargs["response_json_schema"] = text_format.model_json_schema()
        return types.GenerateContentConfig(**cfg_kwargs)

    def _parse_response(self, response: Any, text_format: Optional[type] = None) -> Any:
        """Extract text from Gemini response and optionally parse as structured output."""
        result_text = response.text

        if text_format is not None and hasattr(text_format, "model_validate_json"):
            return text_format.model_validate_json(result_text)

        return result_text

    # ------------------------------------------------------------------
    # Synchronous generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        text_format: Optional[type] = None,
        **kwargs,
    ) -> Any:
        """Generate completion for a single prompt (cached + retried)."""
        key_hash = self._cache_key_hash(prompt, text_format)
        hit, cached = self._read_cache(key_hash, text_format)
        if hit:
            return cached

        result = self._generate_impl(prompt, text_format, **kwargs)
        self._write_cache(key_hash, result)
        return result

    @dynamic_retry_decorator
    def _generate_impl(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        text_format: Optional[type] = None,
        **kwargs,
    ) -> Any:
        prompt_copy = copy.deepcopy(prompt) if not isinstance(prompt, str) else prompt
        contents, system_instruction = self._convert_prompt(prompt_copy)
        config = self._build_config(system_instruction, text_format)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            return self._parse_response(response, text_format)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise GeminiModelError(f"Failed to get completion: {e}") from e

    # ------------------------------------------------------------------
    # Batch generation (async under the hood)
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        batch_prompts: List[Union[str, List[Dict[str, Any]]]],
        text_format: Optional[type] = None,
    ) -> List[Any]:
        """Process multiple prompts in batch with async generation."""
        if not batch_prompts:
            return []
        try:
            return asyncio.run(
                self._async_batch_generation(batch_prompts, text_format=text_format)
            )
        except Exception as e:
            raise GeminiModelError(f"Batch generation failed: {e}") from e

    async def _async_batch_generation(
        self,
        batch_prompts: List[Union[str, List[Dict[str, Any]]]],
        chunk_size: int = 50,
        text_format: Optional[type] = None,
    ) -> List[Any]:
        responses: List[Any] = []
        total_chunks = (len(batch_prompts) + chunk_size - 1) // chunk_size

        for i in range(0, len(batch_prompts), chunk_size):
            chunk_num = (i // chunk_size) + 1
            batch = batch_prompts[i : i + chunk_size]
            logger.info(
                f"Processing chunk {chunk_num}/{total_chunks} ({len(batch)} prompts)"
            )
            tasks = [
                self._generate_single_async(p, text_format) for p in batch
            ]
            batch_responses = await asyncio.gather(*tasks)
            responses.extend(batch_responses)

        return responses

    # ------------------------------------------------------------------
    # Async generation
    # ------------------------------------------------------------------

    async def generate_async(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        text_format: Optional[type] = None,
        **kwargs,
    ) -> Any:
        """Generate completion asynchronously (cached)."""
        key_hash = self._cache_key_hash(prompt, text_format)
        hit, cached = self._read_cache(key_hash, text_format)
        if hit:
            return cached

        result = await self._generate_async_impl(prompt, text_format, **kwargs)
        self._write_cache(key_hash, result)
        return result

    async def _generate_single_async(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        text_format: Optional[type] = None,
    ) -> Any:
        """Single-prompt async helper for batch generation (cached)."""
        key_hash = self._cache_key_hash(prompt, text_format)
        hit, cached = self._read_cache(key_hash, text_format)
        if hit:
            return cached

        result = await self._generate_async_impl(prompt, text_format)
        self._write_cache(key_hash, result)
        return result

    @dynamic_retry_decorator
    async def _generate_async_impl(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        text_format: Optional[type] = None,
        **kwargs,
    ) -> Any:
        prompt_copy = copy.deepcopy(prompt) if not isinstance(prompt, str) else prompt
        contents, system_instruction = self._convert_prompt(prompt_copy)
        config = self._build_config(system_instruction, text_format)

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            return self._parse_response(response, text_format)
        except Exception as e:
            logger.error(f"Async Gemini API error: {e}")
            raise GeminiModelError(f"Failed to get async completion: {e}") from e

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        _CACHE.clear()
        logger.info("In-memory cache cleared")

    def __repr__(self) -> str:
        return f"GeminiModel(model_name='{self.model_name}', kwargs={self.kwargs})"
