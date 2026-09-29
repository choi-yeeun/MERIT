"""
OpenAI GPT Model Wrapper with comprehensive video and image processing capabilities.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import functools
import hashlib
import io
import json
import logging
import os
import sqlite3
from collections import OrderedDict
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
from decord import cpu, VideoReader

from filelock import FileLock
from openai import AsyncOpenAI, OpenAI
from PIL import Image
from pydantic_core import PydanticUndefined
from tqdm.asyncio import tqdm as tqdm_asyncio

from .utils import dynamic_retry_decorator

logger = logging.getLogger(__name__)

# Model configuration
MODEL_DICT = {
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-5": "gpt-5-2025-08-07",
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
    "gpt-5-nano": "gpt-5-nano-2025-08-07",
}

import threading

# Global cache configuration
_CACHE: OrderedDict[Tuple, Any] = OrderedDict()
_MAX_CACHE_SIZE = 500
_CACHE_LOCK = threading.Lock()


def class_str(cls):
    fields = []
    for name, field in cls.model_fields.items():
        annotation = str(field.annotation).replace("typing.", "")
        if field.default is not PydanticUndefined:
            fields.append(f"{name}: {annotation} = {field.default}")
        else:
            fields.append(f"{name}: {annotation}")
    return f"{cls.__name__}({', '.join(fields)})"


def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if args:
            prompt = args[0]
        else:
            prompt = kwargs.get("prompt")
        if prompt is None:
            raise ValueError("Missing required 'prompt' parameter for caching.")

        model = getattr(self, "model_name", None)
        text_format = args[1] if len(args) > 1 else kwargs.get("text_format")

        key_data = {
            "prompt": prompt,
            "model": model,
            "text_format": class_str(text_format) if text_format else None,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        lock_file = self.cache_file_name + ".lock"

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT
                )
            """
            )
            conn.commit()
            c.execute("SELECT message FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message_dict = json.loads(row[0])
                text_format = args[1] if len(args) > 1 else kwargs.get("text_format")
                if text_format and isinstance(message_dict, dict):
                    message = text_format(**message_dict)
                else:
                    message = message_dict
                return message

        message = func(self, *args, **kwargs)

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT
                )
            """
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

        return message

    return wrapper


def async_cache_response(func):
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        if args:
            prompt = args[0]
        else:
            prompt = kwargs.get("prompt")
        if prompt is None:
            raise ValueError("Missing required 'prompt' parameter for caching.")

        model = getattr(self, "model_name", None)
        text_format = args[1] if len(args) > 1 else kwargs.get("text_format")

        key_data = {
            "prompt": prompt,
            "model": model,
            "text_format": str(text_format) if text_format else None,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        lock_file = self.cache_file_name + ".lock"

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT
                )
            """
            )
            conn.commit()
            c.execute("SELECT message FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message_dict = json.loads(row[0])
                text_format = args[1] if len(args) > 1 else kwargs.get("text_format")
                if text_format and isinstance(message_dict, dict):
                    message = text_format(**message_dict)
                else:
                    message = message_dict
                return message

        message = await func(self, *args, **kwargs)

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT
                )
            """
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

        return message

    return wrapper


def async_cache_response_fast(func):
    """Optimized async cache decorator that avoids blocking locks for reads."""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        if args:
            prompt = args[0]
        else:
            prompt = kwargs.get("prompt")
        if prompt is None:
            raise ValueError("Missing required 'prompt' parameter for caching.")

        model = getattr(self, "model_name", None)
        text_format = args[1] if len(args) > 1 else kwargs.get("text_format")

        key_data = {
            "prompt": prompt,
            "model": model,
            "text_format": str(text_format) if text_format else None,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        try:
            if os.path.exists(self.cache_file_name):
                conn = sqlite3.connect(self.cache_file_name, timeout=1.0)
                c = conn.cursor()
                c.execute("SELECT message FROM cache WHERE key = ?", (key_hash,))
                row = c.fetchone()
                conn.close()
                if row is not None:
                    message_dict = json.loads(row[0])
                    if text_format and isinstance(message_dict, dict):
                        return text_format(**message_dict)
                    return message_dict
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            pass

        message = await func(self, *args, **kwargs)

        lock_file = self.cache_file_name + ".lock"
        os.makedirs(os.path.dirname(self.cache_file_name), exist_ok=True)
        with FileLock(lock_file):
            try:
                conn = sqlite3.connect(self.cache_file_name)
                c = conn.cursor()
                c.execute(
                    "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, message TEXT)"
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

        return message

    return wrapper


class OpenAIModelError(Exception):
    """Custom exception for OpenAI model operations."""
    pass


class OpenAIModel:
    """OpenAI GPT model wrapper with video and image processing capabilities."""

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

        if model_name not in MODEL_DICT:
            raise ValueError(
                f"Unsupported model: {model_name}. Available: {list(MODEL_DICT.keys())}"
            )

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIModelError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        try:
            self.async_client = AsyncOpenAI(api_key=api_key)
            self.sync_client = OpenAI(api_key=api_key)
        except Exception as e:
            raise OpenAIModelError("Failed to initialize OpenAI client") from e

        self.model_name = MODEL_DICT[model_name]
        self.max_retries = max(1, max_retries)
        self.max_size = max_size
        self.max_size_video = max_size_video
        self.quality = max(1, min(100, quality))
        self.fps = fps
        self.nframes = nframes
        self.service_tier = service_tier

        if timeout is None and service_tier == "flex":
            timeout = 600.0
        self.timeout = timeout

        kwargs.pop("device_map", None)
        kwargs.pop("device", None)

        self.kwargs = kwargs
        if "seed" not in self.kwargs:
            self.kwargs["seed"] = 42
        if "temperature" not in self.kwargs:
            self.kwargs["temperature"] = 0.0

        self.cache_file_name = os.path.join(
            cache_dir or ".cache", f"openai_cache_{model_name.replace('-', '_')}.db"
        )

        logger.info(f"Initialized OpenAIModel with {self.model_name}")

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
            img_rgb = img.convert("RGB")
            img_rgb.save(buf, format="PNG", optimize=True)
            return hashlib.md5(buf.getvalue()).hexdigest()
        except Exception:
            return f"pil-id-{id(img)}"

    def _manage_cache(self, key: Tuple, value: Any) -> None:
        with _CACHE_LOCK:
            _CACHE[key] = value
            _CACHE.move_to_end(key)
            if len(_CACHE) > _MAX_CACHE_SIZE:
                _CACHE.popitem(last=False)

    def _preprocess_prompt(self, prompt: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompt_copy = copy.deepcopy(prompt)
        content_items = [(i, item) for i, item in enumerate(prompt_copy)]
        max_workers = min(len(content_items), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._process_content, item["content"]): i
                for i, item in content_items
            }
            for future in as_completed(future_to_index):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to process content item: {e}")
        return prompt_copy

    def encode_image(self, image: Union[str, Path, Image.Image]) -> str:
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
            raise OpenAIModelError(f"Failed to encode image: {e}") from e

    def encode_video(self, video_path: Union[str, Path]) -> List[str]:
        video_path = self._validate_file_path(video_path)
        cache_key = (
            str(video_path), self.max_size_video, self.quality, self.fps, self.nframes,
        )
        with _CACHE_LOCK:
            if cache_key in _CACHE:
                _CACHE.move_to_end(cache_key)
                return _CACHE[cache_key]
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)
            if total_frames == 0:
                raise OpenAIModelError(f"Video file appears to be empty: {video_path}")
            sample_indices = self._calculate_sample_indices(vr, total_frames)
            base64_frames = []
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
                raise OpenAIModelError(f"No frames extracted from: {video_path}")
            self._manage_cache(cache_key, base64_frames)
            return base64_frames
        except Exception as e:
            raise OpenAIModelError(f"Failed to encode video: {e}") from e

    def _calculate_sample_indices(self, vr: VideoReader, total_frames: int) -> List[int]:
        if self.fps is not None:
            video_fps = vr.get_avg_fps()
            if video_fps <= 0:
                raise OpenAIModelError("Cannot determine video FPS")
            frame_interval = max(1, int(video_fps / self.fps))
            sample_indices = list(range(0, total_frames, frame_interval))
        elif self.nframes is not None:
            if self.nframes >= total_frames:
                sample_indices = list(range(total_frames))
            else:
                indices = [
                    int(i * (total_frames - 1) / (self.nframes - 1))
                    for i in range(self.nframes)
                ]
                sample_indices = sorted(list(set(indices)))
        else:
            video_fps = vr.get_avg_fps()
            frame_interval = max(1, int(video_fps)) if video_fps > 0 else 30
            sample_indices = list(range(0, total_frames, frame_interval))
        if sample_indices and sample_indices[0] != 0:
            sample_indices.insert(0, 0)
        if sample_indices and sample_indices[-1] != total_frames - 1:
            sample_indices.append(total_frames - 1)
        return sorted(list(set(sample_indices)))

    def _process_content(
        self, content: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    ) -> None:
        if isinstance(content, str):
            return
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list):
            raise ValueError("content must be a str, dict, or list")

        media_tasks: List[Tuple[int, str, Any]] = []
        for i, item in enumerate(content):
            if not isinstance(item, dict):
                raise ValueError(f"Content item must be a dict, got {type(item)}")
            t = item.get("type")
            if t == "text" and "text" in item:
                content[i] = {"type": "text", "text": item["text"]}
            elif t == "image" and "image" in item:
                media_tasks.append((i, "image", item["image"]))
            elif t == "video" and "video" in item:
                media_tasks.append((i, "video", item["video"]))
            else:
                raise ValueError(f"Unsupported media item at index {i}: {item}")

        if not media_tasks:
            return

        max_workers = min(len(media_tasks), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            for task in media_tasks:
                i, media_type, file = task
                if media_type == "image":
                    fut = executor.submit(self.encode_image, file)
                elif media_type == "video":
                    fut = executor.submit(self.encode_video, file)
                else:
                    raise ValueError(f"Unsupported media type: {media_type}")
                future_to_task[fut] = task

            results = []
            for fut in as_completed(future_to_task):
                i, media_type, _ = future_to_task[fut]
                try:
                    if media_type == "image":
                        results.append((i, "image", fut.result()))
                    elif media_type == "video":
                        results.append((i, "video", fut.result()))
                except Exception as e:
                    logger.error(f"Failed to process {media_type} at index {i}: {e}")
                    results.append((i, "error", None))

            results.sort(key=lambda x: x[0])
            shift = 0
            for i, media_type, result in results:
                if media_type == "error":
                    continue
                elif media_type == "image":
                    content[i + shift] = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{result}"},
                    }
                elif media_type == "video":
                    image_items = [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                        }
                        for frame in result
                    ]
                    content[i + shift : i + shift + 1] = image_items
                    shift += len(image_items) - 1

    def _normalize_prompt(
        self, prompt: Union[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    @cache_response
    @dynamic_retry_decorator
    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        text_format: Optional[type] = None,
        **kwargs,
    ) -> Any:
        prompt_copy = copy.deepcopy(self._normalize_prompt(prompt))
        processed_prompt = self._preprocess_prompt(prompt_copy)
        try:
            api_kwargs = {**self.kwargs, **kwargs}
            if "gpt-5" in self.model_name and api_kwargs.get("temperature", 0) == 0:
                api_kwargs["temperature"] = 1

            if text_format is not None:
                if hasattr(self.sync_client.chat.completions, "parse"):
                    response = self.sync_client.chat.completions.parse(
                        model=self.model_name,
                        messages=processed_prompt,
                        response_format=text_format,
                        service_tier=self.service_tier,
                        timeout=self.timeout,
                        **api_kwargs,
                    )
                    return response.choices[0].message.parsed
                elif hasattr(self.sync_client.beta.chat.completions, "parse"):
                    response = self.sync_client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=processed_prompt,
                        response_format=text_format,
                        service_tier=self.service_tier,
                        timeout=self.timeout,
                        **api_kwargs,
                    )
                    return response.choices[0].message.parsed
                else:
                    raise NotImplementedError("chat.completions.parse not found")

            response = self.sync_client.chat.completions.create(
                model=self.model_name,
                messages=processed_prompt,
                service_tier=self.service_tier,
                timeout=self.timeout,
                **api_kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise OpenAIModelError(f"Failed to get completion: {e}") from e

    def generate_batch(
        self,
        batch_prompts: List[Union[str, List[Dict[str, Any]]]],
        text_format: Optional[type] = None,
    ) -> List[Any]:
        batch_prompts_copy = [
            copy.deepcopy(self._normalize_prompt(prompt)) for prompt in batch_prompts
        ]
        if not batch_prompts_copy:
            return []
        max_workers = min(len(batch_prompts_copy), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            processed_prompts = list(
                executor.map(self._preprocess_prompt, batch_prompts_copy)
            )
        try:
            return asyncio.run(
                self.async_generation(processed_prompts, text_format=text_format)
            )
        except Exception as e:
            raise OpenAIModelError(f"Batch generation failed: {e}") from e

    @async_cache_response
    @dynamic_retry_decorator
    async def _generate_single_prompt(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        text_format: Optional[type] = None,
    ) -> Any:
        try:
            prompt = self._normalize_prompt(prompt)
            api_kwargs = {**self.kwargs}
            if "gpt-5" in self.model_name and api_kwargs.get("temperature", 0) == 0:
                api_kwargs["temperature"] = 1

            if text_format is not None:
                if hasattr(self.async_client.chat.completions, "parse"):
                    response = await self.async_client.chat.completions.parse(
                        model=self.model_name,
                        messages=prompt,
                        response_format=text_format,
                        service_tier=self.service_tier,
                        timeout=self.timeout,
                        **api_kwargs,
                    )
                    return response.choices[0].message.parsed
                elif hasattr(self.async_client.beta.chat.completions, "parse"):
                    response = await self.async_client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=prompt,
                        response_format=text_format,
                        service_tier=self.service_tier,
                        timeout=self.timeout,
                        **api_kwargs,
                    )
                    return response.choices[0].message.parsed
                else:
                    raise NotImplementedError("chat.completions.parse not found")

            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                service_tier=self.service_tier,
                timeout=self.timeout,
                **api_kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise OpenAIModelError(f"Failed to get async completion: {e}") from e

    async def async_generation(
        self,
        batch_prompts: List[Union[str, List[Dict[str, Any]]]],
        chunk_size: int = 50,
        text_format: Optional[type] = None,
    ) -> List[Any]:
        responses = []
        total_chunks = (len(batch_prompts) + chunk_size - 1) // chunk_size
        for i in range(0, len(batch_prompts), chunk_size):
            chunk_num = (i // chunk_size) + 1
            batch = batch_prompts[i : i + chunk_size]
            tasks = [
                self._generate_single_prompt(prompt, text_format) for prompt in batch
            ]
            try:
                batch_responses = await tqdm_asyncio.gather(
                    *tasks, desc=f"Chunk {chunk_num}"
                )
                responses.extend(batch_responses)
            except Exception as e:
                logger.error(f"Error in chunk {chunk_num}: {e}")
                raise
        return responses

    def __repr__(self) -> str:
        return f"OpenAIModel(model_name='{self.model_name}', kwargs={self.kwargs})"


class OpenAIModelFast(OpenAIModel):
    """Optimized OpenAIModel with faster async generation."""

    @async_cache_response_fast
    @dynamic_retry_decorator
    async def generate_async(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        text_format: Optional[type] = None,
        **kwargs,
    ) -> Any:
        try:
            prompt = self._normalize_prompt(prompt)
            processed_prompt = self._preprocess_prompt(prompt)
            api_kwargs = {**self.kwargs, **kwargs}
            if "gpt-5-mini" in self.model_name and api_kwargs.get("temperature", 0) == 0:
                api_kwargs["temperature"] = 1

            if text_format is not None:
                if hasattr(self.async_client.chat.completions, "parse"):
                    response = await self.async_client.chat.completions.parse(
                        model=self.model_name,
                        messages=processed_prompt,
                        response_format=text_format,
                        service_tier=self.service_tier,
                        timeout=self.timeout,
                        **api_kwargs,
                    )
                    return response.choices[0].message.parsed
                elif hasattr(self.async_client.beta.chat.completions, "parse"):
                    response = await self.async_client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=processed_prompt,
                        response_format=text_format,
                        service_tier=self.service_tier,
                        timeout=self.timeout,
                        **api_kwargs,
                    )
                    return response.choices[0].message.parsed
                else:
                    raise NotImplementedError("chat.completions.parse not found")

            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=processed_prompt,
                service_tier=self.service_tier,
                timeout=self.timeout,
                **api_kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise OpenAIModelError(f"Failed to get async completion: {e}") from e
