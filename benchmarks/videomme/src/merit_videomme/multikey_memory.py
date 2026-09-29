"""
VideoMME Episodic Memory module with multi-key retrieval.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def format_timestamp(ts: str) -> str:
    """Convert HHMMSS string to HH:MM:SS format."""
    ts = ts.zfill(6)
    return f"{ts[0:2]}:{ts[2:4]}:{ts[4:6]}"


@dataclass
class CaptionEntry:
    """Represents a single caption entry."""
    id: str
    text: str
    start_time: str
    end_time: str
    event_key: str = ""
    dialogue_key: str = ""
    object_key: str = ""
    sum_key: str = ""

    def to_display_str(self) -> str:
        """Format caption for display with time range."""
        return f"[{format_timestamp(self.start_time)} - {format_timestamp(self.end_time)}]"


class MultiKeyMemory:
    """
    Episodic Memory module for VideoMME.
    Uses pre-generated 4 keys (event, dialogue, object, summary) for retrieval.
    Each video has its own set of captions loaded from a JSON file.
    """

    ALL_KEY_TYPES = ["event", "dialogue", "object", "summary"]

    def __init__(
        self,
        embedding_model,
        cache_dir: Optional[str] = None,
        active_key_types: Optional[List[str]] = None,
    ):
        """
        Initialize episodic memory.

        Args:
            embedding_model: Model for encoding text to embeddings
            cache_dir: Directory for caching embeddings
            active_key_types: Subset of ALL_KEY_TYPES to retrieve with.
                              Default (None) uses all of them.
        """
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir or ".cache/multi_keys"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Set active key types (default: all 4 keys)
        if active_key_types is None:
            self.active_key_types = self.ALL_KEY_TYPES.copy()
        else:
            # Validate key types
            invalid_keys = set(active_key_types) - set(self.ALL_KEY_TYPES)
            if invalid_keys:
                raise ValueError(f"Invalid key types: {invalid_keys}. Valid options: {self.ALL_KEY_TYPES}")
            self.active_key_types = list(active_key_types)

        logger.info(f"EpisodicMemory using key types: {self.active_key_types}")

        # Current video's data
        self.current_video_id: Optional[str] = None
        self.captions: List[CaptionEntry] = []
        self.keys: Dict[str, List[str]] = {
            "event": [],
            "dialogue": [],
            "object": [],
            "summary": [],
        }
        self.embeddings: Dict[str, Optional[torch.Tensor]] = {
            "event": None,
            "dialogue": None,
            "object": None,
            "summary": None,
        }
        self.is_indexed: bool = False

    def load_captions_from_file(self, file_path: str, video_id: str) -> None:
        """Load captions and 4 keys from JSON file for a specific video."""
        logger.info(f"Loading episodic captions for video {video_id} from {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._reset()
        self.current_video_id = video_id

        for idx, entry in enumerate(data):
            caption_entry = CaptionEntry(
                id=f"{video_id}_{idx}",
                text=entry.get("text", ""),
                start_time=str(entry.get("start_time", "")),
                end_time=str(entry.get("end_time", "")),
                event_key=entry.get("event_key", ""),
                dialogue_key=entry.get("dialogue_key", ""),
                object_key=entry.get("object_key", ""),
                sum_key=entry.get("sum_key", ""),
            )
            self.captions.append(caption_entry)
            self.keys["event"].append(entry.get("event_key", ""))
            self.keys["dialogue"].append(entry.get("dialogue_key", ""))
            self.keys["object"].append(entry.get("object_key", ""))
            self.keys["summary"].append(entry.get("sum_key", ""))

        logger.info(f"Loaded {len(self.captions)} captions for video {video_id}")

    def _reset(self) -> None:
        """Reset memory state."""
        self.current_video_id = None
        self.captions = []
        self.keys = {"event": [], "dialogue": [], "object": [], "summary": []}
        self.embeddings = {"event": None, "dialogue": None, "object": None, "summary": None}
        self.is_indexed = False

    def index(self) -> None:
        """Compute embeddings for active key types only."""
        if not self.captions:
            logger.warning("No captions loaded to index")
            return

        if self.is_indexed:
            logger.debug("Already indexed, skipping")
            return

        video_id = self.current_video_id
        video_cache_dir = os.path.join(self.cache_dir, video_id)
        os.makedirs(video_cache_dir, exist_ok=True)

        for key_type in self.active_key_types:
            cache_path = os.path.join(video_cache_dir, f"{key_type}_embeddings.pt")

            if os.path.exists(cache_path):
                logger.info(f"Loading cached {key_type} embeddings from {cache_path}")
                self.embeddings[key_type] = torch.load(cache_path)
                if self.embeddings[key_type].shape[0] != len(self.captions):
                    logger.warning(f"Cached embeddings size mismatch, recomputing...")
                    self.embeddings[key_type] = None

            if self.embeddings[key_type] is None:
                keys_to_embed = self.keys[key_type]
                logger.info(f"Computing embeddings for {len(keys_to_embed)} {key_type} keys...")
                embeddings = []
                batch_size = 128

                for i in tqdm(range(0, len(keys_to_embed), batch_size), desc=f"Embedding {key_type}"):
                    batch_keys = keys_to_embed[i:i + batch_size]
                    batch_emb_np = self.embedding_model.encode_text(batch_keys)
                    batch_emb = torch.from_numpy(batch_emb_np)
                    embeddings.append(batch_emb)

                self.embeddings[key_type] = torch.cat(embeddings, dim=0)
                torch.save(self.embeddings[key_type], cache_path)
                logger.info(f"Saved {key_type} embeddings to {cache_path}")

        self.is_indexed = True

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        exclude_texts: Optional[Set[str]] = None,
        as_context: bool = True,
        return_dict: bool = False,
    ) -> Union[str, List[CaptionEntry], Dict[str, Any]]:
        """
        Retrieve top-k captions based on query similarity.
        Uses max similarity across all 4 key types.
        """
        if not self.captions or not self.is_indexed:
            logger.warning("Memory not loaded or indexed.")
            if return_dict:
                return {"retrieved_content": [], "all_candidates": [], "retrieved_entries": []}
            return "" if as_context else []

        exclude_texts = exclude_texts or set()

        # Get query embedding
        query_emb_np = self.embedding_model.encode_text([query])
        # Use the first active key type to get device
        first_key = self.active_key_types[0]
        device = self.embeddings[first_key].device
        query_emb = torch.from_numpy(query_emb_np).to(device)

        # Compute max similarity across active key types
        num_captions = len(self.captions)
        max_scores = torch.full((num_captions,), -1.0, device=device)
        best_key_types = [""] * num_captions

        for key_type in self.active_key_types:
            similarities = torch.nn.functional.cosine_similarity(
                query_emb, self.embeddings[key_type]
            )
            is_better = similarities > max_scores
            max_scores = torch.where(is_better, similarities, max_scores)

            better_indices = torch.nonzero(is_better).squeeze(-1).tolist()
            if isinstance(better_indices, int):
                better_indices = [better_indices]
            for idx in better_indices:
                best_key_types[idx] = key_type

        # Create candidates list
        candidates = []
        for i, score in enumerate(max_scores):
            entry = self.captions[i]
            if entry.text in exclude_texts:
                continue

            candidates.append({
                "idx": i,
                "entry": entry,
                "key_type": best_key_types[i],
                "score": score.item(),
            })

        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Select top-k unique texts
        unique_candidates = []
        seen_texts = set()

        for cand in candidates:
            if cand["entry"].text not in seen_texts:
                unique_candidates.append(cand)
                seen_texts.add(cand["entry"].text)

            if len(unique_candidates) >= top_k:
                break

        # Extract entries
        retrieved_entries = [c["entry"] for c in unique_candidates]

        if return_dict:
            retrieved_content = []
            all_candidates = []

            for rank, cand in enumerate(unique_candidates, 1):
                entry = cand["entry"]
                ts_display = f"[{format_timestamp(entry.start_time)} - {format_timestamp(entry.end_time)}]"
                retrieved_content.append(
                    f"top{rank}: {ts_display}\n{entry.text}"
                )
                all_candidates.append({
                    "timestamp": ts_display,
                    "text": entry.text,
                    "key_type": cand["key_type"],
                    "similarity": round(cand["score"], 4),
                })

            return {
                "retrieved_content": retrieved_content,
                "all_candidates": all_candidates,
                "retrieved_entries": retrieved_entries,
            }

        if as_context:
            return self.format_as_context(retrieved_entries)
        return retrieved_entries

    def format_as_context(self, entries: List[CaptionEntry]) -> str:
        """Format retrieved entries as context string."""
        lines = []
        for idx, entry in enumerate(entries, 1):
            lines.append(f"top{idx}: [{format_timestamp(entry.start_time)} - {format_timestamp(entry.end_time)}]\n{entry.text}")
        return "\n\n".join(lines)

    def cleanup(self) -> None:
        """Free memory."""
        self._reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
