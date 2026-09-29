import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Union

import torch
from tqdm import tqdm

from .embedding import EmbeddingModel
from .types import CaptionEntry, transform_timestamp

logger = logging.getLogger(__name__)


class MultiKeyMemory:
    """Episodic index over 30-second clips, one embedding per retrieval key.

    A clip scores as the maximum similarity over its active key types (MaxSim).
    """

    ALL_KEY_TYPES = ["event", "dialogue", "object", "summary"]

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        cache_dir: Optional[str] = None,
    ):
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir or ".cache/multi_keys"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.captions: List[CaptionEntry] = []
        self.keys: Dict[str, List[str]] = {k: [] for k in self.ALL_KEY_TYPES}
        self.embeddings: Dict[str, Optional[torch.Tensor]] = {
            k: None for k in self.ALL_KEY_TYPES
        }
        self.indexed_time: int = 0
        self.active_key_types: List[str] = list(self.ALL_KEY_TYPES)

    def set_active_keys(self, key_types: List[str]) -> None:
        """Restrict retrieval to a subset of the key types. Default: all four."""
        invalid = set(key_types) - set(self.ALL_KEY_TYPES)
        if invalid:
            raise ValueError(
                f"Invalid key type(s): {sorted(invalid)}. Valid: {self.ALL_KEY_TYPES}"
            )
        self.active_key_types = list(key_types)
        logger.info(f"Active key types set to: {self.active_key_types}")

    @staticmethod
    def _remap_video_path(video_path: Optional[str], video_root: Optional[str]) -> Optional[str]:
        """Re-root a stored ``video_path`` under ``video_root``.

        Keeps the last three components (``{subject}/{DAY}/{clip}.mp4``), and
        falls back to the stored path when the remapped one does not exist.
        """
        if not video_path or not video_root:
            return video_path
        parts = video_path.replace("\\", "/").strip("/").split("/")
        candidate = os.path.join(video_root, *parts[-3:])
        if os.path.exists(candidate):
            return candidate
        if os.path.exists(video_path):
            return video_path
        return candidate

    def load_captions_from_file(
        self, file_path: str, video_root: Optional[str] = None
    ) -> None:
        """Load a ``*_4_key.json`` file, re-rooting video paths if asked."""
        logger.info(f"Loading episodic keys from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.captions = []
        self.keys = {k: [] for k in self.ALL_KEY_TYPES}

        for idx, entry in enumerate(data):
            caption_id = f"30sec_{idx}"
            date_value = entry.get("date", "")
            date_str = str(date_value) if date_value else ""

            caption_entry = CaptionEntry(
                id=caption_id,
                text=entry.get("text", ""),
                start_time=str(entry.get("start_time", "")),
                end_time=str(entry.get("end_time", "")),
                date=date_str,
                video_path=self._remap_video_path(entry.get("video_path"), video_root),
            )
            self.captions.append(caption_entry)
            self.keys["event"].append(entry.get("event_key", ""))
            self.keys["dialogue"].append(entry.get("dialogue_key", ""))
            self.keys["object"].append(entry.get("object_key", ""))
            self.keys["summary"].append(entry.get("sum_key", ""))

        logger.info(f"Loaded {len(self.captions)} captions")

    def index(self, until_time: int) -> None:
        """Compute embeddings for all keys if not already cached."""
        if not self.captions:
            logger.warning("No captions loaded to index")
            return

        for key_type in self.ALL_KEY_TYPES:
            cache_filename = f"30sec_{key_type}_key_embeddings.pt"
            cache_path = os.path.join(self.cache_dir, cache_filename)

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
                    batch_keys = keys_to_embed[i : i + batch_size]
                    batch_emb_np = self.embedding_model.encode_text(batch_keys)
                    batch_emb = torch.from_numpy(batch_emb_np)
                    embeddings.append(batch_emb)

                self.embeddings[key_type] = torch.cat(embeddings, dim=0)
                torch.save(self.embeddings[key_type], cache_path)
                logger.info(f"Saved {key_type} embeddings to {cache_path}")

        self.indexed_time = until_time

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        until_time: Optional[int] = None,
        exclude_texts: Optional[Set[str]] = None,
        as_context: bool = True,
        return_dict: bool = False,
        include_key_content: bool = False,
    ) -> Union[str, List[CaptionEntry], Dict[str, Any]]:
        """Retrieve top-k captions using MaxSim across key types."""
        if not self.captions:
            logger.warning("No captions loaded.")
            if return_dict:
                return {"retrieved_content": [], "all_candidates": []}
            return "" if as_context else []

        query_emb_np = self.embedding_model.encode_text([query])
        device = self.embeddings["event"].device if self.embeddings["event"] is not None else "cpu"
        query_emb = torch.from_numpy(query_emb_np).to(device)

        exclude_texts = exclude_texts or set()

        if any(v is None for v in self.embeddings.values()):
            logger.warning("Memory not indexed.")
            if return_dict:
                return {"retrieved_content": [], "all_candidates": []}
            return "" if as_context else []

        # Filter by time
        valid_indices = []
        for i, entry in enumerate(self.captions):
            if until_time is None or entry.timestamp_int[1] <= until_time:
                valid_indices.append(i)

        if not valid_indices:
            if return_dict:
                return {"retrieved_content": [], "all_candidates": []}
            return "" if as_context else []

        # MaxSim across active key types
        max_scores = torch.full((len(valid_indices),), -1.0, device=query_emb.device)
        best_key_types = [""] * len(valid_indices)

        for key_type in self.active_key_types:
            filtered_embeddings = self.embeddings[key_type][valid_indices]
            similarities = torch.nn.functional.cosine_similarity(
                query_emb, filtered_embeddings
            )
            is_better = similarities > max_scores
            max_scores = torch.where(is_better, similarities, max_scores)
            better_indices = torch.nonzero(is_better).squeeze(-1).tolist()
            for idx in better_indices:
                best_key_types[idx] = key_type

        # Rank and deduplicate
        candidates = []
        for i, score in enumerate(max_scores):
            global_idx = valid_indices[i]
            entry = self.captions[global_idx]
            if entry.text in exclude_texts:
                continue
            candidates.append({
                "global_idx": global_idx,
                "entry": entry,
                "key_type": best_key_types[i],
                "score": score.item()
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)

        unique_candidates = []
        seen_texts = set()
        for cand in candidates:
            if cand["entry"].text not in seen_texts:
                unique_candidates.append(cand)
                seen_texts.add(cand["entry"].text)
            if len(unique_candidates) >= top_k:
                break

        retrieved_entries = [c["entry"] for c in unique_candidates]

        all_candidates = []
        for cand in unique_candidates:
            entry = cand["entry"]
            idx = cand["global_idx"]

            start_ts, end_ts = entry.timestamp_int
            start_time = transform_timestamp(str(start_ts))
            end_time = transform_timestamp(str(end_ts))

            key_content = self.keys[cand["key_type"]][idx]

            all_candidates.append({
                "timestamp": f"[{start_time} - {end_time}]",
                "text": entry.text,
                "key_type": cand["key_type"],
                "key_content": key_content,
                "similarity": round(cand["score"], 4),
            })

        if return_dict:
            retrieved_content = []
            for idx_r, cand in enumerate(all_candidates, 1):
                entry = unique_candidates[idx_r - 1]["entry"]
                start_ts, end_ts = entry.timestamp_int
                start_time = transform_timestamp(str(start_ts))
                end_time = transform_timestamp(str(end_ts))
                if include_key_content:
                    key_content = cand["key_content"]
                    retrieved_content.append(
                        f"top{idx_r}: [{start_time} - {end_time}]\n"
                        f"key-point: {key_content}\n"
                        f"caption: {entry.text}"
                    )
                else:
                    retrieved_content.append(
                        f"top{idx_r}: [{start_time} - {end_time}]\n{entry.text}"
                    )
            return {
                "retrieved_content": retrieved_content,
                "all_candidates": all_candidates,
                "retrieved_entries": retrieved_entries,
            }

        if as_context:
            return self.format_as_context(retrieved_entries)
        return retrieved_entries

    def format_as_context(self, entries: List[CaptionEntry]) -> str:
        lines = []
        for idx, entry in enumerate(entries, 1):
            start_ts, end_ts = entry.timestamp_int
            start_time = transform_timestamp(str(start_ts))
            end_time = transform_timestamp(str(end_ts))
            lines.append(f"top{idx}: [{start_time} - {end_time}]\n{entry.text}")
        return "\n\n".join(lines)

    def reset_index(self) -> None:
        self.indexed_time = 0
        self.embeddings = {k: None for k in self.embeddings}
