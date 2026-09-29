"""Iterative search/answer over multi-key episodic memory."""

import copy
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image

from .embedding import EmbeddingModel
from .llm import LLMModel, PromptTemplateManager
from .llm.templates.neighbor_filter import (
    get_neighbor_filter_prompt,
    format_neighbor_filter_content,
)
from .types import CaptionEntry, transform_timestamp
from .multikey_memory import MultiKeyMemory

logger = logging.getLogger(__name__)


@dataclass
class ClipContext:
    """A single clip's context with original caption, filtered info, and frames."""
    original_caption: str
    filtered_info: str
    frames: List[Image.Image] = field(default_factory=list)
    video_path: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    rank: int = 0


@dataclass
class ReasoningOutput:
    decision: str
    search_query: Optional[str] = None


@dataclass
class RetrievedItem:
    """Stores retrieval results for one round."""
    content: List[ClipContext]
    query: str
    round_num: int


@dataclass
class QAResult:
    question: str
    answer: str
    retrieved_items: List[RetrievedItem]
    round_history: List[Dict[str, Any]]
    num_rounds: int
    raw_responses: List[Dict[str, Any]]


class MeritMemory:
    """Rounds of search/answer over a MultiKeyMemory."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        respond_llm_model: Optional[LLMModel] = None,
        prompt_template_manager: Optional[PromptTemplateManager] = None,
        cache_dir: Optional[str] = None,
        max_rounds: int = 5,
        max_errors: int = 5,
        debug_log_count: int = 5,
    ):
        self.embedding_model = embedding_model
        self.respond_llm_model = respond_llm_model
        self.prompt_template_manager = prompt_template_manager or PromptTemplateManager()
        self.max_rounds = max_rounds
        self.max_errors = max_errors
        self.debug_log_count = debug_log_count

        self.retriever = MultiKeyMemory(
            embedding_model=embedding_model,
            cache_dir=cache_dir,
        )

        self.indexed_time: int = 0
        self.top_k: int = 10

        self._all_entries: List[CaptionEntry] = []
        self._entries_by_timestamp: Dict[Tuple[int, int], CaptionEntry] = {}

        self._qa_count: int = 0

    def load_keys(
        self, file_path: str, video_root: Optional[str] = None
    ) -> None:
        """Load episodic keys from file, optionally re-rooting video paths."""
        self.retriever.load_captions_from_file(file_path, video_root=video_root)

        self._all_entries = list(self.retriever.captions)
        self._all_entries.sort(key=lambda e: e.timestamp_int[0])
        self._entries_by_timestamp = {
            e.timestamp_int: e for e in self._all_entries
        }

    def index(self, until_time: int) -> None:
        """Index episodic memory up to the specified time."""
        if self.indexed_time >= until_time:
            return
        self.retriever.index(until_time)
        self.indexed_time = until_time

    def set_top_k(self, top_k: Optional[int] = None) -> None:
        if top_k is not None:
            self.top_k = top_k

    def set_active_keys(self, key_types: List[str]) -> None:
        self.retriever.set_active_keys(key_types)

    def _find_neighbor_clips(
        self,
        center_entry: CaptionEntry,
        num_before: int = 2,
        num_after: int = 2,
        until_time: Optional[int] = None,
    ) -> List[CaptionEntry]:
        
        center_ts = center_entry.timestamp_int
        center_idx = None
        for i, entry in enumerate(self._all_entries):
            if entry.timestamp_int == center_ts:
                center_idx = i
                break
        if center_idx is None:
            return [center_entry]

        def observable(entry: CaptionEntry) -> bool:
            return until_time is None or entry.timestamp_int[1] <= until_time

        neighbors = []
        for i in range(num_before, 0, -1):
            idx = center_idx - i
            if idx >= 0 and observable(self._all_entries[idx]):
                neighbors.append(self._all_entries[idx])
        neighbors.append(center_entry)
        for i in range(1, num_after + 1):
            idx = center_idx + i
            if idx < len(self._all_entries) and observable(self._all_entries[idx]):
                neighbors.append(self._all_entries[idx])
        return neighbors

    def _extract_frames_from_video(
        self, video_path: str, num_frames: int,
    ) -> List[Image.Image]:
        if not video_path or not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return []
        if num_frames <= 0:
            return []
        try:
            from decord import VideoReader, cpu
        except ImportError:
            logger.error("decord is required for frame extraction")
            return []
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            if total_frames == 0:
                return []
            if num_frames == 1:
                indices = [total_frames // 2]
            else:
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
            video_frames = vr.get_batch(indices).asnumpy()
            return [Image.fromarray(frame_data) for frame_data in video_frames]
        except Exception as e:
            logger.error(f"Failed to extract frames from {video_path}: {e}")
            return []

    def _get_frame_count(self, rank: int, round_num: int) -> int:
        if round_num == 1:
            return 6
        elif round_num == 2:
            return 6 if rank < 5 else 0
        else:
            return 1

    def _filter_neighbor_captions_batch(
        self,
        entries: List[CaptionEntry],
        query: str,
        choices: Dict[str, str],
        question_id: Optional[str] = None,
        until_time: Optional[int] = None,
    ) -> List[str]:
        num_clips = len(entries)
        all_neighbor_captions: List[List[str]] = []

        for entry in entries:
            neighbors = self._find_neighbor_clips(
                entry, num_before=2, num_after=2, until_time=until_time
            )
            neighbor_captions = []
            for neighbor in neighbors:
                start_ts, end_ts = neighbor.timestamp_int
                start_time = transform_timestamp(str(start_ts))
                end_time = transform_timestamp(str(end_ts))
                caption_text = f"[{start_time} - {end_time}] {neighbor.text}"
                neighbor_captions.append(caption_text)
            all_neighbor_captions.append(neighbor_captions)

        filter_prompt = get_neighbor_filter_prompt()
        user_content = format_neighbor_filter_content(
            query=query,
            choices=choices,
            all_neighbor_captions=all_neighbor_captions,
        )
        filter_prompt.append({"role": "user", "content": user_content})

        try:
            response = self.respond_llm_model.generate(filter_prompt)
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            result = []
            for clip_idx in range(1, num_clips + 1):
                info = data.get(f"clip_{clip_idx}", "")
                if not info or info.lower().strip() in [
                    "no relevant information found",
                    "no relevant information",
                    "none", "n/a", "",
                ]:
                    result.append("")
                else:
                    result.append(info)

            should_log = self._qa_count <= self.debug_log_count
            if should_log:
                logger.info(f"\n{'~'*60}")
                logger.info(f"FILTER DEBUG - QA #{self._qa_count} | ID: {question_id}")
                logger.info(f"{'~'*60}")
                logger.info(f"[INPUT] {num_clips} clips x 5 neighbors = {sum(len(c) for c in all_neighbor_captions)} captions")
                for i, info in enumerate(result, 1):
                    if info:
                        logger.info(f"  clip_{i}: {info[:100]}{'...' if len(info) > 100 else ''}")
                    else:
                        logger.info(f"  clip_{i}: (empty)")
                logger.info(f"{'~'*60}\n")

            return result

        except Exception as e:
            logger.error(f"Batch neighbor filtering failed: {e}")
            return [""] * num_clips

    def _build_clip_contexts(
        self,
        entries: List[CaptionEntry],
        filtered_infos: List[str],
        round_num: int,
    ) -> List[ClipContext]:
        contexts = []
        for rank, (entry, filtered_info) in enumerate(zip(entries, filtered_infos)):
            start_ts, end_ts = entry.timestamp_int
            start_time = transform_timestamp(str(start_ts))
            end_time = transform_timestamp(str(end_ts))
            original_caption = f"top{rank+1}: [{start_time} - {end_time}]\n{entry.text}"
            frames = []
            video_path = entry.video_path if hasattr(entry, 'video_path') else None
            if video_path:
                num_frames = self._get_frame_count(rank, round_num)
                frames = self._extract_frames_from_video(video_path, num_frames)
            contexts.append(ClipContext(
                original_caption=original_caption,
                filtered_info=filtered_info,
                frames=frames,
                video_path=video_path,
                start_time=str(start_ts),
                end_time=str(end_ts),
                rank=rank + 1,
            ))
        return contexts

    def _parse_reasoning_response(self, response: str) -> ReasoningOutput:
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            decision = data.get("decision", "answer").lower()
            search_query = data.get("search_query")
            return ReasoningOutput(decision=decision, search_query=search_query)
        except Exception:
            return ReasoningOutput(decision="answer")

    def _format_round_history_text_only(self, rounds: List[Dict[str, Any]]) -> str:
        if not rounds:
            return "[]"
        lines = []
        for r in rounds:
            retrieved_content = r["retrieved_content"]
            if isinstance(retrieved_content, list):
                retrieved_content = "\n\n".join(retrieved_content)
            round_str = f"""### Round {r['round_num']}
Decision: {r['decision']}
Search Query: {r['search_query']}
Retrieved:
{retrieved_content}"""
            lines.append(round_str)
        return "\n\n".join(lines)

    def _format_clip_contexts_as_text(self, contexts: List[ClipContext]) -> List[str]:
        result = []
        for ctx in contexts:
            if ctx.filtered_info:
                text = f"{ctx.original_caption}\n\n[Relevant Info]\n{ctx.filtered_info}"
            else:
                text = ctx.original_caption
            result.append(text)
        return result

    def _build_qa_content(
        self, full_query: str, retrieved_items: List[RetrievedItem], choices: Optional[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        qa_content = [{"type": "text", "text": full_query + "\n\nContext:\n"}]
        for item in retrieved_items:
            qa_content.append({"type": "text", "text": f"## Round {item.round_num} Results:\n"})
            for ctx in item.content:
                qa_content.append({"type": "text", "text": f"{ctx.original_caption}\n"})
                if ctx.filtered_info:
                    qa_content.append({"type": "text", "text": f"\n[Relevant Info]\n{ctx.filtered_info}\n"})
                for img in ctx.frames:
                    qa_content.append({"type": "image", "image": img})
        if choices:
            qa_content.append({
                "type": "text",
                "text": "\nPlease provide only the final answer from the choices given (e.g., A, B, C, or D).",
            })
        return qa_content

    def _log_prompt_debug(
        self, question_id: str, stage: str, round_num: int,
        messages: List[Dict[str, Any]],
    ) -> None:
        if self._qa_count > self.debug_log_count:
            return
        logger.info(f"\n{'='*80}")
        logger.info(f"DEBUG LOG - QA #{self._qa_count} | ID: {question_id} | Stage: {stage} | Round: {round_num}")
        logger.info(f"{'='*80}")
        for msg_idx, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            logger.info(f"\n--- Message {msg_idx} | Role: {role} ---")
            if role == "system":
                if isinstance(content, str):
                    logger.info(f"[SYSTEM PROMPT ({len(content)} chars, showing first 200)]:")
                    logger.info(f"  {content[:200]}...")
                continue
            if isinstance(content, str):
                logger.info(f"[TEXT CONTENT ({len(content)} chars)]")
                lines = content.split('\n')
                for line in lines[:5]:
                    logger.info(f"  {line}")
                if len(lines) > 5:
                    logger.info(f"  ... ({len(lines) - 5} more lines)")
            elif isinstance(content, list):
                text_count = sum(1 for item in content if item.get("type") == "text")
                image_count = sum(1 for item in content if item.get("type") == "image")
                logger.info(f"[MULTIMODAL CONTENT: {text_count} text items, {image_count} images]")
        logger.info(f"\n{'='*80}\n")

    def _retrieve(
        self, query: str, top_k: Optional[int] = None,
        retrieved_set: Optional[Set[str]] = None,
        until_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        top_k = self.top_k
        result = self.retriever.retrieve(
            query=query, top_k=top_k, as_context=False,
            return_dict=True, until_time=until_time,
            exclude_texts=retrieved_set,
        )
        if not result or not result.get("retrieved_entries"):
            return {"retrieved_content": [], "all_candidates": [], "entries": []}

        entries = result["retrieved_entries"]
        if retrieved_set is not None:
            for entry in entries:
                retrieved_set.add(entry.text)

        retrieved_content = []
        for idx, entry in enumerate(entries[:self.top_k], 1):
            start_ts, end_ts = entry.timestamp_int
            start_time = transform_timestamp(str(start_ts))
            end_time = transform_timestamp(str(end_ts))
            retrieved_content.append(
                f"top{idx}: [{start_time} - {end_time}]\n{entry.text}"
            )
        return {
            "retrieved_content": retrieved_content,
            "all_candidates": result.get("all_candidates", []),
            "entries": entries[:self.top_k],
        }

    def answer(
        self, query: str, choices: Optional[Dict[str, str]] = None,
        until_time: Optional[int] = None,
        question_id: Optional[str] = None,
    ) -> QAResult:
        """Answer one question."""
        self._qa_count += 1
        should_log = self._qa_count <= self.debug_log_count

        if until_time and until_time > self.indexed_time:
            self.index(until_time)

        full_query = f"Query: {query}"
        if choices:
            choices_str = " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))
            full_query += f"\nChoices: {choices_str}"

        qa_full_query = f"Query: {query}"
        if until_time:
            query_time_str = transform_timestamp(str(until_time))
            qa_full_query += f"\nQuery Time: {query_time_str}"
        if choices:
            choices_str = " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))
            qa_full_query += f"\nChoices: {choices_str}"

        retrieved_set: Set[str] = set()
        retrieved_items: List[RetrievedItem] = []
        round_history: List[Dict[str, Any]] = []
        raw_responses: List[Dict[str, Any]] = []

        reasoning_prompt = self.prompt_template_manager.render(
            "memory_retrieval_round_reasoning"
        )

        round_num = 0
        err_count = 0

        while round_num < self.max_rounds and err_count < self.max_errors:
            round_num += 1
            logger.info(f"ID {question_id}: Round {round_num} starting...")

            history_str = self._format_round_history_text_only(round_history)
            user_content = f"""{full_query}

Round History:
{history_str}

Task:
Step 1: Decide whether to "search" or "answer".
Step 2 (only if search): Form a keyword(phrase)-style search query."""

            reasoning_messages = copy.deepcopy(reasoning_prompt)
            reasoning_messages.append({"role": "user", "content": user_content})

            if should_log:
                self._log_prompt_debug(question_id, "reasoning", round_num, reasoning_messages)

            try:
                response = self.respond_llm_model.generate(reasoning_messages)
                reasoning_output = self._parse_reasoning_response(response)
                logger.info(f"ID {question_id}: Round {round_num} Decision: {reasoning_output.decision}")
            except Exception as e:
                logger.error(f"Reasoning failed: {e}")
                err_count += 1
                continue

            raw_responses.append({
                "type": "reasoning", "round_num": round_num,
                "raw_response": response, "parsed_decision": reasoning_output.decision,
            })

            if reasoning_output.decision == "answer":
                break

            search_query = reasoning_output.search_query
            if not search_query:
                logger.warning(f"ID {question_id}: Round {round_num} Search decision but no query.")
                err_count += 1
                continue

            logger.info(f"ID {question_id}: Round {round_num} Query: {search_query}")

            retrieved_data = self._retrieve(
                search_query, retrieved_set=retrieved_set, until_time=until_time,
            )

            entries = retrieved_data.get("entries", [])

            if not entries:
                logger.warning(f"ID {question_id}: Round {round_num} No entries retrieved")
                round_history.append({
                    "round_num": round_num, "decision": "search",
                    "search_query": search_query, "retrieved_content": [],
                    "all_candidates": [],
                })
                continue

            logger.info(f"ID {question_id}: Round {round_num} Batch filtering neighbor captions...")
            filtered_infos = self._filter_neighbor_captions_batch(
                entries, query, choices or {}, question_id, until_time=until_time
            )

            clip_contexts = self._build_clip_contexts(
                entries, filtered_infos, round_num
            )

            total_frames = sum(len(ctx.frames) for ctx in clip_contexts)
            num_with_info = sum(1 for info in filtered_infos if info)
            logger.info(
                f"ID {question_id}: Round {round_num} Retrieved {len(entries)} clips, "
                f"{total_frames} frames, {num_with_info}/{len(entries)} with relevant info"
            )

            retrieved_items.append(
                RetrievedItem(content=clip_contexts, query=search_query, round_num=round_num)
            )

            clip_texts = self._format_clip_contexts_as_text(clip_contexts)

            round_history.append({
                "round_num": round_num, "decision": "search",
                "search_query": search_query, "retrieved_content": clip_texts,
                "filtered_infos": filtered_infos,
                "all_candidates": retrieved_data.get("all_candidates", []),
            })

        # Final QA
        qa_prompt = self.prompt_template_manager.render("final_qa_egolife")
        qa_content = self._build_qa_content(qa_full_query, retrieved_items, choices)

        qa_messages = copy.deepcopy(qa_prompt)
        qa_messages.append({"role": "user", "content": qa_content})

        if should_log:
            self._log_prompt_debug(question_id, "qa", round_num, qa_messages)

        try:
            answer = self.respond_llm_model.generate(qa_messages)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = "Error"

        raw_responses.append({"type": "qa", "raw_response": answer})

        return QAResult(
            question=query, answer=answer, retrieved_items=retrieved_items,
            round_history=round_history, num_rounds=round_num,
            raw_responses=raw_responses,
        )

    def cleanup(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
