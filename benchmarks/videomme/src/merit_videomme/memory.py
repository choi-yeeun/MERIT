"""MERIT memory for VideoMME."""

import copy
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from PIL import Image

from merit.llm import LLMModel, PromptTemplateManager
from merit.llm.templates.neighbor_filter import (
    get_neighbor_filter_prompt,
    format_neighbor_filter_content,
)

from .multikey_memory import MultiKeyMemory, CaptionEntry, format_timestamp

logger = logging.getLogger(__name__)


@dataclass
class ClipContext:
    """A single clip's context with original caption, filtered info, and frames."""
    original_caption: str
    filtered_info: str
    frames: List[Image.Image] = field(default_factory=list)
    rank: int = 0


@dataclass
class ReasoningOutput:
    decision: str
    search_query: Optional[str] = None


@dataclass
class RetrievedItem:
    """Stores retrieval results for one round as list of ClipContext pairs."""
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


class VideoMMEMemory:
    """Multi-key retrieval, batched neighbor filtering and iterative
    search/answer over one Video-MME video."""

    def __init__(
        self,
        embedding_model,
        respond_llm_model,
        prompt_template_manager=None,
        cache_dir=None,
        frame_dir=None,
        max_rounds=5,
        max_errors=5,
        debug_log_count=5,
    ):
        self.embedding_model = embedding_model
        self.respond_llm_model = respond_llm_model
        self.prompt_template_manager = prompt_template_manager or PromptTemplateManager()
        self.max_rounds = max_rounds
        self.max_errors = max_errors
        self.frame_dir = frame_dir

        # Initialize episodic memory
        self.retriever = MultiKeyMemory(
            embedding_model=embedding_model,
            cache_dir=cache_dir,
        )

        self.current_video_id: Optional[str] = None
        self.top_k: int = 10
        self.debug_log_count = debug_log_count

        # Sorted entries cache for neighbor lookup
        self._sorted_entries: List[CaptionEntry] = []
        self._qa_count: int = 0

    def load_video_data(self, video_id: str, caption_path: str) -> None:
        if self.current_video_id != video_id:
            logger.info(f"Loading data for video {video_id}")
            self.retriever.load_captions_from_file(caption_path, video_id)
            self.current_video_id = video_id
        self._sorted_entries = sorted(
            self.retriever.captions,
            key=lambda e: e.start_time,
        )

    def index(self) -> None:
        """Index episodic memory."""
        self.retriever.index()

    def set_top_k(self, top_k: Optional[int] = None) -> None:
        """Set top-k for episodic retrieval."""
        if top_k is not None:
            self.top_k = top_k

    def _load_frames_for_entry(self, entry: CaptionEntry, num_frames: int) -> List[Image.Image]:
        """
        Load pre-extracted frames for a caption entry.

        Args:
            entry: CaptionEntry with start_time (HHMMSS format)
            num_frames: Number of frames to sample

        Returns:
            List of PIL Image frames
        """
        clip_dir = os.path.join(
            self.frame_dir,
            self.current_video_id,
            f"{self.current_video_id}_{entry.start_time}",
        )

        if not os.path.isdir(clip_dir):
            logger.warning(f"Frame directory not found: {clip_dir}")
            return []

        # List and sort frame files
        frame_files = sorted([
            f for f in os.listdir(clip_dir) if f.endswith(".jpg")
        ])
        if not frame_files:
            logger.warning(f"No frame files in {clip_dir}")
            return []

        total_available = len(frame_files)
        num_frames = min(num_frames, total_available)

        if num_frames <= 0:
            return []

        if num_frames == 1:
            # Middle frame
            indices = [total_available // 2]
        else:
            # Uniform sampling
            indices = np.linspace(0, total_available - 1, num_frames, dtype=int).tolist()

        frames = []
        for idx in indices:
            frame_path = os.path.join(clip_dir, frame_files[idx])
            try:
                img = Image.open(frame_path).convert("RGB")
                frames.append(img)
            except Exception as e:
                logger.warning(f"Failed to load frame {frame_path}: {e}")

        return frames

    def _parse_reasoning_response(self, response: str) -> ReasoningOutput:
        """Parse LLM reasoning response (search/answer + search_query)."""
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            decision = data.get("decision", "answer").lower()
            search_query = data.get("search_query")

            return ReasoningOutput(decision=decision, search_query=search_query)
        except Exception as e:
            logger.warning(f"Failed to parse reasoning response: {e}")
            return ReasoningOutput(decision="answer")

    def _retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        retrieved_set: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Retrieve from episodic memory, returning entries."""
        top_k = top_k or self.top_k

        result = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            as_context=False,
            return_dict=True,
            exclude_texts=retrieved_set,
        )

        if not result or not result.get("retrieved_entries"):
            return {"retrieved_content": [], "all_candidates": [], "entries": []}

        entries = result["retrieved_entries"]

        # Update retrieved_set
        if retrieved_set is not None:
            for entry in entries:
                retrieved_set.add(entry.text)

        # Format text content
        retrieved_content = []
        for idx, entry in enumerate(entries, 1):
            retrieved_content.append(
                f"top{idx}: [{format_timestamp(entry.start_time)} - "
                f"{format_timestamp(entry.end_time)}]\n{entry.text}"
            )

        return {
            "retrieved_content": retrieved_content,
            "all_candidates": result.get("all_candidates", []),
            "entries": entries,
        }

    def cleanup(self) -> None:
        """Clean up memory."""
        self.retriever.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _find_neighbor_clips(
        self, center_entry: CaptionEntry, num_before: int = 2, num_after: int = 2
    ) -> List[CaptionEntry]:
        """Find neighbor clips around the center entry based on temporal order."""
        center_idx = None
        for i, entry in enumerate(self._sorted_entries):
            if entry.id == center_entry.id:
                center_idx = i
                break

        if center_idx is None:
            return [center_entry]

        neighbors = []

        for i in range(num_before, 0, -1):
            idx = center_idx - i
            if idx >= 0:
                neighbors.append(self._sorted_entries[idx])

        neighbors.append(center_entry)

        for i in range(1, num_after + 1):
            idx = center_idx + i
            if idx < len(self._sorted_entries):
                neighbors.append(self._sorted_entries[idx])

        return neighbors

    def _filter_neighbor_captions_batch(
        self,
        entries: List[CaptionEntry],
        query: str,
        choices: Dict[str, str],
        question_id: Optional[str] = None,
    ) -> List[str]:
        """
        Filter neighbor captions for all entries in ONE batch call.

        Collects 5 neighbor captions per clip, sends all at once,
        parses per-clip relevant_info from the response.

        Returns:
            List of filtered info strings (one per entry, empty if no relevant info)
        """
        num_clips = len(entries)

        # Collect neighbor captions for all clips
        all_neighbor_captions: List[List[str]] = []

        for entry in entries:
            neighbors = self._find_neighbor_clips(entry, num_before=2, num_after=2)

            neighbor_captions = []
            for neighbor in neighbors:
                caption_text = (
                    f"[{format_timestamp(neighbor.start_time)} - "
                    f"{format_timestamp(neighbor.end_time)}] {neighbor.text}"
                )
                neighbor_captions.append(caption_text)

            all_neighbor_captions.append(neighbor_captions)

        # Build text-only prompt (one batch call)
        filter_prompt = get_neighbor_filter_prompt()
        user_content = format_neighbor_filter_content(
            query=query,
            choices=choices,
            all_neighbor_captions=all_neighbor_captions,
        )
        filter_prompt.append({"role": "user", "content": user_content})

        # Call respond_model once
        try:
            response = self.respond_llm_model.generate(filter_prompt)

            # Parse JSON response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            # Parse per-clip relevant_info
            result = []
            for clip_idx in range(1, num_clips + 1):
                info = data.get(f"clip_{clip_idx}", "")

                if not info or info.lower().strip() in [
                    "no relevant information found",
                    "no relevant information",
                    "none",
                    "n/a",
                    "",
                ]:
                    result.append("")
                else:
                    result.append(info)

            # Debug logging
            should_log = self._qa_count <= self.debug_log_count
            if should_log:
                logger.info(f"\n{'~'*60}")
                logger.info(f"FILTER DEBUG - QA #{self._qa_count} | ID: {question_id}")
                logger.info(f"{'~'*60}")
                logger.info(
                    f"[INPUT] {num_clips} clips x 5 neighbors = "
                    f"{sum(len(c) for c in all_neighbor_captions)} captions"
                )
                response_preview = response[:500] + "..." if len(response) > 500 else response
                logger.info(f"[RAW RESPONSE]: {response_preview}")
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
        """Build ClipContext list with caption, filtered info, and frames."""
        contexts = []

        for rank, (entry, filtered_info) in enumerate(zip(entries, filtered_infos)):
            original_caption = (
                f"top{rank+1}: [{format_timestamp(entry.start_time)} - "
                f"{format_timestamp(entry.end_time)}]\n{entry.text}"
            )

            if round_num == 1:
                frames = self._load_frames_for_entry(entry, num_frames=6)
            elif round_num == 2:
                frames = self._load_frames_for_entry(entry, num_frames=6) if rank < 5 else []
            else:
                frames = self._load_frames_for_entry(entry, num_frames=1)

            contexts.append(ClipContext(
                original_caption=original_caption,
                filtered_info=filtered_info,
                frames=frames,
                rank=rank + 1,
            ))

        return contexts

    def _format_clip_contexts_as_text(self, contexts: List[ClipContext]) -> List[str]:
        """Format ClipContext list as text-only. Include [Relevant Info] only if non-empty."""
        result = []
        for ctx in contexts:
            if ctx.filtered_info:
                text = f"{ctx.original_caption}\n\n[Relevant Info]\n{ctx.filtered_info}"
            else:
                text = ctx.original_caption
            result.append(text)
        return result

    def _format_round_history_text_only(self, rounds: List[Dict[str, Any]]) -> str:
        """Format round history for reasoning prompt (text only, with relevant info)."""
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

    def _build_qa_content(
        self,
        full_query: str,
        retrieved_items: List[RetrievedItem],
        choices: Optional[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Build final QA content (interleaved). Include [Relevant Info] only if non-empty."""
        qa_content = [{"type": "text", "text": full_query + "\n\nContext:\n"}]

        for item in retrieved_items:
            qa_content.append({"type": "text", "text": f"## Round {item.round_num} Results:\n"})
            for ctx in item.content:
                qa_content.append({"type": "text", "text": f"{ctx.original_caption}\n"})
                if ctx.filtered_info:
                    qa_content.append({
                        "type": "text",
                        "text": f"\n[Relevant Info]\n{ctx.filtered_info}\n",
                    })
                for img in ctx.frames:
                    qa_content.append({"type": "image", "image": img})

        if choices:
            qa_content.append({
                "type": "text",
                "text": "\nPlease provide only the final answer from the choices given (e.g., A, B, C, or D).",
            })

        return qa_content

    def answer(
        self,
        query: str,
        choices: Optional[Dict[str, str]] = None,
        question_id: Optional[str] = None,
    ) -> QAResult:
        """
        Answer one question.
        """
        self._qa_count += 1
        should_log = self._qa_count <= self.debug_log_count

        self.index()

        full_query = f"Query: {query}"
        if choices:
            choices_str = " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))
            full_query += f"\nChoices: {choices_str}"

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

            # Build user content
            history_str = self._format_round_history_text_only(round_history)
            user_content = f"""{full_query}

Round History:
{history_str}

Task:
Step 1: Decide whether to "search" or "answer".
Step 2 (only if search): Form a keyword(phrase)-style search query."""

            reasoning_messages = copy.deepcopy(reasoning_prompt)
            reasoning_messages.append({"role": "user", "content": user_content})

            try:
                response = self.respond_llm_model.generate(reasoning_messages)
                reasoning_output = self._parse_reasoning_response(response)
                logger.info(
                    f"ID {question_id}: Round {round_num} Decision: {reasoning_output.decision}"
                )
            except Exception as e:
                logger.error(f"Reasoning failed: {e}")
                err_count += 1
                continue

            raw_responses.append({
                "type": "reasoning",
                "round_num": round_num,
                "raw_response": response,
                "parsed_decision": reasoning_output.decision,
            })

            if reasoning_output.decision == "answer":
                break

            search_query = reasoning_output.search_query
            if not search_query:
                logger.warning(
                    f"ID {question_id}: Round {round_num} Search decision but no query."
                )
                err_count += 1
                continue

            logger.info(f"ID {question_id}: Round {round_num} Query: {search_query}")

            # Retrieve from episodic memory
            retrieved_data = self._retrieve(
                search_query,
                top_k=self.top_k,
                retrieved_set=retrieved_set,
            )

            entries = retrieved_data.get("entries", [])

            if not entries:
                logger.warning(f"ID {question_id}: Round {round_num} No entries retrieved")
                round_history.append({
                    "round_num": round_num,
                    "decision": "search",
                    "search_query": search_query,
                    "retrieved_content": [],
                    "all_candidates": [],
                })
                continue

            # Batch filter neighbor captions (ONE call)
            logger.info(
                f"ID {question_id}: Round {round_num} Batch filtering neighbor captions..."
            )
            filtered_infos = self._filter_neighbor_captions_batch(
                entries, query, choices or {}, question_id
            )

            # Build clip contexts with frames
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
                RetrievedItem(
                    content=clip_contexts,
                    query=search_query,
                    round_num=round_num,
                )
            )

            # Format for text-only history (includes relevant info)
            clip_texts = self._format_clip_contexts_as_text(clip_contexts)

            round_history.append({
                "round_num": round_num,
                "decision": "search",
                "search_query": search_query,
                "retrieved_content": clip_texts,
                "filtered_infos": filtered_infos,
                "all_candidates": retrieved_data.get("all_candidates", []),
            })

        # Final QA
        qa_prompt = self.prompt_template_manager.render("final_qa_video_benchmark")
        qa_content = self._build_qa_content(full_query, retrieved_items, choices)

        qa_messages = copy.deepcopy(qa_prompt)
        qa_messages.append({"role": "user", "content": qa_content})

        try:
            answer = self.respond_llm_model.generate(qa_messages)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = "Error"

        raw_responses.append({"type": "qa", "raw_response": answer})

        return QAResult(
            question=query,
            answer=answer,
            retrieved_items=retrieved_items,
            round_history=round_history,
            num_rounds=round_num,
            raw_responses=raw_responses,
        )
