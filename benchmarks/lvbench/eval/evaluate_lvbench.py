#!/usr/bin/env python3
"""MERIT evaluation on LVBench."""

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Run straight from a checkout, without installing.
MERIT_ROOT = PROJECT_ROOT.parent.parent
MERIT_SRC = MERIT_ROOT / "src"
if (MERIT_SRC / "merit").is_dir():
    sys.path.insert(0, str(MERIT_SRC))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# LVBench QA data loading
# ---------------------------------------------------------------------------

def load_lvbench_qa(meta_path: str) -> List[Dict[str, Any]]:
    """
    Load LVBench QA data from video_info.meta.jsonl.

    Each line is a video entry with 'key', 'type', 'qa' array.
    We flatten into per-question items with parsed options.
    """
    qa_items = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            video_id = entry["key"]
            video_type = entry.get("type", "")

            for qa in entry.get("qa", []):
                question_text, choices = parse_question_with_options(qa["question"])
                qa_items.append({
                    "video_id": video_id,
                    "video_type": video_type,
                    "uid": str(qa["uid"]),
                    "question": question_text,
                    "choices": choices,  # {"A": "...", "B": "...", ...}
                    "answer": qa["answer"],
                    "question_type": qa.get("question_type", []),
                    "time_reference": qa.get("time_reference", ""),
                })
    return qa_items


def parse_question_with_options(raw_question: str) -> Tuple[str, Dict[str, str]]:
    """LVBench embeds the options in the question text."""
    pattern = re.compile(r'\n\(([A-D])\)\s*')
    parts = pattern.split(raw_question)

    question_text = parts[0].strip()
    choices = {}
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            choices[parts[i]] = parts[i + 1].strip()
    return question_text, choices


def group_qa_by_video(qa_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {}
    for item in qa_data:
        video_id = item["video_id"]
        if video_id not in grouped:
            grouped[video_id] = []
        grouped[video_id].append(item)
    return grouped


# ---------------------------------------------------------------------------
# Answer parsing & evaluation
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    return text.lower().strip().rstrip(".,)")


def extract_choice_letter(text: str) -> Optional[str]:
    match = re.match(r"\(?([A-Da-d])[\.\)]?\s*", text.strip())
    return match.group(1).upper() if match else None


def parse_reasoning_answer(response: str) -> Tuple[str, str]:
    answer = ""
    reasoning = ""
    answer_match = re.search(r"ANSWER:\s*([A-D])", response, re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).upper()
    reasoning_match = re.search(
        r"REASONING:\s*(.+?)(?=\n\n|$)", response, re.IGNORECASE | re.DOTALL
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    return answer, reasoning


def evaluate_prediction(prediction: str, gold_answer: str) -> bool:
    pred_letter = extract_choice_letter(prediction)
    if pred_letter:
        return pred_letter == gold_answer.upper()
    return normalize(prediction) == normalize(gold_answer)


# ---------------------------------------------------------------------------
# Thread-safe embedding model wrapper
# ---------------------------------------------------------------------------

class ThreadSafeEmbeddingModel:
    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()

    def encode_text(self, *args, **kwargs):
        with self.lock:
            return self.model.encode_text(*args, **kwargs)

    def encode_image(self, *args, **kwargs):
        with self.lock:
            return self.model.encode_image(*args, **kwargs)

    def encode_video(self, *args, **kwargs):
        with self.lock:
            return self.model.encode_video(*args, **kwargs)

    def encode(self, *args, **kwargs):
        with self.lock:
            return self.model.encode(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        with self.lock:
            return self.model.load_model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


# ---------------------------------------------------------------------------
# Worker function (multiprocessing for local models)
# ---------------------------------------------------------------------------

def process_video_batch(
    worker_id: int,
    video_batch: List[Tuple[str, List[Dict[str, Any]]]],
    existing_results: List[Dict[str, Any]],
    args: argparse.Namespace,
    result_queue: Queue,
    worker_log_file: Optional[str] = None,
    results_dir: Optional[str] = None,
    qa_progress_file: Optional[str] = None,
    completed_uids: Optional[Dict[str, dict]] = None,
) -> None:
    """Worker function: process a batch of videos."""
    if worker_log_file:
        worker_logger = logging.getLogger()
        worker_logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(worker_log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        worker_logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        worker_logger.addHandler(console_handler)
        worker_logger.setLevel(logging.INFO)

    # GPU setup
    from merit.eval_utils import setup_worker_gpus
    device, llm_device_map = setup_worker_gpus(
        worker_id,
        getattr(args, "gpu_ids", None) or None,
        getattr(args, "gpus_per_worker", 1),
    )

    import time
    time.sleep(worker_id * 10)

    # Monkey-patch IMAGE_MAX_TOKEN_NUM for Qwen3VL
    if args.image_max_tokens is not None:
        try:
            import qwen_vl_utils.vision_process as vp
            vp.IMAGE_MAX_TOKEN_NUM = args.image_max_tokens
            logger.info(f"Worker {worker_id}: Set IMAGE_MAX_TOKEN_NUM = {args.image_max_tokens}")
        except ImportError:
            logger.info(f"Worker {worker_id}: qwen_vl_utils not available, skipping override")

    from merit.embedding import EmbeddingModel
    from merit.llm import LLMModel
    from merit_lvbench.memory import LVBenchMemory

    embedding_model = EmbeddingModel(device=device)
    ts_embedding_model = ThreadSafeEmbeddingModel(embedding_model)

    llm_kwargs = dict(model_name=args.respond_model, device_map=llm_device_map)
    if args.image_max_size is not None:
        llm_kwargs["max_size"] = (args.image_max_size, args.image_max_size)
    if getattr(args, "service_tier", None):
        llm_kwargs["service_tier"] = args.service_tier
    if getattr(args, "timeout", None):
        llm_kwargs["timeout"] = args.timeout

    respond_llm = LLMModel(**llm_kwargs)

    logger.info(f"Worker {worker_id}: Force loading text embedding model...")
    ts_embedding_model.load_model("text")
    logger.info(f"Worker {worker_id}: Text embedding model loaded.")

    completed_video_ids = {r["video_id"] for r in existing_results}
    worker_results = list(existing_results)

    for video_id, questions in video_batch:
        if video_id in completed_video_ids:
            logger.info(f"Worker {worker_id}: Skipping completed video {video_id}")
            continue

        logger.info(f"Worker {worker_id}: Processing video {video_id} ({len(questions)} questions)")

        memory = LVBenchMemory(
            embedding_model=ts_embedding_model,
            respond_llm_model=respond_llm,
            cache_dir=".cache/multi_keys",
            frame_dir=args.frame_dir,
            max_rounds=args.max_rounds,
            debug_log_count=args.debug_log_count,
        )
        memory.set_top_k(args.top_k)

        caption_path = os.path.join(args.caption_dir, f"{video_id}.json")
        try:
            memory.load_video_data(video_id=video_id, caption_path=caption_path)
        except Exception as e:
            logger.error(f"Worker {worker_id}: Failed to load data for video {video_id}: {e}")
            continue

        memory.index()

        video_result = {
            "video_id": video_id,
            "video_type": questions[0].get("video_type", ""),
            "questions": [],
        }

        for q_item in questions:
            uid = q_item["uid"]
            question = q_item["question"]
            choices = q_item["choices"]
            gold_answer = q_item["answer"]

            if completed_uids and uid in completed_uids:
                prev = completed_uids[uid]
                question_result = {
                    "uid": uid,
                    "question_type": q_item.get("question_type", []),
                    "question": question,
                    "choices": choices,
                    "answer": gold_answer,
                    "response": prev.get("response", prev.get("parsed_answer", "")),
                    "parsed_answer": prev.get("parsed_answer", ""),
                    "reasoning": prev.get("reasoning", ""),
                    "correct": prev.get("correct", False),
                    "round_history": prev.get("round_history", []),
                    "raw_responses": prev.get("raw_responses", []),
                    "resumed": True,
                }
                video_result["questions"].append(question_result)
                logger.info(
                    f"Worker {worker_id}: [{video_id}] Q{uid}: "
                    f"(resumed) {prev.get('parsed_answer', '')} | Gold: {gold_answer} | "
                    f"Correct: {prev.get('correct', False)}"
                )
                continue

            try:
                qa_result = memory.answer(
                    query=question, choices=choices, question_id=uid,
                )
                response = qa_result.answer
                parsed_answer, reasoning = parse_reasoning_answer(response)
            except Exception as e:
                logger.error(f"Worker {worker_id}: Error answering question {uid}: {e}")
                response = "Error"
                parsed_answer = ""
                reasoning = ""
                qa_result = None

            eval_text = parsed_answer if parsed_answer else response
            is_correct = evaluate_prediction(eval_text, gold_answer)

            question_result = {
                "uid": uid,
                "question_type": q_item.get("question_type", []),
                "question": question,
                "choices": choices,
                "answer": gold_answer,
                "response": response,
                "parsed_answer": parsed_answer,
                "reasoning": reasoning,
                "correct": is_correct,
                "round_history": qa_result.round_history if qa_result else [],
                "raw_responses": qa_result.raw_responses if qa_result else [],
            }
            video_result["questions"].append(question_result)

            logger.info(
                f"Worker {worker_id}: [{video_id}] Q{uid}: "
                f"{parsed_answer or response[:30]} | Gold: {gold_answer} | "
                f"Correct: {is_correct}"
            )

            if qa_progress_file:
                with open(qa_progress_file, "a", encoding="utf-8") as f:
                    entry = {
                        "video_id": video_id,
                        "uid": uid,
                        "question": question[:100],
                        "gold_answer": gold_answer,
                        "parsed_answer": parsed_answer,
                        "correct": is_correct,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if results_dir:
                save_json(video_result, os.path.join(results_dir, f"{video_id}.json"))

        memory.cleanup()

        video_correct = sum(1 for q in video_result["questions"] if q.get("correct"))
        video_total = len(video_result["questions"])
        logger.info(f"Worker {worker_id}: [{video_id}] Completed: {video_correct}/{video_total} correct")

        worker_results.append(video_result)

    result_queue.put((worker_id, worker_results))


# ---------------------------------------------------------------------------
# API model worker (ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def process_video_api(
    video_id: str,
    questions: List[Dict[str, Any]],
    args: argparse.Namespace,
    embedding_model,
    respond_llm,
    embedding_lock: threading.Lock,
    qa_progress_file: Optional[str] = None,
    progress_lock: Optional[threading.Lock] = None,
    completed_uids: Optional[Dict[str, dict]] = None,
    results_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Process a single video using API model."""
    from merit_lvbench.memory import LVBenchMemory

    memory = LVBenchMemory(
        embedding_model=embedding_model,
        respond_llm_model=respond_llm,
        cache_dir=".cache/multi_keys",
        frame_dir=args.frame_dir,
        max_rounds=args.max_rounds,
        debug_log_count=args.debug_log_count,
    )
    memory.set_top_k(args.top_k)

    caption_path = os.path.join(args.caption_dir, f"{video_id}.json")
    try:
        memory.load_video_data(video_id=video_id, caption_path=caption_path)
    except Exception as e:
        logger.error(f"Failed to load data for video {video_id}: {e}")
        return None

    with embedding_lock:
        memory.index()

    video_result = {
        "video_id": video_id,
        "video_type": questions[0].get("video_type", ""),
        "questions": [],
    }

    for q_item in questions:
        uid = q_item["uid"]
        question = q_item["question"]
        choices = q_item["choices"]
        gold_answer = q_item["answer"]

        if completed_uids and uid in completed_uids:
            prev = completed_uids[uid]
            question_result = {
                "uid": uid,
                "question_type": q_item.get("question_type", []),
                "question": question,
                "choices": choices,
                "answer": gold_answer,
                "response": prev.get("response", prev.get("parsed_answer", "")),
                "parsed_answer": prev.get("parsed_answer", ""),
                "reasoning": prev.get("reasoning", ""),
                "correct": prev.get("correct", False),
                "round_history": prev.get("round_history", []),
                "raw_responses": prev.get("raw_responses", []),
                "resumed": True,
            }
            video_result["questions"].append(question_result)
            logger.info(
                f"[{video_id}] Q{uid}: "
                f"(resumed) {prev.get('parsed_answer', '')} | Gold: {gold_answer} | "
                f"Correct: {prev.get('correct', False)}"
            )
            if results_dir:
                save_json(video_result, os.path.join(results_dir, f"{video_id}.json"))
            continue

        try:
            qa_result = memory.answer(query=question, choices=choices, question_id=uid)
            response = qa_result.answer
            parsed_answer, reasoning = parse_reasoning_answer(response)
        except Exception as e:
            logger.error(f"Error answering question {uid}: {e}")
            response = "Error"
            parsed_answer = ""
            reasoning = ""
            qa_result = None

        eval_text = parsed_answer if parsed_answer else response
        is_correct = evaluate_prediction(eval_text, gold_answer)

        question_result = {
            "uid": uid,
            "question_type": q_item.get("question_type", []),
            "question": question,
            "choices": choices,
            "answer": gold_answer,
            "response": response,
            "parsed_answer": parsed_answer,
            "reasoning": reasoning,
            "correct": is_correct,
            "round_history": qa_result.round_history if qa_result else [],
            "raw_responses": qa_result.raw_responses if qa_result else [],
        }
        video_result["questions"].append(question_result)

        logger.info(
            f"[{video_id}] Q{uid}: "
            f"{parsed_answer or response[:30]} | Gold: {gold_answer} | Correct: {is_correct}"
        )

        if qa_progress_file:
            entry = json.dumps({
                "video_id": video_id, "uid": uid,
                "question": question[:100], "gold_answer": gold_answer,
                "parsed_answer": parsed_answer, "correct": is_correct,
            }, ensure_ascii=False) + "\n"
            if progress_lock:
                with progress_lock:
                    with open(qa_progress_file, "a", encoding="utf-8") as f:
                        f.write(entry)
            else:
                with open(qa_progress_file, "a", encoding="utf-8") as f:
                    f.write(entry)

        if results_dir:
            save_json(video_result, os.path.join(results_dir, f"{video_id}.json"))

    memory.cleanup()
    video_correct = sum(1 for q in video_result["questions"] if q.get("correct"))
    video_total = len(video_result["questions"])
    logger.info(f"[{video_id}] Completed: {video_correct}/{video_total} correct")

    return video_result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_json(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def is_api_model(model_name: str) -> bool:
    return "gpt" in model_name.lower()


def compute_lvbench_accuracy(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-question-type and overall accuracy for LVBench."""
    category_right = defaultdict(int)
    category_total = defaultdict(int)
    total_correct = 0
    total_questions = 0

    for vr in all_results:
        for q in vr["questions"]:
            is_correct = q.get("correct", False)
            total_questions += 1
            if is_correct:
                total_correct += 1
            for qtype in q.get("question_type", []):
                category_total[qtype] += 1
                if is_correct:
                    category_right[qtype] += 1

    overall_acc = total_correct / total_questions if total_questions > 0 else 0

    # LVBench standard category mapping
    name_map = {
        "key information retrieval": "KIR",
        "event understanding": "EU",
        "summarization": "Sum",
        "entity recognition": "ER",
        "reasoning": "Rea",
        "temporal grounding": "TG",
    }

    category_acc = {}
    for cat, total in sorted(category_total.items()):
        acc = category_right[cat] / total if total > 0 else 0
        short_name = name_map.get(cat, cat)
        category_acc[short_name] = {
            "accuracy": round(acc, 4),
            "correct": category_right[cat],
            "total": total,
        }

    return {
        "overall": {"accuracy": round(overall_acc, 4), "correct": total_correct, "total": total_questions},
        "per_category": category_acc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="MERIT evaluation on LVBench"
    )
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument(
        "--meta-file", type=str,
        default=str(PROJECT_ROOT / "data" / "video_info.meta.jsonl"),
    )
    parser.add_argument(
        "--caption-dir", type=str,
        default=str(PROJECT_ROOT / "data" / "keys"),
    )
    parser.add_argument(
        "--frame-dir", type=str,
        default=str(PROJECT_ROOT / "data" / "frames"),
    )
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "output" / "eval"))
    parser.add_argument("--log-dir", type=str, default=str(PROJECT_ROOT / "log"))
    parser.add_argument("--respond-model", type=str, default="qwen3vl-8b")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    parser.add_argument("--gpus-per-worker", type=int, default=1)
    parser.add_argument("--sample-file", type=str, default=None)
    parser.add_argument("--video-ids", type=str, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug-log-count", type=int, default=5)
    # High-resolution image settings
    parser.add_argument("--image-max-tokens", type=int, default=512)
    parser.add_argument("--image-max-size", type=int, default=2048)
    # API model options
    parser.add_argument("--service-tier", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=None)

    args = parser.parse_args()

    if args.gpu_ids:
        args.gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]

    if not is_api_model(args.respond_model):
        from merit.eval_utils import validate_gpu_config
        validate_gpu_config(
            args.num_workers, args.gpu_ids or None, getattr(args, "gpus_per_worker", 1)
        )

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, args.exp_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    log_file = os.path.join(log_dir, f"main_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_format)
    root_logger.addHandler(file_handler)

    logger.info(f"Arguments: {args}")
    logger.info(f"Logging to {log_dir}")

    # Output
    exp_output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    output_file = os.path.join(exp_output_dir, f"{args.exp_name}.json")
    qa_progress_file = os.path.join(exp_output_dir, f"{args.exp_name}_qa_progress.jsonl")

    # Load QA data
    logger.info(f"Loading QA data from {args.meta_file}")
    qa_data = load_lvbench_qa(args.meta_file)
    logger.info(f"Loaded {len(qa_data)} QA items from {len(set(q['video_id'] for q in qa_data))} videos")

    grouped_qa = group_qa_by_video(qa_data)

    # Apply filters
    if args.sample_file:
        with open(args.sample_file, "r") as f:
            content = f.read()
        sample_ids = {s.strip() for s in content.replace(",", " ").split() if s.strip()}
        grouped_qa = {k: v for k, v in grouped_qa.items() if k in sample_ids}
        logger.info(f"Filtered to {len(grouped_qa)} videos from sample file")

    if args.video_ids:
        target_ids = set(args.video_ids.split(","))
        grouped_qa = {k: v for k, v in grouped_qa.items() if k in target_ids}
        logger.info(f"Filtered to {len(grouped_qa)} target videos")

    if args.max_videos:
        video_ids = list(grouped_qa.keys())[:args.max_videos]
        grouped_qa = {k: grouped_qa[k] for k in video_ids}
        logger.info(f"Limited to {len(grouped_qa)} videos")

    # Check caption availability
    available_videos = {}
    for video_id, questions in grouped_qa.items():
        caption_path = os.path.join(args.caption_dir, f"{video_id}.json")
        if os.path.exists(caption_path):
            available_videos[video_id] = questions
        else:
            logger.warning(f"Caption file not found for video {video_id}, skipping")

    logger.info(f"Processing {len(available_videos)} videos with available captions")

    if not available_videos:
        logger.error("No videos to process")
        return

    # Resume
    results_dir = os.path.join(exp_output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    completed_video_ids = set()
    completed_uids: Dict[str, dict] = {}

    if args.resume:
        # Load from per-video result files
        for result_file in sorted(Path(results_dir).glob("*.json")):
            try:
                vr = load_json(str(result_file))
                vid = vr["video_id"]
                video_questions = vr.get("questions", [])
                for q in video_questions:
                    completed_uids[str(q["uid"])] = q
                # Mark video as fully complete if all questions are done
                if vid in available_videos and len(video_questions) >= len(available_videos[vid]):
                    completed_video_ids.add(vid)
            except Exception as e:
                logger.warning(f"Failed to load result file {result_file}: {e}")
        logger.info(
            f"Resumed {len(completed_uids)} completed questions "
            f"from {len(list(Path(results_dir).glob('*.json')))} video files "
            f"({len(completed_video_ids)} fully completed videos)"
        )

    videos_to_process = [
        (vid, qs) for vid, qs in available_videos.items()
        if vid not in completed_video_ids
    ]
    logger.info(f"Videos remaining to process: {len(videos_to_process)}")

    if not videos_to_process:
        logger.info("All videos already processed, generating final output")
    elif is_api_model(args.respond_model):
        # API model path
        from merit.embedding import EmbeddingModel
        from merit.llm import LLMModel

        logger.info("API model detected, using ThreadPoolExecutor")

        embedding_model = EmbeddingModel()
        ts_embedding_model = ThreadSafeEmbeddingModel(embedding_model)
        ts_embedding_model.load_model("text")
        embedding_lock = threading.Lock()

        llm_kwargs = dict(model_name=args.respond_model)
        if args.image_max_size is not None:
            llm_kwargs["max_size"] = (args.image_max_size, args.image_max_size)
        if args.service_tier:
            llm_kwargs["service_tier"] = args.service_tier
        if args.timeout:
            llm_kwargs["timeout"] = args.timeout
        respond_llm = LLMModel(**llm_kwargs)

        concurrency = args.concurrency or args.num_workers
        progress_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            for video_id, questions in videos_to_process:
                future = executor.submit(
                    process_video_api,
                    video_id=video_id, questions=questions, args=args,
                    embedding_model=ts_embedding_model, respond_llm=respond_llm,
                    embedding_lock=embedding_lock,
                    qa_progress_file=qa_progress_file,
                    progress_lock=progress_lock,
                    completed_uids=completed_uids,
                    results_dir=results_dir,
                )
                futures[future] = video_id

            completed_count = 0
            total_to_process = len(futures)
            for future in as_completed(futures):
                video_id = futures[future]
                completed_count += 1
                try:
                    future.result()
                    logger.info(f"Progress: {completed_count}/{total_to_process} videos done")
                except Exception as e:
                    logger.error(f"Video {video_id} failed: {e}")
    else:
        # Local model: multiprocessing
        num_workers = min(args.num_workers, len(videos_to_process))
        batches: List[List[Tuple[str, List[Dict[str, Any]]]]] = [[] for _ in range(num_workers)]
        for i, video_item in enumerate(videos_to_process):
            batches[i % num_workers].append(video_item)

        logger.info(f"Starting {num_workers} workers, batch sizes: {[len(b) for b in batches]}")

        result_queue = Queue()
        processes = []

        for i, batch in enumerate(batches):
            worker_log = os.path.join(log_dir, f"worker_{i}.log")

            p = Process(
                target=process_video_batch,
                args=(i, batch, [], args, result_queue,
                      worker_log, results_dir, qa_progress_file, completed_uids),
            )
            p.start()
            processes.append(p)

        all_results = []
        for _ in range(num_workers):
            _, worker_results = result_queue.get()
            all_results.extend(worker_results)

        for p in processes:
            p.join()

    # Merge all per-video result files for final output
    all_results = []
    for result_file in sorted(Path(results_dir).glob("*.json")):
        try:
            all_results.append(load_json(str(result_file)))
        except Exception as e:
            logger.warning(f"Failed to load result file {result_file}: {e}")

    logger.info(f"Merged {len(all_results)} video results from {results_dir}")

    # Calculate accuracy
    accuracy_stats = compute_lvbench_accuracy(all_results)
    logger.info(f"Final Accuracy: {accuracy_stats['overall']['accuracy']:.4f} "
                f"({accuracy_stats['overall']['correct']}/{accuracy_stats['overall']['total']})")
    for cat, stats in accuracy_stats["per_category"].items():
        logger.info(f"  {cat}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

    # Save final output
    final_output = {
        "metadata": {
            "exp_name": args.exp_name,
            "respond_model": args.respond_model,
            "max_rounds": args.max_rounds,
            "top_k": args.top_k,
            "image_max_tokens": args.image_max_tokens,
            "image_max_size": args.image_max_size,
            "frame_dir": args.frame_dir,
            "num_workers": args.num_workers,
            "total_videos": len(all_results),
            "key_types": "3-key (event, object, summary)",
        },
        "accuracy": accuracy_stats,
        "results": all_results,
    }

    save_json(final_output, output_file)
    logger.info(f"Results saved to {output_file}")

    # LVBench leaderboard format (result.json compatible)
    leaderboard_result = {cat: stats["accuracy"] for cat, stats in accuracy_stats["per_category"].items()}
    leaderboard_result["Overall"] = accuracy_stats["overall"]["accuracy"]
    leaderboard_file = os.path.join(exp_output_dir, f"{args.exp_name}_leaderboard.json")
    save_json(leaderboard_result, leaderboard_file)
    logger.info(f"Leaderboard format saved to {leaderboard_file}")


if __name__ == "__main__":
    main()
