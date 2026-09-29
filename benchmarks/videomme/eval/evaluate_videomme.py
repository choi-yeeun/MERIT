#!/usr/bin/env python3
"""MERIT evaluation on the Video-MME long split."""

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import threading
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


def load_json(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


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


def parse_options_to_choices(options: List[str]) -> Dict[str, str]:
    choices = {}
    for option in options:
        match = re.match(r"([A-D])\.\s*(.+)", option, re.DOTALL)
        if match:
            choices[match.group(1)] = match.group(2).strip()
    return choices


def group_qa_by_video(qa_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {}
    for item in qa_data:
        video_id = item["videoID"]
        if video_id not in grouped:
            grouped[video_id] = []
        grouped[video_id].append(item)
    return grouped


class ThreadSafeEmbeddingModel:
    """Thread-safe wrapper for EmbeddingModel."""
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


def process_video_batch(
    worker_id: int,
    video_batch: List[Tuple[str, List[Dict[str, Any]]]],
    existing_results: List[Dict[str, Any]],
    args: argparse.Namespace,
    result_queue: Queue,
    worker_log_file: Optional[str] = None,
    worker_temp_file: Optional[str] = None,
    qa_progress_file: Optional[str] = None,
) -> None:
    """Worker function: process a batch of videos."""
    # Setup logging for worker
    if worker_log_file:
        worker_logger = logging.getLogger()
        worker_logger.handlers.clear()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
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

    # Monkey-patch IMAGE_MAX_TOKEN_NUM for Qwen3VL's smart_resize
    if args.image_max_tokens is not None:
        try:
            import qwen_vl_utils.vision_process as vp
            vp.IMAGE_MAX_TOKEN_NUM = args.image_max_tokens
            logger.info(
                f"Worker {worker_id}: Set IMAGE_MAX_TOKEN_NUM = {args.image_max_tokens}"
            )
        except ImportError:
            logger.info(
                f"Worker {worker_id}: qwen_vl_utils not available, "
                "skipping IMAGE_MAX_TOKEN_NUM override"
            )

    # Initialize models
    from merit.embedding import EmbeddingModel
    from merit.llm import LLMModel
    from merit_videomme.memory import VideoMMEMemory

    embedding_model = EmbeddingModel(device=device)
    ts_embedding_model = ThreadSafeEmbeddingModel(embedding_model)

    llm_kwargs = dict(
        model_name=args.respond_model,
        device_map=llm_device_map,
    )
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

    # Track completed video_ids from existing results
    completed_video_ids = {r["video_id"] for r in existing_results}
    worker_results = list(existing_results)

    for video_id, questions in video_batch:
        if video_id in completed_video_ids:
            logger.info(f"Worker {worker_id}: Skipping completed video {video_id}")
            continue

        logger.info(
            f"Worker {worker_id}: Processing video {video_id} "
            f"({len(questions)} questions)"
        )

        # Create memory instance per video
        memory = VideoMMEMemory(
            embedding_model=ts_embedding_model,
            respond_llm_model=respond_llm,
            cache_dir=".cache/multi_keys",
            frame_dir=args.frame_dir,
            max_rounds=args.max_rounds,
            debug_log_count=args.debug_log_count,
        )
        memory.set_top_k(args.top_k)

        # Load video data
        caption_path = os.path.join(args.caption_dir, f"{video_id}.json")
        try:
            memory.load_video_data(
                video_id=video_id,
                caption_path=caption_path,
            )
        except Exception as e:
            logger.error(f"Worker {worker_id}: Failed to load data for video {video_id}: {e}")
            continue

        # Index
        memory.index()

        # Get video metadata from first question
        first_q = questions[0]
        video_result = {
            "video_id": first_q["video_id"],
            "videoID": first_q["videoID"],
            "duration": first_q["duration"],
            "domain": first_q["domain"],
            "sub_category": first_q["sub_category"],
            "questions": [],
        }

        # Process each question
        for q_item in questions:
            question_id = q_item["question_id"]
            question = q_item["question"]
            options = q_item["options"]
            gold_answer = q_item["answer"]

            choices = parse_options_to_choices(options)

            try:
                qa_result = memory.answer(
                    query=question,
                    choices=choices,
                    question_id=question_id,
                )
                response = qa_result.answer
                parsed_answer, reasoning = parse_reasoning_answer(response)
            except Exception as e:
                logger.error(
                    f"Worker {worker_id}: Error answering question {question_id}: {e}"
                )
                response = "Error"
                parsed_answer = ""
                reasoning = ""
                qa_result = None

            eval_text = parsed_answer if parsed_answer else response
            is_correct = evaluate_prediction(eval_text, gold_answer)

            question_result = {
                "question_id": question_id,
                "task_type": q_item.get("task_type", ""),
                "question": question,
                "options": options,
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
                f"Worker {worker_id}: [{video_id}] Q{question_id}: "
                f"{parsed_answer or response[:30]} | Gold: {gold_answer} | "
                f"Correct: {is_correct}"
            )

        # Cleanup per-video memory
        memory.cleanup()

        video_correct = sum(1 for q in video_result["questions"] if q.get("correct"))
        video_total = len(video_result["questions"])
        logger.info(
            f"Worker {worker_id}: [{video_id}] Completed: "
            f"{video_correct}/{video_total} correct"
        )

        worker_results.append(video_result)

        # Write per-question progress (append, cross-process safe for small writes)
        if qa_progress_file:
            with open(qa_progress_file, "a", encoding="utf-8") as f:
                for q in video_result["questions"]:
                    entry = {
                        "video_id": video_result["video_id"],
                        "question_id": q["question_id"],
                        "question": q["question"][:100],
                        "gold_answer": q["answer"],
                        "parsed_answer": q["parsed_answer"],
                        "correct": q["correct"],
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Save intermediate results
        if worker_temp_file:
            save_json(worker_results, worker_temp_file)

    result_queue.put((worker_id, worker_results))


def is_api_model(model_name: str) -> bool:
    """Check if the model is an API-based model (not local)."""
    name = model_name.lower()
    return "gpt" in name or "gemini" in name


def process_video_api(
    video_id: str,
    questions: List[Dict[str, Any]],
    args: argparse.Namespace,
    embedding_model,
    respond_llm,
    embedding_lock: threading.Lock,
) -> Optional[Dict[str, Any]]:
    """Process a single video using API model (ThreadPoolExecutor compatible)."""
    from merit_videomme.memory import VideoMMEMemory

    memory = VideoMMEMemory(
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
        memory.load_video_data(
            video_id=video_id,
            caption_path=caption_path,
        )
    except Exception as e:
        logger.error(f"Failed to load data for video {video_id}: {e}")
        return None

    with embedding_lock:
        memory.index()

    first_q = questions[0]
    video_result = {
        "video_id": first_q["video_id"],
        "videoID": first_q["videoID"],
        "duration": first_q["duration"],
        "domain": first_q["domain"],
        "sub_category": first_q["sub_category"],
        "questions": [],
    }

    for q_item in questions:
        question_id = q_item["question_id"]
        question = q_item["question"]
        options = q_item["options"]
        gold_answer = q_item["answer"]
        choices = parse_options_to_choices(options)

        try:
            qa_result = memory.answer(
                query=question,
                choices=choices,
                question_id=question_id,
            )
            response = qa_result.answer
            parsed_answer, reasoning = parse_reasoning_answer(response)
        except Exception as e:
            logger.error(f"Error answering question {question_id}: {e}")
            response = "Error"
            parsed_answer = ""
            reasoning = ""
            qa_result = None

        eval_text = parsed_answer if parsed_answer else response
        is_correct = evaluate_prediction(eval_text, gold_answer)

        question_result = {
            "question_id": question_id,
            "task_type": q_item.get("task_type", ""),
            "question": question,
            "options": options,
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
            f"[{video_id}] Q{question_id}: "
            f"{parsed_answer or response[:30]} | Gold: {gold_answer} | "
            f"Correct: {is_correct}"
        )

    memory.cleanup()

    video_correct = sum(1 for q in video_result["questions"] if q.get("correct"))
    video_total = len(video_result["questions"])
    logger.info(f"[{video_id}] Completed: {video_correct}/{video_total} correct")

    return video_result


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="MERIT evaluation on the Video-MME long split"
    )
    parser.add_argument(
        "--exp-name", type=str, required=True,
        help="Experiment name (used for output filenames and resume)",
    )
    parser.add_argument(
        "--qa-file", type=str,
        default=str(PROJECT_ROOT / "data" / "test_qa.json"),
    )
    parser.add_argument(
        "--caption-dir", type=str,
        default=str(PROJECT_ROOT / "data" / "keys"),
    )
    parser.add_argument(
        "--frame-dir", type=str,
        default=str(PROJECT_ROOT / "data" / "frames"),
        help="Directory containing pre-extracted frames (1fps)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "output" / "eval"),
    )
    parser.add_argument("--log-dir", type=str, default=str(PROJECT_ROOT / "log"))
    parser.add_argument("--respond-model", type=str, default="qwen3vl-8b")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    parser.add_argument("--gpus-per-worker", type=int, default=1,
                        help="Number of GPUs per worker")
    parser.add_argument("--sample-file", type=str, default=None)
    parser.add_argument("--video-ids", type=str, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--debug-log-count", type=int, default=5,
        help="Number of QAs to log full prompts for",
    )
    # High-resolution image settings
    parser.add_argument(
        "--image-max-tokens", type=int, default=512,
        help="IMAGE_MAX_TOKEN_NUM for Qwen3VL smart_resize. "
             "512 -> ~616x616 (484 tokens/frame). Default 512.",
    )
    parser.add_argument(
        "--image-max-size", type=int, default=2048,
        help="Max image size for thumbnail. Set large (e.g. 2048) to disable thumbnail "
             "and let smart_resize handle resolution. Default 2048.",
    )
    # API model options
    parser.add_argument("--service-tier", type=str, default=None,
                        help="Service tier for API models (e.g., 'flex')")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Timeout for API model calls")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Concurrency for API models (ThreadPoolExecutor workers)")

    args = parser.parse_args()

    # Process args
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
    logger.info(
        f"High-res settings: IMAGE_MAX_TOKEN_NUM={args.image_max_tokens}, "
        f"max_size=({args.image_max_size}, {args.image_max_size})"
    )

    # Create output directory
    exp_output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    output_file = os.path.join(exp_output_dir, f"{args.exp_name}.json")
    qa_progress_file = os.path.join(exp_output_dir, f"{args.exp_name}_qa_progress.jsonl")
    logger.info(f"Output: {output_file}")
    logger.info(f"Progress: {qa_progress_file}")

    # Load QA data
    logger.info(f"Loading QA data from {args.qa_file}")
    qa_data = load_json(args.qa_file)

    long_qa_data = [item for item in qa_data if item.get("duration") == "long"]
    logger.info(f"Filtered {len(long_qa_data)} QA items for long videos")

    grouped_qa = group_qa_by_video(long_qa_data)
    logger.info(f"Total {len(grouped_qa)} unique long videos")

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

    # Resume: collect existing results per worker
    existing_results_per_worker: Dict[int, List[Dict[str, Any]]] = {
        i: [] for i in range(args.num_workers)
    }
    completed_video_ids = set()

    if args.resume:
        for i in range(args.num_workers):
            temp_path = os.path.join(
                exp_output_dir, f"{args.exp_name}_temp_{i}.json"
            )
            if os.path.exists(temp_path):
                try:
                    data = load_json(temp_path)
                    if isinstance(data, list):
                        existing_results_per_worker[i] = data
                        for vr in data:
                            completed_video_ids.add(vr.get("videoID", vr["video_id"]))
                except Exception as e:
                    logger.warning(f"Failed to load temp results from {temp_path}: {e}")

        logger.info(f"Resumed {len(completed_video_ids)} completed videos")

    # Filter out completed videos
    videos_to_process = [
        (vid, qs) for vid, qs in available_videos.items()
        if vid not in completed_video_ids
    ]
    logger.info(f"Videos remaining to process: {len(videos_to_process)}")

    if not videos_to_process:
        logger.info("All videos already processed, generating final output")
        all_results = []
        for worker_list in existing_results_per_worker.values():
            all_results.extend(worker_list)
    elif is_api_model(args.respond_model):
        # API model: use ThreadPoolExecutor (single process, concurrent API calls)
        from merit.embedding import EmbeddingModel
        from merit.llm import LLMModel

        logger.info("API model detected, using ThreadPoolExecutor")

        embedding_model = EmbeddingModel()
        ts_embedding_model = ThreadSafeEmbeddingModel(embedding_model)
        logger.info("Loading text embedding model...")
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
        temp_file = os.path.join(exp_output_dir, f"{args.exp_name}_temp_0.json")

        # Collect all existing results
        all_results = []
        for worker_list in existing_results_per_worker.values():
            all_results.extend(worker_list)

        results_lock = threading.Lock()
        logger.info(
            f"Starting ThreadPoolExecutor with {concurrency} concurrent workers"
        )

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            for video_id, questions in videos_to_process:
                future = executor.submit(
                    process_video_api,
                    video_id=video_id,
                    questions=questions,
                    args=args,
                    embedding_model=ts_embedding_model,
                    respond_llm=respond_llm,
                    embedding_lock=embedding_lock,
                )
                futures[future] = video_id

            completed_count = 0
            total_to_process = len(futures)
            for future in as_completed(futures):
                video_id = futures[future]
                completed_count += 1
                try:
                    video_result = future.result()
                    if video_result:
                        with results_lock:
                            all_results.append(video_result)
                            save_json(all_results, temp_file)
                        if qa_progress_file:
                            with open(qa_progress_file, "a", encoding="utf-8") as f:
                                for q in video_result["questions"]:
                                    entry = {
                                        "video_id": video_result["video_id"],
                                        "question_id": q["question_id"],
                                        "question": q["question"][:100],
                                        "gold_answer": q["answer"],
                                        "parsed_answer": q["parsed_answer"],
                                        "correct": q["correct"],
                                    }
                                    f.write(
                                        json.dumps(entry, ensure_ascii=False) + "\n"
                                    )
                        logger.info(
                            f"Progress: {completed_count}/{total_to_process} videos done"
                        )
                except Exception as e:
                    logger.error(f"Video {video_id} failed: {e}")
    else:
        # Divide videos into worker batches
        num_workers = min(args.num_workers, len(videos_to_process))
        batches: List[List[Tuple[str, List[Dict[str, Any]]]]] = [
            [] for _ in range(num_workers)
        ]
        for i, video_item in enumerate(videos_to_process):
            batches[i % num_workers].append(video_item)

        logger.info(
            f"Starting {num_workers} workers, batch sizes: "
            f"{[len(b) for b in batches]}"
        )

        result_queue = Queue()
        processes = []

        for i, batch in enumerate(batches):
            temp_file = os.path.join(
                exp_output_dir, f"{args.exp_name}_temp_{i}.json"
            )
            worker_log = os.path.join(log_dir, f"worker_{i}.log")
            worker_existing = existing_results_per_worker.get(i, [])

            p = Process(
                target=process_video_batch,
                args=(
                    i,
                    batch,
                    worker_existing,
                    args,
                    result_queue,
                    worker_log,
                    temp_file,
                    qa_progress_file,
                ),
            )
            p.start()
            processes.append(p)

        # Collect results
        all_results = []
        for _ in range(num_workers):
            _, worker_results = result_queue.get()
            all_results.extend(worker_results)

        for p in processes:
            p.join()

    # Calculate final accuracy
    total_correct = sum(
        1 for vr in all_results for q in vr["questions"] if q.get("correct")
    )
    total_questions = sum(len(vr["questions"]) for vr in all_results)
    accuracy = total_correct / total_questions if total_questions > 0 else 0
    logger.info(f"Final Accuracy: {accuracy:.4f} ({total_correct}/{total_questions})")

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
            "total_questions": total_questions,
            "total_correct": total_correct,
            "accuracy": accuracy,
        },
        "results": all_results,
    }

    save_json(final_output, output_file)
    logger.info(f"Results saved to {output_file}")

    # Template format for submission
    output_template_format = []
    for video_result in all_results:
        template_item = {
            "video_id": video_result["video_id"],
            "duration": video_result["duration"],
            "domain": video_result["domain"],
            "sub_category": video_result["sub_category"],
            "questions": [],
        }
        for q in video_result["questions"]:
            template_item["questions"].append({
                "question_id": q["question_id"],
                "task_type": q["task_type"],
                "question": q["question"],
                "options": q["options"],
                "answer": q["answer"],
                "response": q["response"],
            })
        output_template_format.append(template_item)

    template_output_file = output_file.replace(".json", "_template.json")
    save_json(output_template_format, template_output_file)
    logger.info(f"Template format results saved to {template_output_file}")

    # Cleanup temp files on success
    for i in range(args.num_workers):
        temp_path = os.path.join(
            exp_output_dir, f"{args.exp_name}_temp_{i}.json"
        )
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Cleaned up temp file: {temp_path}")


if __name__ == "__main__":
    main()
