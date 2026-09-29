#!/usr/bin/env python3
"""MERIT evaluation on EgoLifeQA."""

import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(REPO_ROOT, "src", "merit")):
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from merit.embedding import EmbeddingModel  # noqa: E402
from merit.llm import LLMModel  # noqa: E402
from merit import MeritMemory  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

VALID_KEYS = ("event", "dialogue", "object", "summary")


class ThreadSafeEmbeddingModel:
    """Serialise embedding calls so worker threads can share one GPU model."""

    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()

    def encode_text(self, *args, **kwargs):
        with self.lock:
            return self.model.encode_text(*args, **kwargs)

    def encode(self, *args, **kwargs):
        with self.lock:
            return self.model.encode(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


# ---------------------------------------------------------------- data helpers


def load_json(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(text: str) -> str:
    return text.lower().strip().rstrip(".,)")


def parse_reasoning_answer(response: str) -> Tuple[str, str]:
    """Split an ``ANSWER: X`` / ``REASONING: ...`` response into its two parts."""
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


def extract_choice_letter(text: str) -> Optional[str]:
    match = re.match(r"\(?([A-Za-z])[\.\)]?\s*", text.strip())
    return match.group(1).upper() if match else None


def evaluate_prediction(
    prediction: str, gold_letter: str, choices: Dict[str, str]
) -> bool:
    pred_norm = normalize(prediction)
    gold_candidate = normalize(choices[gold_letter])
    if pred_norm == gold_candidate:
        return True
    pred_letter = extract_choice_letter(prediction)
    if pred_letter == gold_letter:
        return True
    return False


def query_time_int(row: Dict[str, Any]) -> int:
    """``{"date": "DAY1", "time": "11094300"}`` -> ``111094300`` (DHHMMSS00)."""
    qt = row["query_time"]
    if isinstance(qt, int):
        return qt
    day = qt["date"].replace("DAY", "").replace("Day", "")
    return int(day + qt["time"].zfill(8))


def temp_file_path(output_dir: str, exp_name: str, worker_id: int) -> str:
    return os.path.join(output_dir, f"{exp_name}_temp_{worker_id}.json")


def collect_existing_results(
    output_dir: str, exp_name: str, num_workers: int
) -> Dict[int, List[Dict[str, Any]]]:
    """Load per-worker temp files written by a previous (interrupted) run."""
    existing_results: Dict[int, List[Dict[str, Any]]] = {
        i: [] for i in range(num_workers)
    }
    for i in range(num_workers):
        path = temp_file_path(output_dir, exp_name, i)
        try:
            if os.path.exists(path):
                data = load_json(path)
                if isinstance(data, list):
                    existing_results[i] = data
        except Exception as e:
            logger.warning(f"Failed to load existing results from {path}: {e}")
    return existing_results


# ------------------------------------------------------------------ the worker


def process_qa_batch(
    worker_id: int,
    qa_batch: List[Dict[str, Any]],
    existing_results: List[Dict[str, Any]],
    args: argparse.Namespace,
    result_queue: Queue,
    worker_log_file: Optional[str] = None,
    worker_temp_file: Optional[str] = None,
) -> None:
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

    from merit.eval_utils import setup_worker_gpus

    device, llm_device_map = setup_worker_gpus(
        worker_id,
        getattr(args, "gpu_ids", None) or None,
        getattr(args, "gpus_per_worker", 1),
    )

    # Stagger startup so workers do not load model weights at the same moment.
    time.sleep(worker_id * args.worker_start_delay)

    # Must be set before the processor is constructed.
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

    embedding_device = (
        f"cuda:{args.gpus_per_worker - 1}" if args.gpus_per_worker > 1 else device
    )
    embedding_model = EmbeddingModel(
        text_model_name=args.embedding_model, device=embedding_device
    )
    logger.info(
        f"Worker {worker_id}: Embedding model on {embedding_device} "
        f"(LLM on {llm_device_map})"
    )

    llm_kwargs = dict(
        model_name=args.respond_model,
        service_tier=args.service_tier,
        timeout=args.timeout,
        device_map=llm_device_map,
    )
    if args.image_max_size is not None:
        llm_kwargs["max_size"] = (args.image_max_size, args.image_max_size)

    respond_llm_model = LLMModel(**llm_kwargs)

    memory = MeritMemory(
        embedding_model=embedding_model,
        respond_llm_model=respond_llm_model,
        cache_dir=args.cache_dir,
        max_rounds=args.max_rounds,
        debug_log_count=args.debug_log_count,
    )
    memory.set_top_k(args.top_k)
    if args.key is not None:
        memory.set_active_keys(args.key)
    memory.load_keys(args.key_path, video_root=args.video_root)

    if qa_batch:
        memory.index(max(query_time_int(r) for r in qa_batch))

    memory.embedding_model = ThreadSafeEmbeddingModel(memory.embedding_model)
    memory.retriever.embedding_model = memory.embedding_model

    logger.info(f"Worker {worker_id}: Force loading text embedding model...")
    memory.embedding_model.load_model("text")
    logger.info(f"Worker {worker_id}: Text embedding model loaded.")

    def process_single_question(row: Dict[str, Any]) -> Dict[str, Any]:
        ID = row["ID"]
        question = row["question"]
        answer = row["answer"]
        choices = {
            label: row[f"choice_{label.lower()}"]
            for label in ["A", "B", "C", "D"]
            if f"choice_{label.lower()}" in row
        }
        until_time = query_time_int(row)
        query_type = row.get("type", "")

        logger.info(
            f"Worker {worker_id}: Processing Question ID {ID} | "
            f"Query: {question[:100]}..."
        )
        try:
            qa_result = memory.answer(
                query=question, choices=choices, until_time=until_time, question_id=ID
            )
            response = qa_result.answer
            parsed_answer, reasoning = parse_reasoning_answer(response)
        except Exception as e:
            logger.error(f"Worker {worker_id} Error ID {ID}: {e}")
            response = "Error"
            parsed_answer = ""
            reasoning = ""
            qa_result = None

        eval_text = parsed_answer if parsed_answer else response
        return {
            "ID": ID,
            "type": query_type,
            "question": question,
            "choices": choices,
            "answer": answer,
            "response": response,
            "parsed_answer": parsed_answer,
            "reasoning": reasoning,
            "evaluate": evaluate_prediction(eval_text, answer, choices),
            "query_time": row["query_time"],
            "query_time_int": until_time,
            "round_history": qa_result.round_history if qa_result else [],
            "raw_responses": qa_result.raw_responses if qa_result else [],
        }

    worker_results = list(existing_results)

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        future_to_row = {
            executor.submit(process_single_question, row): row for row in qa_batch
        }
        for future in as_completed(future_to_row):
            try:
                worker_results.append(future.result())
                if worker_temp_file:
                    with open(worker_temp_file, "w") as f:
                        json.dump(worker_results, f, indent=4)
            except Exception as e:
                logger.error(f"Worker {worker_id}: Task failed with error: {e}")

    worker_results.sort(key=query_time_int)
    result_queue.put((worker_id, worker_results))
    memory.cleanup()


# ------------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MERIT evaluation on EgoLifeQA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- what to evaluate ---
    p.add_argument("--subject", default="A1_JAKE", help="EgoLife subject id")
    p.add_argument(
        "--exp-name",
        default=None,
        help="Run name, used for the output directory and result filenames",
    )
    p.add_argument(
        "--data-dir", default="data/EgoLife", help="Root of the prepared EgoLife data"
    )
    p.add_argument(
        "--qa-file",
        default=None,
        help="EgoLifeQA json. Defaults to <data-dir>/EgoLifeQA/EgoLifeQA_<subject>.json",
    )
    p.add_argument(
        "--caption-dir",
        default=None,
        help="Defaults to <data-dir>/captions/<subject>",
    )
    p.add_argument(
        "--video-root",
        default=None,
        help="Clips laid out as <subject>/<DAY>/<clip>.mp4. Overrides the "
        "video_path stored in the key file. Defaults to <data-dir>/videos",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Key embedding cache. Defaults to .cache/multi_keys/<subject>",
    )
    p.add_argument("--output-dir", default="output", help="Root for run outputs")
    p.add_argument("--log-dir", default=None, help="Root for per-worker logs")

    # --- models ---
    p.add_argument(
        "--respond-model",
        default="gpt-5",
        help="LLM for reasoning, neighbor filtering and final QA",
    )
    p.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-4B",
        help="Sentence-transformers model used to embed retrieval keys",
    )

    # --- retrieval / inference ---
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--key",
        nargs="*",
        default=None,
        help=f"Key types used for MaxSim. Default: all of {', '.join(VALID_KEYS)}",
    )
    p.add_argument(
        "--image-max-tokens",
        type=int,
        default=512,
        help="Qwen3VL IMAGE_MAX_TOKEN_NUM budget",
    )
    p.add_argument(
        "--image-max-size",
        type=int,
        default=2048,
        help="Thumbnail cap for API models; keep large to disable thumbnailing",
    )

    # --- execution ---
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Concurrent questions per worker. Keep at 1 for local models",
    )
    p.add_argument("--gpu-ids", default=None, help="Comma-separated, e.g. 0,1,2,3")
    p.add_argument("--gpus-per-worker", type=int, default=1)
    p.add_argument(
        "--worker-start-delay",
        type=float,
        default=10.0,
        help="Seconds between worker startups",
    )
    p.add_argument("--timeout", type=float, default=None, help="Per-request timeout (s)")
    p.add_argument(
        "--service-tier", default=None, help='OpenAI service tier, e.g. "flex"'
    )

    # --- subsetting / debugging ---
    p.add_argument("--max-day", type=int, default=None)
    p.add_argument("--start-num", type=int, default=None, help="1-indexed, inclusive")
    p.add_argument("--end-num", type=int, default=None, help="1-indexed, inclusive")
    p.add_argument(
        "--question-ids",
        nargs="+",
        default=None,
        help="Evaluate only these question IDs (skips resume)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing temp files and start over",
    )
    p.add_argument("--debug-log-count", type=int, default=5)
    return p


def resolve_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Fill in every path that defaults off --data-dir / --subject."""
    if args.qa_file is None:
        args.qa_file = os.path.join(
            args.data_dir, "EgoLifeQA", f"EgoLifeQA_{args.subject}.json"
        )
    if args.caption_dir is None:
        args.caption_dir = os.path.join(args.data_dir, "captions", args.subject)
    if args.video_root is None:
        args.video_root = os.path.join(args.data_dir, "videos")
    if args.cache_dir is None:
        args.cache_dir = os.path.join(".cache", "multi_keys", args.subject)

    if not os.path.exists(args.qa_file):
        parser.error(f"QA file not found: {args.qa_file}")

    key_files = sorted(glob.glob(os.path.join(args.caption_dir, "*_30sec_4_key.json")))
    if not key_files:
        parser.error(
            f"No '*_30sec_4_key.json' found in {args.caption_dir}. "
            "Run preprocess/extract_keys.py first, or point "
            "--caption-dir at the prepared directory."
        )
    if len(key_files) > 1:
        logger.warning(
            f"Multiple key files in {args.caption_dir}; using {key_files[0]}"
        )
    args.key_path = key_files[0]

    if not os.path.isdir(args.video_root):
        logger.warning(
            f"--video-root {args.video_root} does not exist; frame sampling will "
            "fall back to the video_path stored in the key file."
        )


def default_exp_name(args: argparse.Namespace) -> str:
    name = f"merit_{args.respond_model}_top{args.top_k}"
    if args.key is not None:
        name += f"_key_{'_'.join(args.key)}"
    return name


def main() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = build_parser()
    args = parser.parse_args()

    if args.key is not None:
        args.key = [{"dialog": "dialogue"}.get(k, k) for k in args.key]
        for k in args.key:
            if k not in VALID_KEYS:
                parser.error(f"Invalid --key '{k}'. Valid: {list(VALID_KEYS)}")
        logger.info(f"Active key types: {args.key}")

    if args.gpu_ids:
        args.gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]

    from merit.eval_utils import validate_gpu_config

    validate_gpu_config(args.num_workers, args.gpu_ids or None, args.gpus_per_worker)

    resolve_paths(args, parser)

    if args.exp_name is None:
        args.exp_name = default_exp_name(args)
    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = args.log_dir or os.path.join(".log", args.exp_name)
    log_dir = os.path.join(log_root, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    logger.info(f"Experiment    : {args.exp_name}")
    logger.info(f"Subject       : {args.subject}")
    logger.info(f"Respond model : {args.respond_model}")
    logger.info(f"Key file      : {args.key_path}")
    logger.info(f"Video root    : {args.video_root}")
    logger.info(f"Cache dir     : {args.cache_dir}")
    logger.info(f"Output dir    : {output_dir}")
    logger.info(f"Log dir       : {log_dir}")
    logger.info(
        f"top_k={args.top_k} max_rounds={args.max_rounds} "
        f"image_max_tokens={args.image_max_tokens} image_max_size={args.image_max_size}"
    )

    eval_data = load_json(args.qa_file)
    if args.max_day:
        eval_data = [
            r
            for r in eval_data
            if int(r["query_time"]["date"].replace("DAY", "")) <= args.max_day
        ]

    if args.question_ids is not None:
        target_ids = set(args.question_ids)
        eval_data = [r for r in eval_data if r["ID"] in target_ids]
        eval_data.sort(key=query_time_int)
        logger.info(f"Question ID mode: {len(eval_data)} questions selected")
        existing_results_per_worker = {i: [] for i in range(args.num_workers)}
    else:
        if args.no_resume:
            existing_results_per_worker = {i: [] for i in range(args.num_workers)}
        else:
            existing_results_per_worker = collect_existing_results(
                output_dir, args.exp_name, args.num_workers
            )
        existing_ids = {
            r["ID"]
            for worker_list in existing_results_per_worker.values()
            for r in worker_list
            if "ID" in r
        }
        if existing_ids:
            logger.info(f"Resuming: {len(existing_ids)} questions already done")
        eval_data = [r for r in eval_data if r["ID"] not in existing_ids]
        eval_data.sort(key=query_time_int)

        total = len(eval_data)
        range_start = (args.start_num - 1) if args.start_num is not None else 0
        range_end = args.end_num if args.end_num is not None else total
        range_start = max(0, min(range_start, total))
        range_end = max(0, min(range_end, total))
        if args.start_num is not None or args.end_num is not None:
            logger.info(f"QA range: {range_start + 1}-{range_end} of {total}")
        eval_data = eval_data[range_start:range_end]

    all_results: List[Dict[str, Any]] = []
    if not eval_data:
        logger.info("No samples left to process.")
        for worker_list in existing_results_per_worker.values():
            all_results.extend(worker_list)
        if not all_results:
            return
    if eval_data:
        # On CPU, so no GPU memory is held while the workers spawn.
        logger.info("Pre-indexing episodic keys (on CPU)...")
        embedding_model = EmbeddingModel(
            text_model_name=args.embedding_model, device="cpu"
        )
        memory = MeritMemory(
            embedding_model=embedding_model,
            respond_llm_model=None,
            cache_dir=args.cache_dir,
        )
        if args.key is not None:
            memory.set_active_keys(args.key)
        memory.load_keys(
            args.key_path, video_root=args.video_root
        )
        memory.index(max(query_time_int(r) for r in eval_data))
        del embedding_model, memory

    num_samples = len(eval_data)
    num_workers = args.num_workers
    batches = [
        eval_data[(i * num_samples) // num_workers : ((i + 1) * num_samples) // num_workers]
        for i in range(num_workers)
    ]

    result_queue: Queue = Queue()
    processes: List[Process] = []
    workers_launched = 0

    for i, batch in enumerate(batches):
        worker_existing = existing_results_per_worker.get(i, [])
        if not batch:
            all_results.extend(worker_existing)
            continue
        workers_launched += 1
        p = Process(
            target=process_qa_batch,
            args=(
                i,
                batch,
                worker_existing,
                args,
                result_queue,
                os.path.join(log_dir, f"worker_{i}.log"),
                temp_file_path(output_dir, args.exp_name, i),
            ),
        )
        p.start()
        processes.append(p)

    for _ in range(workers_launched):
        _, worker_results = result_queue.get()
        all_results.extend(worker_results)

    for p in processes:
        p.join()

    all_results.sort(key=query_time_int)

    final_output = os.path.join(output_dir, f"{args.exp_name}_final.json")
    with open(final_output, "w") as f:
        json.dump(all_results, f, indent=4)

    correct = sum(1 for r in all_results if r.get("evaluate"))
    accuracy = correct / len(all_results) if all_results else 0.0
    logger.info(f"Final Accuracy: {accuracy:.4f}  ({correct}/{len(all_results)})")
    logger.info(f"Results saved to {final_output}")
    logger.info(
        f"Per-type breakdown: python eval/score_egolife.py {final_output}"
    )


if __name__ == "__main__":
    main()
