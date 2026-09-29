#!/usr/bin/env python3
"""Extract the four retrieval keys from 30-second clip captions."""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List

from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(REPO_ROOT, "src", "merit")):
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from merit.llm import LLMModel  # noqa: E402
from merit.llm.templates.multikey_extract import (  # noqa: E402
    get_multikey_extract_prompt,
    parse_multikey_response,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

RESUME_FIELD = "event_key"
OUTPUT_SUFFIX = "_4_key"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract 4 retrieval keys from episodic captions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-file", required=True, help="30-second caption JSON")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to the directory of --input-file",
    )
    p.add_argument("--model", default="gpt-5-mini", help="Extraction LLM")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Re-extract keys even for entries that already have them",
    )
    p.add_argument(
        "--flex",
        action="store_true",
        help='Use the OpenAI "flex" service tier (cheaper, slower)',
    )
    p.add_argument("--timeout", type=float, default=None, help="Per-request timeout (s)")
    return p.parse_args()


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_keys_to_item(item: Dict[str, Any], keys: Dict[str, str]) -> None:
    """Insert the extracted keys right after the caption text, in place."""
    new_item: Dict[str, Any] = {}
    keys_inserted = False
    for k, v in item.items():
        if k in ("generated_text", "text"):
            new_item["text"] = v
            if not keys_inserted:
                new_item.update(keys)
                keys_inserted = True
        else:
            new_item[k] = v
    if not keys_inserted:
        new_item.update(keys)
    item.clear()
    item.update(new_item)


async def process_item(
    llm: LLMModel,
    item: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
) -> None:
    content = item.get("generated_text") or item.get("text")
    if content is None:
        pbar.update(1)
        return

    async with semaphore:
        try:
            response = await llm.generate_async(get_multikey_extract_prompt(content))
            apply_keys_to_item(item, parse_multikey_response(response))
        except Exception as e:
            logger.error(f"Error processing item: {e}")
        finally:
            pbar.update(1)


async def run(
    args: argparse.Namespace,
    results: List[Dict[str, Any]],
    to_process: List[int],
    output_file: str,
) -> None:
    async def periodic_save() -> None:
        while True:
            await asyncio.sleep(10)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    llm_kwargs: Dict[str, Any] = {}
    if args.flex:
        llm_kwargs["service_tier"] = "flex"
        llm_kwargs["timeout"] = 600.0 if args.timeout is None else args.timeout
    elif args.timeout is not None:
        llm_kwargs["timeout"] = args.timeout

    llm = LLMModel(model_name=args.model, **llm_kwargs)
    semaphore = asyncio.Semaphore(args.concurrency)
    save_task = asyncio.create_task(periodic_save())
    try:
        with tqdm(total=len(to_process), desc="Extracting keys") as pbar:
            await asyncio.gather(
                *(process_item(llm, results[i], semaphore, pbar) for i in to_process)
            )
    finally:
        save_task.cancel()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input_file):
        raise SystemExit(f"Input file not found: {args.input_file}")

    logger.info(f"Loading data from {args.input_file}")
    input_data = load_json(args.input_file)
    logger.info(f"Loaded {len(input_data)} entries")

    output_dir = args.output_dir or os.path.dirname(args.input_file) or "."
    os.makedirs(output_dir, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(args.input_file))
    output_file = os.path.join(output_dir, f"{stem}{OUTPUT_SUFFIX}{ext}")
    logger.info(f"Output: {output_file}")

    if args.resume and os.path.exists(output_file):
        logger.info(f"Found existing output; resuming from {output_file}")
        results = load_json(output_file)
    else:
        results = [dict(item) for item in input_data]

    to_process = [
        i
        for i, item in enumerate(results)
        if RESUME_FIELD not in item or not item[RESUME_FIELD]
    ]
    if not to_process:
        logger.info("All items already have keys. Nothing to do.")
        return

    logger.info(f"Items to process: {len(to_process)}")
    asyncio.run(run(args, results, to_process, output_file))

    logger.info(f"Saving final results to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info("Done.")


if __name__ == "__main__":
    main()
