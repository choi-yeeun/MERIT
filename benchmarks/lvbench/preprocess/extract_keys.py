"""Extract the three retrieval keys from LVBench captions."""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Make the shared `merit` package importable without an install step.
MERIT_SRC = PROJECT_ROOT.parent.parent / "src"
if (MERIT_SRC / "merit").is_dir():
    sys.path.insert(0, str(MERIT_SRC))
from merit.llm import LLMModel


def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)

    log_dir = PROJECT_ROOT / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extract_3keys_{timestamp}.log"
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")
    return logger


logger = setup_logging()


def parse_args():
    parser = argparse.ArgumentParser(description="LVBench: Extract 3 retrieval keys from captions")
    parser.add_argument("--input_dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "captions"),
                        help="Directory of per-video caption JSON files")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "keys"),
                        help="Directory the 3-key JSON files are written to")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Number of concurrent LLM requests (default: 20)")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                        help="Model name (default: gpt-5-mini)")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", action="store_false", dest="resume")
    parser.add_argument("--flex", action="store_true",
                        help="Enable OpenAI flex service tier")
    parser.add_argument("--timeout", type=float, default=None,
                        help="API timeout in seconds")
    return parser.parse_args()


def load_data(json_file: str) -> List[Dict]:
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_3keys_prompt(caption: str) -> str:
    """
    Construct prompt for extracting THREE retrieval keys from a video clip caption:
    (1) event/action, (2) object-state, (3) summary key.
    No dialogue key since LVBench has no subtitles.
    """
    return (
        "You are extracting retrieval keys from an episodic video memory clip.\n\n"

        "Each input value corresponds to a ~30-second video clip and consists of:\n"
        "- physical actions and movements\n"
        "- interactions with objects\n"
        "- reflect the entire clip by summarizing the clip\n\n"

        "Your task is to extract EXACTLY THREE retrieval keys from the value.\n"
        "Do NOT write extra explanations.\n"
        "Do NOT invent events.\n"
        "Use only information explicitly present in the value.\n\n"

        "The three keys MUST correspond to the following categories:\n\n"

        "1. Event / Action key\n"
        "- What physical actions or events actually happened?\n"
        "- Focus on observable actions and interactions.\n"
        "- Use one short sentence or phrase.\n"
        "- Include the agent if identifiable (use actual names if present).\n\n"

        "2. Object-state / Item-centric key\n"
        "- What object was handled, requested, moved, or referenced?\n"
        "- Describe the object and its state or role in the scene.\n"
        "- Use one short sentence or phrase.\n\n"

        "3. Summary / Retrieval key\n"
        "- Generate ONE concise retrieval key that best represents the core event of the clip.\n"
        "- Abstract away redundant or repeated actions.\n"
        "- Capture the main entities, actions, and intent.\n"
        "- Be concise and retrieval-friendly.\n"
        "- Prefer compact keyword-style phrasing (not a full sentence).\n"
        "- Use spaces between words.\n"
        "- Stay grounded in the value; do not add details.\n\n"

        "Formatting rules:\n"
        "- Output exactly three lines\n"
        "- One key per line, in the order: event, object, summary\n"
        "- Use spaces between words\n"
        "- DO NOT use underscores (_)\n"
        "- Do not include numbering, bullets, or extra explanations\n"
        "- Each of lines 1-2 must be a single sentence or a single clause\n"
        "- Line 3 should be a short keyword-style phrase (not necessarily a sentence)\n\n"

        "# Few-shot Examples:\n\n"

        "Value:\n"
        "\"A group sits around a table with notebooks and laptops, discussing plans for an upcoming workshop. "
        "We talk about the expected number of participants and how long the session should be. "
        "Mina suggests keeping the workshop short to avoid fatigue, while Daniel proposes adding a short break. "
        "The discussion shifts to preparation details, with Sarah mentioning handouts and name tags. "
        "We decide to use digital slides instead and bring extra chargers. "
        "Before ending the meeting, we joke about setting up the projector, and I confirm the final agenda.\"\n"
        "Output:\n"
        "People discuss workshop planning and schedule during a meeting\n"
        "Workshop materials and equipment are decided and assigned\n"
        "group meeting workshop planning schedule materials logistics\n\n"

        "Value:\n"
        "\"A lone samurai walks through a snowy mountain pass, her sword strapped to her back. "
        "She pauses at a cliff edge to survey the valley below, where smoke rises from a distant village. "
        "After adjusting her straw hat against the wind, she descends the slope carefully, "
        "leaving deep footprints in the fresh snow.\"\n"
        "Output:\n"
        "A samurai walks through a snowy mountain pass and descends toward a village\n"
        "A sword is strapped to the samurai's back and a straw hat shields against wind\n"
        "samurai snowy mountain pass descent toward smoking village\n\n"

        "Now extract the three retrieval keys from the following value.\n\n"
        f"Value:\n\"{caption}\"\n\n"
        "Output:"
    )


async def process_item(llm: LLMModel, item: Dict, semaphore: asyncio.Semaphore, pbar: tqdm):
    """Process a single item asynchronously."""
    content = item.get("text")
    if content is None:
        pbar.update(1)
        return

    async with semaphore:
        try:
            prompt = get_3keys_prompt(content)
            response = await llm.generate_async(prompt)

            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            event_key = lines[0].strip("'\"") if len(lines) > 0 else ""
            object_key = lines[1].strip("'\"") if len(lines) > 1 else ""
            summary_key = lines[2].strip("'\"") if len(lines) > 2 else ""

            item["event_key"] = event_key
            item["object_key"] = object_key
            item["sum_key"] = summary_key
        except Exception as e:
            logger.error(f"Error processing item: {e}")
        finally:
            pbar.update(1)


async def process_single_file(
    llm: LLMModel,
    input_file: str,
    output_file: str,
    concurrency: int,
    resume: bool,
):
    """Process a single JSON file."""
    input_data = load_data(input_file)

    results = []
    if resume and os.path.exists(output_file):
        logger.info(f"Resuming from existing output: {output_file}")
        results = load_data(output_file)
        if len(results) != len(input_data):
            logger.warning(
                f"Output file length ({len(results)}) differs from input ({len(input_data)}). "
                "Starting fresh."
            )
            results = [dict(item) for item in input_data]
    else:
        results = [dict(item) for item in input_data]

    # Identify items to process (check for event_key since we use 3 keys)
    to_process_indices = [
        i for i, item in enumerate(results) if "event_key" not in item or not item["event_key"]
    ]

    if not to_process_indices:
        logger.info(f"All items already have keys in {os.path.basename(input_file)}. Skipping.")
        return

    logger.info(f"Processing {len(to_process_indices)}/{len(results)} items from {os.path.basename(input_file)}")

    # Periodic save
    save_event = asyncio.Event()

    async def periodic_save():
        while not save_event.is_set():
            try:
                await asyncio.wait_for(save_event.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    semaphore = asyncio.Semaphore(concurrency)
    save_task = asyncio.create_task(periodic_save())

    try:
        with tqdm(total=len(to_process_indices), desc=os.path.basename(input_file), leave=False) as pbar:
            tasks = [
                process_item(llm, results[idx], semaphore, pbar)
                for idx in to_process_indices
            ]
            await asyncio.gather(*tasks)
    finally:
        save_event.set()
        await save_task

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


async def async_main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        return

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    json_files = [
        f for f in os.listdir(args.input_dir)
        if f.endswith(".json")
    ]

    if not json_files:
        logger.error(f"No JSON files found in {args.input_dir}")
        return

    logger.info(f"Found {len(json_files)} JSON files to process")

    llm_kwargs = {}
    if args.flex:
        llm_kwargs["service_tier"] = "flex"
        if args.timeout is None:
            llm_kwargs["timeout"] = 600.0
        else:
            llm_kwargs["timeout"] = args.timeout
    elif args.timeout is not None:
        llm_kwargs["timeout"] = args.timeout

    llm = LLMModel(model_name=args.model, **llm_kwargs)

    for json_file in tqdm(json_files, desc="Files"):
        input_file = os.path.join(args.input_dir, json_file)
        output_file = os.path.join(output_dir, json_file)

        await process_single_file(
            llm=llm,
            input_file=input_file,
            output_file=output_file,
            concurrency=args.concurrency,
            resume=args.resume,
        )

    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(async_main())
