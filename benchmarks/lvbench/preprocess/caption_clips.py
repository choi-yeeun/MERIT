"""Caption each 30-second window of an LVBench video."""

import argparse
import asyncio
import glob
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from tqdm.asyncio import tqdm as tqdm_asyncio

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Make the shared `merit` package importable without an install step.
MERIT_SRC = PROJECT_ROOT.parent.parent / "src"
if (MERIT_SRC / "merit").is_dir():
    sys.path.insert(0, str(MERIT_SRC))
from merit.llm import LLMModel


def setup_logger():
    logger = logging.getLogger("gpt5_captioning")
    logger.setLevel(logging.INFO)

    log_dir = PROJECT_ROOT / "log"
    os.makedirs(log_dir, exist_ok=True)

    script_name = Path(__file__).stem
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"{script_name}_{current_time}.log"

    fh = logging.FileHandler(str(log_filename), mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def hhmmss_to_seconds(hhmmss):
    """Converts HHMMSS string to seconds."""
    if len(hhmmss) != 6:
        raise ValueError(f"Invalid HHMMSS format: {hhmmss}")
    hours = int(hhmmss[0:2])
    minutes = int(hhmmss[2:4])
    seconds = int(hhmmss[4:6])
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_hhmmss(seconds):
    """Converts seconds to HHMMSS string."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}{minutes:02d}{secs:02d}"


async def process_clip(llm, video_id, clip_folder, video_frame_dir, prompt_template, semaphore, logger):
    """Processes a single clip using GPT-5."""
    suffix = clip_folder.split('_')[-1]
    start_seconds = hhmmss_to_seconds(suffix)
    clip_path = os.path.join(video_frame_dir, clip_folder)
    frame_files = sorted(glob.glob(os.path.join(clip_path, "*.jpg")))

    if not frame_files:
        return None

    # Build message with frames only (no subtitle for LVBench)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frame_path} for frame_path in frame_files
            ] + [
                {"type": "text", "text": prompt_template},
            ],
        }
    ]

    async with semaphore:
        try:
            response = await llm.generate_async(messages)
            return {
                "start_time": suffix,
                "end_time": seconds_to_hhmmss(start_seconds + 29),
                "text": response,
                "frame_path": f"data/frames/{video_id}/{clip_folder}",
            }
        except Exception as e:
            logger.error(f"Error processing clip {video_id}/{clip_folder}: {e}")
            return None


async def process_video(llm, video_id, frame_dir, output_dir, prompt_template, semaphore, logger):
    """Processes all clips of a single video."""
    video_frame_dir = os.path.join(frame_dir, video_id)
    output_json_path = os.path.join(output_dir, f"{video_id}.json")

    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    else:
        results = []

    processed_clips = {r['start_time'] for r in results}

    clip_folders = sorted([
        d for d in os.listdir(video_frame_dir)
        if os.path.isdir(os.path.join(video_frame_dir, d))
    ])

    clips_to_process = [cf for cf in clip_folders if cf.split('_')[-1] not in processed_clips]

    if not clips_to_process:
        logger.info(f"No new clips to process for video {video_id}.")
        return

    logger.info(f"Processing video {video_id}: {len(clips_to_process)} clips to process.")

    tasks = [
        process_clip(llm, video_id, cf, video_frame_dir, prompt_template, semaphore, logger)
        for cf in clips_to_process
    ]

    clip_results = await asyncio.gather(*tasks)

    for res in clip_results:
        if res:
            results.append(res)

    # Sort by start_time
    results.sort(key=lambda x: x['start_time'])

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logger.info(f"Finished video {video_id}: {len(clips_to_process)} clips processed.")


async def async_main():
    parser = argparse.ArgumentParser(description="LVBench: GPT-5 Video Captioning (no subtitles)")
    parser.add_argument("--sample_file", type=str, default=None,
                        help="Path to a sample file containing video IDs to process")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Number of concurrent API calls")
    parser.add_argument("--flex", action="store_true",
                        help="Enable OpenAI flex service tier")
    parser.add_argument("--timeout", type=float, default=None,
                        help="API timeout in seconds (defaults to 600.0 if flex is enabled)")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Captioning model")
    parser.add_argument("--frame-dir", type=str, default=str(PROJECT_ROOT / "data" / "frames"),
                        help="1fps frames produced by preprocess/sample_frames.py")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "data" / "captions"),
                        help="Where per-video caption JSON files are written")
    parser.add_argument("--prompt-file", type=str,
                        default=str(PROJECT_ROOT / "prompt" / "captioning_gpt5_prompt.txt"))
    args = parser.parse_args()

    frame_dir = Path(args.frame_dir)
    output_dir = Path(args.output_dir)
    prompt_path = Path(args.prompt_file)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger()
    logger.info("LVBench GPT-5 captioning started (no subtitles).")

    if not frame_dir.exists():
        logger.error(f"Frame directory not found: {frame_dir}")
        logger.error("Run preprocess/sample_frames.py first.")
        return

    all_video_ids = sorted([d.name for d in frame_dir.iterdir() if d.is_dir()])

    # Filter by sample file
    if args.sample_file:
        sample_path = Path(args.sample_file)
        if not sample_path.exists():
            logger.error(f"Sample file not found: {args.sample_file}")
            return
        with open(sample_path, 'r', encoding='utf-8') as f:
            content = f.read()
            sample_ids = {s.strip() for s in re.split(r'[,\s\n]+', content) if s.strip()}
        all_video_ids = sorted([vid for vid in all_video_ids if vid in sample_ids])
        logger.info(f"Filtered videos using sample file: {len(all_video_ids)} videos match.")

    # Filter pending videos
    pending_video_ids = []
    for video_id in all_video_ids:
        video_frame_dir = frame_dir / video_id
        output_json_path = output_dir / f"{video_id}.json"
        clip_folders = [d.name for d in video_frame_dir.iterdir() if d.is_dir()]

        if output_json_path.exists():
            with open(output_json_path, 'r', encoding='utf-8') as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    results = []
            if len(results) >= len(clip_folders):
                continue
        pending_video_ids.append(video_id)

    logger.info(f"Processing {len(pending_video_ids)} videos ({len(all_video_ids) - len(pending_video_ids)} already done).")

    # Load prompt
    with open(str(prompt_path), 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # Initialize LLM
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
    semaphore = asyncio.Semaphore(args.concurrency)

    for video_id in tqdm_asyncio(pending_video_ids, desc="Total Progress"):
        await process_video(llm, video_id, str(frame_dir), str(output_dir), prompt_template, semaphore, logger)


if __name__ == "__main__":
    asyncio.run(async_main())
