import os
import sys
import json
import glob
import re
import math
import random
import argparse
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any
from tqdm.asyncio import tqdm as tqdm_asyncio

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Make the shared `merit` package importable without an install step.
MERIT_SRC = PROJECT_ROOT.parent.parent / "src"
if (MERIT_SRC / "merit").is_dir():
    sys.path.insert(0, str(MERIT_SRC))
from merit.llm import LLMModel

def setup_logger():
    """Sets up a logger for the script."""
    logger = logging.getLogger("gpt5_captioning")
    logger.setLevel(logging.INFO)

    # Define log directory (relative to project root)
    log_dir = PROJECT_ROOT / "log"
    os.makedirs(log_dir, exist_ok=True)
    
    # File name format: script_name_timestamp
    script_name = Path(__file__).stem
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"{script_name}_{current_time}.log"
    
    # File handler
    fh = logging.FileHandler(str(log_filename), mode='w')
    fh.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    return logger

def parse_srt_time(time_str):
    """Converts SRT timestamp string to seconds."""
    hours, minutes, seconds = time_str.replace(',', '.').split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

def parse_srt(srt_path):
    """Parses an SRT file into a list of subtitles."""
    if not os.path.exists(srt_path):
        return []
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to match SRT blocks
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:(?!\n\n).)*)', re.DOTALL)
    matches = pattern.findall(content)

    subtitles = []
    for match in matches:
        index, start_str, end_str, text = match
        start_time = parse_srt_time(start_str)
        end_time = parse_srt_time(end_str)
        clean_text = text.replace('\n', ' ').strip()
        # Remove HTML tags if any (e.g. <font color="...">)
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        
        subtitles.append({
            'start': start_time,
            'end': end_time,
            'text': clean_text
        })
    return subtitles

def get_subtitle_for_clip(subtitles, clip_start, clip_end):
    """Extracts subtitles that overlap with the clip time range."""
    clip_subs = []
    for sub in subtitles:
        # Check for overlap
        # Overlap exists if (sub_start < clip_end) and (sub_end > clip_start)
        if sub['start'] < clip_end and sub['end'] > clip_start:
            clip_subs.append(sub['text'])
    
    return "\n".join(clip_subs)

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

async def process_clip(llm, video_id, clip_folder, video_frame_dir, subtitles, prompt_template, semaphore, logger):
    """Processes a single clip using GPT-5-Mini."""
    suffix = clip_folder.split('_')[-1]
    start_seconds = hhmmss_to_seconds(suffix)
    clip_subtitle = get_subtitle_for_clip(subtitles, start_seconds, start_seconds + 30)
    clip_path = os.path.join(video_frame_dir, clip_folder)
    frame_files = sorted(glob.glob(os.path.join(clip_path, "*.jpg")))
    
    if not frame_files:
        return None

    # Prepare message for GPT-5-Mini
    # OpenAIModel wrapper handles image paths by encoding them to base64
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frame_path} for frame_path in frame_files
            ] + [
                {"type": "text", "text": prompt_template.format(subtitle=clip_subtitle, caption="")},
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

async def process_video(llm, video_id, video_dir, subtitle_dir, output_dir, prompt_template, semaphore, logger):
    """Processes all clips of a single video."""
    video_frame_dir = os.path.join(video_dir, video_id)
    srt_path = os.path.join(subtitle_dir, f"{video_id}.srt")
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
    subtitles = parse_srt(srt_path)
    clip_folders = sorted([d for d in os.listdir(video_frame_dir)
                           if os.path.isdir(os.path.join(video_frame_dir, d))])
    
    clips_to_process = [cf for cf in clip_folders if cf.split('_')[-1] not in processed_clips]
    
    if not clips_to_process:
        logger.info(f"No new clips to process for video {video_id}.")
        return

    logger.info(f"Processing video {video_id}: {len(clips_to_process)} clips to process.")

    tasks = [
        process_clip(llm, video_id, cf, video_frame_dir, subtitles, prompt_template, semaphore, logger)
        for cf in clips_to_process
    ]
    
    clip_results = await asyncio.gather(*tasks)
    
    for res in clip_results:
        if res:
            results.append(res)
            
    # Sort results by start_time
    results.sort(key=lambda x: x['start_time'])
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Finished video {video_id}: {len(clips_to_process)} clips processed.")

async def async_main():
    parser = argparse.ArgumentParser(description="GPT-5-Mini Video Captioning")
    parser.add_argument("--sample_file", type=str, default=None, help="Path to a sample file containing video IDs to process")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent API calls")
    parser.add_argument("--flex", action="store_true", help="Enable OpenAI flex service tier")
    parser.add_argument("--timeout", type=float, default=None, help="API timeout in seconds (defaults to 600.0 if flex is enabled)")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Captioning model")
    parser.add_argument("--frame-dir", type=str, default=str(PROJECT_ROOT / "data" / "frames"),
                        help="1fps frames produced by preprocess/sample_frames.py")
    parser.add_argument("--subtitle-dir", type=str, default=str(PROJECT_ROOT / "data" / "subtitle"),
                        help="Video-MME SRT subtitles")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "data" / "captions"),
                        help="Where per-video caption JSON files are written")
    parser.add_argument("--prompt-file", type=str,
                        default=str(PROJECT_ROOT / "prompt" / "captioning_gpt5_prompt.txt"))
    args = parser.parse_args()

    video_dir = Path(args.frame_dir)
    subtitle_dir = Path(args.subtitle_dir)
    output_dir = Path(args.output_dir)
    prompt_path = Path(args.prompt_file)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logger()
    logger.info("GPT-5-Mini processing started.")

    # Get all video IDs
    if not video_dir.exists():
        logger.error(f"Directory not found: {video_dir}")
        return

    all_video_ids = sorted([d.name for d in video_dir.iterdir() if d.is_dir()])
    
    # Filter by sample file if provided
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
        video_frame_dir = video_dir / video_id
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

    logger.info(f"Processing {len(pending_video_ids)} videos.")

    # Load Prompt
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
    
    # Process videos one by one (to manage JSON files) but clips within video concurrently
    for video_id in tqdm_asyncio(pending_video_ids, desc="Total Progress"):
        await process_video(llm, video_id, str(video_dir), str(subtitle_dir), str(output_dir), prompt_template, semaphore, logger)

if __name__ == "__main__":
    asyncio.run(async_main())
