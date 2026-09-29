"""Extract 1fps frames per 30-second interval of each LVBench video."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def get_video_duration(video_path):
    """Get the duration of a video using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration for {video_path}: {e}")
        return None


def format_timestamp(seconds_total):
    """Format seconds into HHMMSS string."""
    hours = seconds_total // 3600
    minutes = (seconds_total % 3600) // 60
    seconds = seconds_total % 60
    return f"{hours:02d}{minutes:02d}{seconds:02d}"


def extract_frames_as_images(video_path, video_output_base_dir, video_id):
    """Extract 1fps frames for each 30s interval."""
    duration = get_video_duration(video_path)
    if duration is None:
        return

    interval = 30
    for start_time in range(0, int(duration), interval):
        timestamp_dir = format_timestamp(start_time)
        interval_dir_name = f"{video_id}_{timestamp_dir}"
        interval_output_dir = os.path.join(video_output_base_dir, interval_dir_name)
        os.makedirs(interval_output_dir, exist_ok=True)

        end_time = min(start_time + interval, int(duration))
        for current_time in range(start_time, end_time):
            timestamp_file = format_timestamp(current_time)
            output_filename = f"{video_id}_{timestamp_file}.jpg"
            output_path = os.path.join(interval_output_dir, output_filename)

            # Resume: skip if file already exists and is not empty
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                continue

            cmd = [
                'ffmpeg', '-y',
                '-ss', str(current_time),
                '-i', str(video_path),
                '-frames:v', '1',
                '-q:v', '2',
                str(output_path)
            ]

            try:
                subprocess.run(cmd, capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error extracting frame at {current_time}s: {e.stderr.decode()}")

    print(f"Finished processing Video ID: {video_id}")


def get_all_video_ids(meta_path):
    """Get all video IDs (keys) from video_info.meta.jsonl."""
    video_ids = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            video_ids.append(entry["key"])
    return sorted(set(video_ids))


def main():
    parser = argparse.ArgumentParser(description="LVBench: Extract 1fps frames per 30s interval.")
    parser.add_argument("--sample_file", type=str, default=None,
                        help="Path to a file containing video IDs to process.")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="Number of parallel workers.")
    base_dir = Path(__file__).resolve().parent.parent
    parser.add_argument("--meta-file", type=str,
                        default=str(base_dir / "data" / "video_info.meta.jsonl"),
                        help="LVBench video_info.meta.jsonl")
    parser.add_argument("--video-dir", type=str,
                        default=str(base_dir / "data" / "all_videos"),
                        help="Directory holding the original <videoID>.mp4 files")
    parser.add_argument("--output-dir", type=str,
                        default=str(base_dir / "data" / "frames"),
                        help="Where the output is written")
    args = parser.parse_args()

    meta_path = Path(args.meta_file)
    video_data_dir = Path(args.video_dir)
    output_base_dir = Path(args.output_dir)

    all_video_ids = get_all_video_ids(str(meta_path))
    print(f"Found {len(all_video_ids)} videos in metadata.")

    if args.sample_file:
        sample_path = Path(args.sample_file)
        if not sample_path.exists():
            print(f"Error: Sample file not found: {args.sample_file}")
            return
        with open(sample_path, 'r') as f:
            sample_ids = {s.strip() for s in f.read().replace(',', ' ').split() if s.strip()}
        all_video_ids = sorted(set(all_video_ids) & sample_ids)
        print(f"Filtered to {len(all_video_ids)} videos from sample file.")

    print(f"Starting parallel processing with {args.num_workers} workers...")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for video_id in all_video_ids:
            video_file = video_data_dir / f"{video_id}.mp4"
            if not video_file.exists():
                print(f"Warning: Video file not found: {video_file}")
                continue
            video_output_dir = output_base_dir / video_id
            futures.append(
                executor.submit(extract_frames_as_images, str(video_file), str(video_output_dir), video_id)
            )

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error during processing: {e}")

    print("\nFrame sampling complete.")


if __name__ == "__main__":
    main()
