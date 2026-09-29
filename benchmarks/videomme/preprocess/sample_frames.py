import argparse
import json
import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    """Extract 30 frames at 1 fps for each 30s interval, naming each with its timestamp."""
    duration = get_video_duration(video_path)
    if duration is None:
        return

    # 30-second intervals
    interval = 30
    for start_time in range(0, int(duration), interval):
        # Calculate timestamp for directory name: HHMMSS
        timestamp_dir = format_timestamp(start_time)
        
        # Directory for this interval: {videoID}_HHMMSS
        interval_dir_name = f"{video_id}_{timestamp_dir}"
        interval_output_dir = os.path.join(video_output_base_dir, interval_dir_name)
        os.makedirs(interval_output_dir, exist_ok=True)
        
        # Extract each frame individually to ensure correct naming
        # We extract 30 frames (or until the end of the video)
        end_time = min(start_time + interval, int(duration))
        for current_time in range(start_time, end_time):
            timestamp_file = format_timestamp(current_time)
            output_filename = f"{video_id}_{timestamp_file}.jpg"
            output_path = os.path.join(interval_output_dir, output_filename)
            
            # Resume check: skip if file already exists and is not empty
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                continue

            # ffmpeg command to extract a single frame at a specific time
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(current_time),
                '-i', str(video_path),
                '-frames:v', '1',
                '-q:v', '2',  # High quality
                str(output_path)
            ]
            
            try:
                # Use capture_output=True to keep the console clean
                subprocess.run(cmd, capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error extracting frame at {current_time}s: {e.stderr.decode()}")
    
    print(f"Finished processing Video ID: {video_id}")

def main():
    base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Extract 1fps frames per 30-second interval of each long video."
    )
    parser.add_argument("--qa-file", type=str,
                        default=str(base_dir / "data" / "test_qa.json"),
                        help="Video-MME test_qa.json")
    parser.add_argument("--video-dir", type=str,
                        default=str(base_dir / "data" / "full_video"),
                        help="Directory holding the original <videoID>.mp4 files")
    parser.add_argument("--output-dir", type=str,
                        default=str(base_dir / "data" / "frames"),
                        help="Where the extracted frames are written")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="Number of parallel workers")
    args = parser.parse_args()

    json_path = Path(args.qa_file)
    video_data_dir = Path(args.video_dir)
    output_base_dir = Path(args.output_dir)

    # Load JSON
    print(f"Loading {json_path}...")
    with open(str(json_path), 'r') as f:
        data = json.load(f)

    # Filter for long videos and get unique videoIDs
    long_video_ids = set()
    for entry in data:
        if entry.get("duration") == "long":
            video_id = entry.get("videoID")
            if video_id:
                long_video_ids.add(video_id)

    print(f"Found {len(long_video_ids)} unique long videos.")

    # Process each video in parallel
    num_workers = args.num_workers
    print(f"Starting parallel processing with {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for video_id in sorted(list(long_video_ids)):
            video_file = video_data_dir / f"{video_id}.mp4"
            if not video_file.exists():
                print(f"Warning: Video file not found: {video_file}")
                continue

            video_output_dir = output_base_dir / video_id
            futures.append(executor.submit(extract_frames_as_images, str(video_file), str(video_output_dir), video_id))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred during parallel processing: {e}")

    print("\nPreprocessing complete.")

if __name__ == "__main__":
    main()
