"""
shared/video2frame.py
Extract frames from videos while preserving folder structure.
"""

import os
import random
import math
import argparse
import multiprocessing
from glob import glob
from pathlib import Path
from moviepy.editor import VideoFileClip


def get_video_length(file_path):
    """Get video duration in seconds."""
    video = VideoFileClip(file_path)
    duration = video.duration
    video.close()
    return duration


def process_video(video_path, dataset_path):
    """
    Extract frames from a single video.
    Preserves folder structure for fake videos.
    """
    video_path = Path(video_path)
    dataset_path = Path(dataset_path)
    
    # Get relative path from dataset root
    rel_path = video_path.relative_to(dataset_path / "video")
    
    # Remove file extension for folder name
    video_name = video_path.stem
    
    # Create frame directory path
    # This preserves subfolder structure (e.g., fake/model1/video1)
    frame_dir = dataset_path / "frames" / rel_path.parent / video_name
    
    # Check if frames already exist
    if frame_dir.exists() and list(frame_dir.glob('*.jpg')):
        print(f"✓ {video_name} frames exist")
        return
    
    print(f"Processing: {video_path.relative_to(dataset_path)}")
    
    try:
        # Get video duration
        video_length = get_video_length(video_path)
        
        # Settings
        frame_rate = 8
        duration = 3
        
        # Calculate start time
        if video_length <= duration:
            start_time = 0
        else:
            start_time = math.floor(random.uniform(0, video_length - duration))
        
        # Create frame directory
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract frames using ffmpeg
        output_pattern = str(frame_dir / "%d.jpg")
        
        import subprocess
        cmd = [
            "ffmpeg",
            "-loglevel", "quiet",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", str(video_path),
            "-vf", f"fps={frame_rate}",
            output_pattern
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Extracted frames for: {video_name}")
        
    except Exception as e:
        with open('error.log', 'a') as f:
            f.write(f"{video_path} error: {e}\n")
        print(f"✗ Error: {video_name}")


def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--dataset-path', type=str, default='GenVideo',
                       help='Path to the dataset directory')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers')
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    video_dir = dataset_path / "video"
    
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        return
    
    # Find all video files (recursively to handle subfolders)
    video_paths = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_paths.extend(video_dir.rglob(ext))
    
    print(f"Found {len(video_paths)} videos!")
    
    # Process in parallel
    args_list = [(vp, dataset_path) for vp in video_paths]
    
    with multiprocessing.Pool(processes=args.workers) as pool:
        pool.starmap(process_video, args_list)
    
    print("✅ Frame extraction complete!")


if __name__ == '__main__':
    main()