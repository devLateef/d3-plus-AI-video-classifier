"""
scripts/video2frame.py
Extract frames from videos - Updated for Real/Fake folder structure.
"""

import os
import random
import math
import argparse
import multiprocessing
from glob import glob
from pathlib import Path
from moviepy import VideoFileClip


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
    
    # Get relative path from dataset root (preserves Real/Fake structure)
    rel_path = video_path.relative_to(dataset_path)
    
    # Remove file extension for folder name
    video_name = video_path.stem
    
    # Create frame directory path
    # Example: GenVideo/Real/video1.mp4 → GenVideo/frames/Real/video1/
    #          GenVideo/Fake/model1/video2.mp4 → GenVideo/frames/Fake/model1/video2/
    frame_dir = dataset_path / "frames" / rel_path.parent / video_name
    
    # Check if frames already exist
    if frame_dir.exists() and list(frame_dir.glob('*.jpg')):
        print(f"✓ {video_name} frames exist")
        return
    
    print(f"Processing: {rel_path}")
    
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
    
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return
    
    # Find all video files in Real/ and Fake/ folders
    video_paths = []
    
    real_dir = dataset_path / "Real"
    fake_dir = dataset_path / "Fake"
    
    if real_dir.exists():
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*gif']:
            video_paths.extend(real_dir.rglob(ext))
        print(f"Found {len(video_paths)} real videos in {real_dir}")
    
    if fake_dir.exists():
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_paths.extend(fake_dir.rglob(ext))
        print(f"Found {len([p for p in video_paths if 'Fake' in str(p)])} fake videos in {fake_dir}")
    
    total_before = len(video_paths)
    
    # Also check if there are videos directly in Real/ and Fake/ (not in subfolders)
    if real_dir.exists():
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_paths.extend(real_dir.glob(ext))
    
    if fake_dir.exists():
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_paths.extend(fake_dir.glob(ext))
    
    # Remove duplicates
    video_paths = list(set(video_paths))
    
    print(f"Found {len(video_paths)} total videos!")
    
    # Process in parallel
    args_list = [(vp, dataset_path) for vp in video_paths]
    
    with multiprocessing.Pool(processes=args.workers) as pool:
        pool.starmap(process_video, args_list)
    
    print("✅ Frame extraction complete!")


if __name__ == '__main__':
    main()