"""
scripts/extract_all_features_fast.py
FULL VERSION: D3 + Color + Temporal + BITRATE + GIF SUPPORT
"""

import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy.stats import skew, kurtosis
import subprocess
import json
import imageio
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.data.dataset import D3Dataset
from research.models.d3_model import D3Model


def extract_color_features(frames):
    """Extract color features (mean, std, skew, kurtosis) from frames."""
    color_features = []
    for frame in frames:
        for c in range(3):
            channel = frame[c, :, :].flatten().cpu().numpy()
            color_features.extend([
                np.mean(channel),
                np.std(channel),
                skew(channel),
                kurtosis(channel)
            ])
    return np.array(color_features)


def extract_temporal_features(frames):
    """Extract temporal features (mean diff, std diff, max diff)."""
    diffs = []
    for i in range(1, len(frames)):
        diff = torch.abs(frames[i] - frames[i-1]).mean().item()
        diffs.append(diff)
    
    if diffs:
        return np.array([np.mean(diffs), np.std(diffs), np.max(diffs)])
    else:
        return np.zeros(3)


def extract_bitrate_features_video(video_path):
    """
    Extract bitrate and metadata from video files (.mp4, .avi, .mov, .mkv).
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-show_format',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return np.zeros(6)
        
        data = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if video_stream is None:
            return np.zeros(6)
        
        format_info = data.get('format', {})
        
        # Extract features
        bitrate = float(video_stream.get('bit_rate', 0))
        frame_rate = float(eval(video_stream.get('avg_frame_rate', '0/1')))
        width = float(video_stream.get('width', 0))
        height = float(video_stream.get('height', 0))
        duration = float(format_info.get('duration', 0))
        is_h264 = 1.0 if video_stream.get('codec_name') == 'h264' else 0.0
        
        return np.array([bitrate, frame_rate, width, height, duration, is_h264])
        
    except Exception as e:
        return np.zeros(6)


def extract_bitrate_features_gif(gif_path):
    """
    Extract features from GIF files.
    GIFs don't have bitrate, so we extract alternative features.
    """
    try:
        # Get GIF info using imageio
        reader = imageio.get_reader(gif_path)
        n_frames = reader.get_length()
        
        # Get first frame for dimensions
        first_frame = reader.get_data(0)
        height, width = first_frame.shape[:2]
        
        # Get file size
        file_size = os.path.getsize(gif_path) / (1024 * 1024)  # MB
        
        # Get duration (if available)
        try:
            metadata = reader.get_meta_data()
            duration = metadata.get('duration', 0) / 1000  # Convert to seconds
        except:
            duration = 0
        
        # GIF features: [duration, frame_count, width, height, size_mb, is_gif]
        return np.array([
            float(duration),      # Duration in seconds
            float(n_frames),      # Number of frames
            float(width),         # Width
            float(height),        # Height
            float(file_size),     # File size in MB
            1.0                   # Is GIF flag
        ])
        
    except Exception as e:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Default GIF features


def extract_bitrate_features(file_path):
    """
    Extract bitrate/metadata features from video or GIF files.
    Auto-detects file type.
    """
    file_path = Path(file_path)
    
    # Check if it's a GIF
    if file_path.suffix.lower() == '.gif':
        return extract_bitrate_features_gif(file_path)
    else:
        return extract_bitrate_features_video(file_path)


def find_original_file_path(frame_dir_path, dataset_root):
    """
    Find the original video/GIF file path from the frame directory path.
    
    Example:
        frame_dir:  GenVideo/frames/Real/video_001/
        video:      GenVideo/Real/video_001.mp4
    """
    frame_dir = Path(frame_dir_path)
    dataset_root = Path(dataset_root)
    
    # Get relative path from dataset root
    rel_path = frame_dir.relative_to(dataset_root)
    parts = list(rel_path.parts)
    
    # Remove 'frames' prefix
    if parts[0] == 'frames':
        parts = parts[1:]
    
    # video_name is the last part
    video_name = parts[-1]
    
    # The rest is the subfolder path
    subfolder = '/'.join(parts[:-1])  # 'Real' or 'Fake/model1'
    
    # Check for video/GIF file with common extensions
    video_dir = dataset_root / subfolder
    
    # Check all supported formats
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.gif']:
        file_path = video_dir / f"{video_name}{ext}"
        if file_path.exists():
            return file_path
    
    return None


def extract_all_features_fast(csv_path, dataset_root, output_path="full_features.npz"):
    """
    Extract ALL features including bitrate, with GIF support.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset root: {dataset_root}")
    
    # Load model
    print("Loading XCLIP model (pretrained)...")
    model = D3Model(encoder_type='XCLIP-16', loss_type='l2').to(device)
    
    # Load dataset
    dataset = D3Dataset(csv_path)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"Processing {len(dataset)} videos...")
    
    all_features = []
    all_labels = []
    bitrate_errors = 0
    video_count = 0
    gif_count = 0
    
    with torch.no_grad():
        for idx, (frames, label) in enumerate(tqdm(loader)):
            frames = frames.to(device)
            
            # 1. D3 Features
            _, d3_avg, d3_std = model(frames)
            
            # 2. Color Features
            frames_cpu = frames[0].cpu()
            color_feats = extract_color_features(frames_cpu)
            
            # 3. Temporal Features
            temporal_feats = extract_temporal_features(frames_cpu)
            
            # 4. Bitrate Features
            frame_dir = dataset.df.iloc[idx]['frame_dir']
            original_file = find_original_file_path(frame_dir, dataset_root)
            
            if original_file and original_file.exists():
                bitrate_feats = extract_bitrate_features(original_file)
                
                # Count file types for reporting
                if original_file.suffix.lower() == '.gif':
                    gif_count += 1
                else:
                    video_count += 1
            else:
                bitrate_feats = np.zeros(6)
                bitrate_errors += 1
            
            # 5. Combine ALL features
            combined = np.concatenate([
                [d3_avg.cpu().numpy()[0]],      # D3 avg (1)
                [d3_std.cpu().numpy()[0]],      # D3 std (1)
                color_feats,                    # Color (192)
                temporal_feats,                 # Temporal (3)
                bitrate_feats                   # Bitrate (6)
            ])
            
            all_features.append(combined)
            all_labels.append(label.numpy()[0])
    
    # Save
    features_arr = np.array(all_features)
    labels_arr = np.array(all_labels)
    
    np.savez_compressed(output_path, 
                        features=features_arr, 
                        labels=labels_arr)
    
    print(f"\n✅ Features saved to: {output_path}")
    print(f"   Feature shape: {features_arr.shape}")
    print(f"   Labels shape: {labels_arr.shape}")
    print(f"   Feature vector length: {features_arr.shape[1]}")
    print(f"   Videos processed: {video_count}")
    print(f"   GIFs processed: {gif_count}")
    print(f"   Bitrate errors: {bitrate_errors}/{len(dataset)}")
    
    return features_arr, labels_arr


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to dataset.csv')
    parser.add_argument('--dataset-root', type=str, required=True,
                       help='Path to the dataset root (where Real/ and Fake/ are)')
    parser.add_argument('--output', type=str, default='full_features.npz',
                       help='Output path for features')
    
    args = parser.parse_args()
    
    extract_all_features_fast(
        csv_path=args.csv,
        dataset_root=args.dataset_root,
        output_path=args.output
    )