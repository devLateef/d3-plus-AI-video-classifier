"""
scripts/extract_all_features_fast.py
FINAL VERSION: Proper path handling for dataset root.
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
    """Extract bitrate and metadata from video files."""
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
        
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if video_stream is None:
            return np.zeros(6)
        
        format_info = data.get('format', {})
        
        return np.array([
            float(video_stream.get('bit_rate', 0)),
            float(eval(video_stream.get('avg_frame_rate', '0/1'))),
            float(video_stream.get('width', 0)),
            float(video_stream.get('height', 0)),
            float(format_info.get('duration', 0)),
            1.0 if video_stream.get('codec_name') == 'h264' else 0.0
        ])
        
    except Exception as e:
        return np.zeros(6)


def extract_bitrate_features_gif(gif_path):
    """Extract features from GIF files."""
    try:
        reader = imageio.get_reader(gif_path)
        n_frames = reader.get_length()
        
        first_frame = reader.get_data(0)
        height, width = first_frame.shape[:2]
        
        file_size = os.path.getsize(gif_path) / (1024 * 1024)
        
        try:
            metadata = reader.get_meta_data()
            duration = metadata.get('duration', 0) / 1000
        except:
            duration = 0
        
        return np.array([
            float(duration),
            float(n_frames),
            float(width),
            float(height),
            float(file_size),
            1.0
        ])
        
    except Exception as e:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def extract_bitrate_features(file_path):
    """Extract bitrate/metadata features from video or GIF files."""
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.gif':
        return extract_bitrate_features_gif(file_path)
    else:
        return extract_bitrate_features_video(file_path)


def find_original_file_path(frame_dir_path, dataset_root):
    """
    Find the original video/GIF file path from the frame directory path.
    """
    frame_dir = Path(frame_dir_path)
    dataset_root = Path(dataset_root)
    
    # Get video name (last part of the path)
    video_name = frame_dir.name
    
    # Try to find the video file by searching
    extensions = ['.mp4', '.avi', '.mov', '.mkv', '.gif']
    
    # Strategy 1: Try to match the path structure
    # frame_dir: .../frames/Fake/Show_1/show1_664
    # video:     .../Fake/Show_1/show1_664.mp4
    
    # Get the relative path from dataset root
    try:
        rel_path = frame_dir.relative_to(dataset_root)
        parts = list(rel_path.parts)
    except ValueError:
        # Try to find 'Real' or 'Fake' in the path
        parts = list(frame_dir.parts)
        # Find where 'Real' or 'Fake' appears
        base_folder = None
        for i, part in enumerate(parts):
            if part in ['Real', 'Fake']:
                base_folder = part
                parts = parts[i+1:]  # Everything after Real/Fake
                break
    
    # Remove 'frames' prefix if present
    if parts and parts[0] == 'frames':
        parts = parts[1:]
    
    # If we have parts, reconstruct the subfolder
    if parts:
        subfolder_parts = parts[:-1]  # Everything except the video name
        subfolder = '/'.join(subfolder_parts) if subfolder_parts else ''
    else:
        subfolder = ''
    
    # Determine base folder
    base_folder = None
    for folder in ['Real', 'Fake']:
        if folder in str(frame_dir):
            base_folder = folder
            break
    
    # Search paths
    search_paths = []
    
    # Path 1: dataset_root / base_folder / subfolder / video_name.ext
    if base_folder:
        if subfolder:
            search_paths.append(dataset_root / base_folder / subfolder)
        search_paths.append(dataset_root / base_folder)
    
    # Path 2: dataset_root / subfolder
    if subfolder:
        search_paths.append(dataset_root / subfolder)
    
    # Path 3: dataset_root / base_folder
    if base_folder:
        search_paths.append(dataset_root / base_folder)
    
    # Path 4: dataset_root only
    search_paths.append(dataset_root)
    
    # Try all paths and extensions
    for search_path in search_paths:
        for ext in extensions:
            file_path = search_path / f"{video_name}{ext}"
            if file_path.exists():
                return file_path
    
    # Additional fallback: search recursively in Real and Fake
    for folder in ['Real', 'Fake']:
        folder_path = dataset_root / folder
        if folder_path.exists():
            for ext in extensions:
                # Search for the video file anywhere in the folder
                matches = list(folder_path.rglob(f"{video_name}{ext}"))
                if matches:
                    return matches[0]
    
    return None


def extract_all_features_fast(csv_path, dataset_root, output_path="full_features.npz"):
    """
    Extract ALL features including bitrate, with GIF support.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset root: {dataset_root}")
    
    # Resolve dataset root
    dataset_root = Path(dataset_root).resolve()
    print(f"Resolved dataset root: {dataset_root}")
    
    # Check if Real and Fake folders exist
    real_path = dataset_root / 'Real'
    fake_path = dataset_root / 'Fake'
    
    if not real_path.exists() and not fake_path.exists():
        print(f"❌ ERROR: No 'Real' or 'Fake' folders found in {dataset_root}")
        print(f"   Real path: {real_path}")
        print(f"   Fake path: {fake_path}")
        print("\nPlease check your dataset path.")
        
        # Try to find the dataset by searching in parent directories
        parent = dataset_root.parent
        for _ in range(3):  # Search up to 3 levels up
            if (parent / 'Real').exists() and (parent / 'Fake').exists():
                print(f"✅ Found dataset in: {parent}")
                dataset_root = parent
                real_path = dataset_root / 'Real'
                fake_path = dataset_root / 'Fake'
                break
            parent = parent.parent
        
        if not real_path.exists() and not fake_path.exists():
            return
    
    print(f"Real folder: {real_path}")
    print(f"Fake folder: {fake_path}")
    
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
                
                if original_file.suffix.lower() == '.gif':
                    gif_count += 1
                else:
                    video_count += 1
            else:
                bitrate_feats = np.zeros(6)
                bitrate_errors += 1
                if idx < 5:
                    print(f"File not found for: {frame_dir}")
            
            # 5. Combine ALL features
            combined = np.concatenate([
                [d3_avg.cpu().numpy()[0]],
                [d3_std.cpu().numpy()[0]],
                color_feats,
                temporal_feats,
                bitrate_feats
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