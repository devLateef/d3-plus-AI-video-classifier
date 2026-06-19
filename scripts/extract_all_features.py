"""
scripts/extract_all_features_fast.py
UPDATED: Full feature extraction including bitrate.
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
import ffmpeg  # For bitrate extraction

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


def extract_bitrate_features(video_path):
    """
    Extract bitrate and metadata from the original video file.
    
    Returns:
        np.array: [bitrate, frame_rate, width, height, duration, codec_encoded]
    """
    try:
        probe = ffmpeg.probe(str(video_path))
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        format_info = probe.get('format', {})
        
        return np.array([
            float(video_info.get('bit_rate', 0)),              # Bitrate
            float(eval(video_info.get('avg_frame_rate', '0/1'))),  # Frame rate
            float(video_info.get('width', 0)),                 # Width
            float(video_info.get('height', 0)),                # Height
            float(format_info.get('duration', 0)),             # Duration
            float(video_info.get('codec_name') == 'h264')      # Codec (binary)
        ])
    except Exception as e:
        print(f"Error extracting bitrate: {e}")
        return np.zeros(6)  # Return zeros if extraction fails


def extract_all_features_fast(csv_path, output_path="full_features.npz"):
    """
    Extract ALL features WITHOUT a trained model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model WITHOUT loading trained weights
    print("Loading XCLIP model (pretrained)...")
    model = D3Model(encoder_type='XCLIP-16', loss_type='l2').to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,} total")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
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
    
    # Get original video paths from dataset (you'll need to map this)
    # For now, we'll use the CSV to get paths
    df = pd.read_csv(csv_path)
    
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
            
            # 4. Bitrate Features (NEW!)
            # Get the video path from the dataframe
            video_path = df.iloc[idx]['frame_dir']  # This might need adjustment
            # Convert frame_dir to original video path
            # This mapping depends on your structure
            original_video_path = Path(video_path).parent.parent / "video" / f"{Path(video_path).stem}.mp4"
            bitrate_feats = extract_bitrate_features(original_video_path)
            
            # 5. Combine ALL features
            combined = np.concatenate([
                [d3_avg.cpu().numpy()[0]],      # D3 avg (1)
                [d3_std.cpu().numpy()[0]],      # D3 std (1)
                color_feats,                    # Color (16 frames * 12 = 192)
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
    
    return features_arr, labels_arr


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to dataset.csv')
    parser.add_argument('--output', type=str, default='full_features.npz',
                       help='Output path for features')
    
    args = parser.parse_args()
    
    extract_all_features_fast(
        csv_path=args.csv,
        output_path=args.output
    )