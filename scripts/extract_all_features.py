"""
scripts/extract_all_features_fast.py
Extract ALL features WITHOUT training a model first.
Uses pre-trained XCLIP for D3 features directly.
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


def extract_all_features_fast(csv_path, output_path="full_features.npz"):
    """
    Extract ALL features WITHOUT a trained model.
    Uses D3 features directly from the pretrained encoder.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model WITHOUT loading trained weights (uses pretrained encoder)
    print("Loading XCLIP model (pretrained)...")
    model = D3Model(encoder_type='XCLIP-16', loss_type='l2').to(device)
    # Don't load any weights - just use the pretrained encoder!
    # The encoder is already pretrained and frozen
    
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
    
    with torch.no_grad():
        for frames, label in tqdm(loader):
            frames = frames.to(device)
            
            # 1. D3 Features (from the model - no training needed!)
            _, d3_avg, d3_std = model(frames)
            
            # 2. Color Features (handcrafted)
            frames_cpu = frames[0].cpu()
            color_feats = extract_color_features(frames_cpu)
            
            # 3. Temporal Features (handcrafted)
            temporal_feats = extract_temporal_features(frames_cpu)
            
            # 4. Combine ALL features into one vector
            combined = np.concatenate([
                [d3_avg.cpu().numpy()[0]],
                [d3_std.cpu().numpy()[0]],
                color_feats,
                temporal_feats
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