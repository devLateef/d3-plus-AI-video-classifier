"""
research/data/dataset.py
Updated to return raw features for ablation studies.
"""

import os
import re
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from torch.utils.data import Dataset
import albumentations as A


def get_number_from_filename(filename: str) -> int:
    """Extract number from filename for sorting."""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')


def crop_center_by_percentage(image: np.ndarray, percentage: float) -> np.ndarray:
    """Crop center of image by percentage."""
    height, width = image.shape[:2]
    if width > height:
        left_pixels = int(width * percentage)
        right_pixels = int(width * percentage)
        return image[:, left_pixels:width - right_pixels]
    else:
        up_pixels = int(height * percentage)
        down_pixels = int(height * percentage)
        return image[up_pixels:height - down_pixels, :]


def get_preprocessing_pipeline(aug_type: Optional[str] = None, 
                               aug_quality: Optional[int] = None) -> A.Compose:
    """Get preprocessing pipeline for images."""
    aug_list = [A.Resize(224, 224)]
    
    if aug_type == 'Gaussian_blur':
        aug_list.append(
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(aug_quality, aug_quality), p=1.0)
        )
    elif aug_type == 'JPEG_compression':
        aug_list.append(
            A.ImageCompression(quality_range=(aug_quality, aug_quality), p=1.0)
        )
    
    aug_list.append(
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    )
    
    return A.Compose(aug_list)


class D3Dataset(Dataset):
    """
    D3 Dataset - loads from a single CSV file with labels.
    Returns frames for D3 model + raw features for ablation.
    """
    
    def __init__(
        self,
        csv_path: Path,
        max_samples: int = 9999999,
        aug_type: Optional[str] = None,
        aug_quality: Optional[int] = None
    ):
        super(D3Dataset, self).__init__()
        
        self.df = pd.read_csv(csv_path).head(max_samples)
        self.trans = get_preprocessing_pipeline(aug_type, aug_quality)
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        if 'label' in self.df.columns:
            real_count = len(self.df[self.df['label'] == 0])
            fake_count = len(self.df[self.df['label'] == 1])
            print(f"  Real: {real_count}, Fake: {fake_count}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Returns frames, label, and raw features for ablation.
        """
        row = self.df.iloc[idx]
        label = int(row['label'])
        frame_dir = Path(row['frame_dir'])
        
        # Read frames
        frames, raw_features = self._read_video_frames_with_features(frame_dir)
        
        return frames, label, raw_features
    
    def _read_video_frames_with_features(self, frame_dir: Path) -> Tuple[torch.Tensor, Dict]:
        """Read frames and extract raw features for ablation."""
        frame_files = sorted(
            [p for p in frame_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']],
            key=lambda x: get_number_from_filename(x.stem)
        )
        
        total_frames = len(frame_files)
        if total_frames < 8:
            raise ValueError(f"Not enough frames in {frame_dir}: {total_frames}")
        
        n_frames = 8 if total_frames < 16 else 16
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        
        frames = []
        raw_features = {
            'frame_count': total_frames,
            'mean_color': [],
            'std_color': [],
            'frame_diffs': []
        }
        
        prev_frame = None
        
        for idx in indices:
            frame_path = frame_files[idx]
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            
            # For D3 model (preprocessed)
            image_processed = crop_center_by_percentage(image.copy(), 0.1)
            augmented = self.trans(image=image_processed)
            image_tensor = augmented["image"]
            frames.append(image_tensor.transpose(2, 0, 1))
            
            # For ablation features (raw)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Color statistics (for color features ablation)
            for c, name in enumerate(['R', 'G', 'B']):
                channel = image_rgb[:, :, c].flatten()
                raw_features['mean_color'].append(np.mean(channel))
                raw_features['std_color'].append(np.std(channel))
            
            # Frame differences (for temporal features ablation)
            if prev_frame is not None:
                diff = np.mean(np.abs(image_rgb - prev_frame))
                raw_features['frame_diffs'].append(diff)
            
            prev_frame = image_rgb
        
        # Aggregate temporal features
        if raw_features['frame_diffs']:
            raw_features['mean_diff'] = np.mean(raw_features['frame_diffs'])
            raw_features['std_diff'] = np.std(raw_features['frame_diffs'])
            raw_features['max_diff'] = np.max(raw_features['frame_diffs'])
        else:
            raw_features['mean_diff'] = 0.0
            raw_features['std_diff'] = 0.0
            raw_features['max_diff'] = 0.0
        
        # Aggregate color features
        raw_features['mean_color'] = np.mean(raw_features['mean_color'])
        raw_features['std_color'] = np.mean(raw_features['std_color'])
        
        return torch.tensor(np.stack(frames), dtype=torch.float32), raw_features