"""
research/data/dataset.py
Fully fixed: consistent frame size, proper tensor handling.
"""

import re
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
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
    Returns a FIXED number of frames (16) for all videos.
    """
    
    # Fixed number of frames for all samples
    N_FRAMES = 16
    
    def __init__(
        self,
        csv_path: Path,
        max_samples: int = 9999999,
        aug_type: Optional[str] = None,
        aug_quality: Optional[int] = None
    ):
        super(D3Dataset, self).__init__()
        
        # Load CSV
        self.df = pd.read_csv(csv_path).head(max_samples)
        self.trans = get_preprocessing_pipeline(aug_type, aug_quality)
        
        # Display stats
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        if 'label' in self.df.columns:
            real_count = len(self.df[self.df['label'] == 0])
            fake_count = len(self.df[self.df['label'] == 1])
            print(f"  Real: {real_count}, Fake: {fake_count}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get video frames and label.
        
        Returns:
            frames: Tensor of shape (16, 3, 224, 224) - FIXED SIZE
            label: Integer label (0=real, 1=fake)
        """
        row = self.df.iloc[idx]
        label = int(row['label'])
        frame_dir = Path(row['frame_dir'])
        
        # Read video frames from directory
        frames = self._read_video_frames(frame_dir)
        
        return frames, label
    
    def _read_video_frames(self, frame_dir: Path) -> torch.Tensor:
        """
        Read and preprocess frames from directory.
        Returns a FIXED number of frames (16) for all videos.
        """
        # Get sorted frame paths
        frame_files = sorted(
            [p for p in frame_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']],
            key=lambda x: get_number_from_filename(x.stem)
        )
        
        total_frames = len(frame_files)
        if total_frames < 8:
            raise ValueError(f"Not enough frames in {frame_dir}: {total_frames}")
        
        # Use FIXED number of frames (16)
        n_frames = self.N_FRAMES
        
        # Sample indices
        if total_frames < n_frames:
            # Pad with replacement
            indices = np.random.choice(total_frames, n_frames, replace=True)
        else:
            # Sample evenly
            indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        
        frames = []
        for idx in indices:
            frame_path = frame_files[idx]
            image = cv2.imread(str(frame_path))
            if image is None:
                # If image fails to load, use a blank image
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Preprocess
            image = crop_center_by_percentage(image, 0.1)
            augmented = self.trans(image=image)
            image = augmented["image"]
            
            # Add channel dimension
            frames.append(image.transpose(2, 0, 1))
        
        # Stack into tensor: (16, 3, 224, 224)
        # Use torch.from_numpy to ensure proper gradient flow
        return torch.from_numpy(np.stack(frames)).float()