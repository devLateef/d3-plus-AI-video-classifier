"""
Feature extraction utilities for D3+.
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class FeatureExtractor:
    """
    Extract additional features for D3+ detection.
    Includes color, temporal, and bitrate features.
    """
    
    def __init__(self, sampling_rate: int = 8):
        self.sampling_rate = sampling_rate
    
    def extract_color_features(self, frames: torch.Tensor) -> np.ndarray:
        """
        Extract RGB color space features.
        
        Args:
            frames: Frame tensor (frames, channels, height, width)
        
        Returns:
            Color features array
        """
        # Convert to numpy
        frames_np = frames.numpy() if torch.is_tensor(frames) else frames
        
        # For each frame, compute statistics
        features = []
        
        for frame in frames_np:
            # Frame is (C, H, W) -> (H, W, C) for OpenCV
            frame_cv = frame.transpose(1, 2, 0)
            
            # RGB channels
            for c, name in enumerate(['R', 'G', 'B']):
                channel = frame_cv[:, :, c].flatten()
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.percentile(channel, 50),  # median
                    np.min(channel),
                    np.max(channel)
                ])
            
            # HSV conversion
            hsv = cv2.cvtColor((frame_cv * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            for c, name in enumerate(['H', 'S', 'V']):
                channel = hsv[:, :, c].flatten().astype(np.float32) / 255.0
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.percentile(channel, 50)
                ])
        
        return np.array(features)
    
    def extract_temporal_features(self, frames: torch.Tensor) -> np.ndarray:
        """
        Extract temporal channel relationship features.
        
        Args:
            frames: Frame tensor (frames, channels, height, width)
        
        Returns:
            Temporal features array
        """
        if len(frames) < 2:
            return np.array([0.0, 0.0, 0.0])
        
        # Compute frame-to-frame differences
        features = []
        
        for i in range(len(frames) - 1):
            diff = frames[i] - frames[i+1]
            features.extend([
                torch.norm(diff).item(),
                torch.std(diff).item(),
                torch.mean(torch.abs(diff)).item()
            ])
        
        # Aggregate over all frames
        return np.array([
            np.mean(features[::3]),  # mean of first feature
            np.std(features[1::3]),   # std of second feature
            np.max(features[2::3])    # max of third feature
        ])
    
    def extract_bitrate_features(self, video_path) -> Dict[str, float]:
        """
        Extract bitrate and metadata features from video.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary of bitrate features
        """
        try:
            import ffmpeg
            
            probe = ffmpeg.probe(str(video_path))
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            format_info = probe.get('format', {})
            
            duration = float(format_info.get('duration', 0))
            size_bytes = int(format_info.get('size', 0))
            bitrate = int(format_info.get('bit_rate', 0))
            
            return {
                'duration': duration,
                'size_bytes': size_bytes,
                'bitrate': bitrate,
                'width': int(video_info.get('width', 0)),
                'height': int(video_info.get('height', 0)),
                'frame_rate': float(video_info.get('avg_frame_rate', '0/1').split('/')[0]) 
                              if '/' in video_info.get('avg_frame_rate', '') else 0
            }
        except Exception:
            return {
                'duration': 0.0,
                'size_bytes': 0,
                'bitrate': 0,
                'width': 0,
                'height': 0,
                'frame_rate': 0.0
            }