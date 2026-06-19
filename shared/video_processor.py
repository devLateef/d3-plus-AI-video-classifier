"""
Video processing utilities for D3+.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import numpy as np
import torch
import shutil


class VideoProcessor:
    """
    Process videos for D3+ inference.
    Handles frame extraction and preprocessing.
    """
    
    def __init__(
        self,
        temp_dir: Optional[Path] = None,
        frame_rate: int = 8,
        duration: int = 3,
        target_size: Tuple[int, int] = (224, 224),
        crop_percentage: float = 0.1
    ):
        """
        Initialize video processor.
        
        Args:
            temp_dir: Directory for temporary files
            frame_rate: Frames per second to extract
            duration: Duration to extract in seconds
            target_size: Target image size (height, width)
            crop_percentage: Percentage to crop from edges
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "d3_plus"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.frame_rate = frame_rate
        self.duration = duration
        self.target_size = target_size
        self.crop_percentage = crop_percentage
    
    def process_video(self, video_path: Path) -> torch.Tensor:
        """
        Process video and return frames tensor.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Frames tensor of shape (frames, channels, height, width)
        """
        # Create unique temp directory
        video_id = f"vid_{os.urandom(4).hex()}"
        video_temp_dir = self.temp_dir / video_id
        video_temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract frames
            frame_dir = self._extract_frames(video_path, video_temp_dir)
            
            # Load and process frames
            frames = self._load_frames(frame_dir)
            
            return frames
            
        finally:
            # Clean up
            shutil.rmtree(video_temp_dir, ignore_errors=True)
    
    def _extract_frames(self, video_path: Path, output_dir: Path) -> Path:
        """Extract frames from video using ffmpeg."""
        frame_dir = output_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        output_pattern = str(frame_dir / "%d.jpg")
        
        cmd = [
            "ffmpeg",
            "-loglevel", "quiet",
            "-i", str(video_path),
            "-vf", f"fps={self.frame_rate}",
            "-t", str(self.duration),
            output_pattern
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")
        
        return frame_dir
    
    def _load_frames(self, frame_dir: Path) -> torch.Tensor:
        """Load and preprocess frames."""
        frame_files = sorted(frame_dir.glob("*.jpg"))
        
        if not frame_files:
            raise RuntimeError("No frames extracted from video")
        
        frames = []
        for frame_file in frame_files:
            img = cv2.imread(str(frame_file))
            if img is None:
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Crop center
            img = self._crop_center(img)
            
            # Resize
            img = cv2.resize(img, self.target_size)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
            # Convert to CHW format
            img = img.transpose(2, 0, 1)
            frames.append(img)
        
        if not frames:
            raise RuntimeError("No valid frames loaded")
        
        # Stack into tensor
        return torch.tensor(np.stack(frames), dtype=torch.float32)
    
    def _crop_center(self, image: np.ndarray) -> np.ndarray:
        """Crop center of image by percentage."""
        height, width = image.shape[:2]
        
        if width > height:
            crop_x = int(width * self.crop_percentage)
            return image[:, crop_x:width - crop_x]
        else:
            crop_y = int(height * self.crop_percentage)
            return image[crop_y:height - crop_y, :]
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)