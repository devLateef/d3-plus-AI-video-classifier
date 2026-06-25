"""
api/app/dependencies.py
Dependency injection for FastAPI.
"""

import joblib
import numpy as np
from pathlib import Path
from fastapi import Request
import torch

from api.app.config import settings
from shared.d3_predictor import get_predictor
from shared.video_processor import VideoProcessor


def get_video_processor() -> VideoProcessor:
    """Get video processor instance."""
    return VideoProcessor(
        temp_dir=settings.temp_dir,
        frame_rate=8,
        duration=3
    )


def get_predictor_dependency() -> D3PlusPredictor:
    """Get predictor instance."""
    return get_predictor(settings.model_path.parent)  # Pass the directory


# Singleton instances
_predictor_instance = None
_video_processor_instance = None


def get_predictor_singleton() -> D3PlusPredictor:
    """Get or create predictor singleton."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = get_predictor(settings.model_path.parent)
    return _predictor_instance


def get_video_processor_singleton() -> VideoProcessor:
    """Get or create video processor singleton."""
    global _video_processor_instance
    if _video_processor_instance is None:
        _video_processor_instance = get_video_processor()
    return _video_processor_instance