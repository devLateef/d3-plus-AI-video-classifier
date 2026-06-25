"""
Dependency injection for FastAPI.
"""

from fastapi import Request
from shared.d3_predictor import get_predictor
from shared.video_processor import VideoProcessor
from api.app.config import settings


def get_video_processor() -> VideoProcessor:
    """Get video processor instance."""
    return VideoProcessor(
        temp_dir=settings.temp_dir,
        frame_rate=8,
        duration=3
    )


def get_predictor_dependency() -> any:
    """Get predictor instance."""
    return get_predictor(settings.model_path)