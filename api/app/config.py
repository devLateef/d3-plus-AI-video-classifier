"""
api/app/config.py
Updated with reports directory.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Model settings
    model_path: Path = Path("trained_models/d3_plus_model.pth")
    encoder_type: str = "XCLIP-16"
    loss_type: str = "l2"
    
    # API settings
    api_title: str = "D3+ AI Video Detector"
    api_version: str = "1.0.0"
    api_description: str = "Detect AI-generated videos with confidence scores"
    
    # File settings
    max_file_size_mb: int = 500
    allowed_extensions: list = ['.mp4', '.avi', '.mov', '.mkv']
    
    # CORS
    allowed_origins: list = ["*"]
    
    # Directories
    temp_dir: Path = Path("tmp")
    reports_dir: Path = Path("reports")
    
    class Config:
        env_file = ".env"


settings = Settings()