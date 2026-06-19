"""
api/app/main.py
FastAPI application with report generation.
"""

import os
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
import tempfile

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

from api.app.config import settings
from api.app.schemas import (
    PredictionResponse, HealthResponse, ErrorResponse,
    DetailedReportResponse, FeatureBreakdown
)
from api.app.dependencies import get_video_processor, get_predictor_dependency
from shared.video_processor import VideoProcessor

# Create directories
settings.temp_dir.mkdir(parents=True, exist_ok=True)
settings.reports_dir.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
video_processor = get_video_processor()
predictor = get_predictor_dependency()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        model_loaded=True,
        gpu_available=torch.cuda.is_available()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Predict if a video is AI-generated.
    """
    start_time = time.time()
    
    # Validate file type
    if not any(file.filename.lower().endswith(ext) for ext in settings.allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(settings.allowed_extensions)}"
        )
    
    # Validate file size
    file_size = await file.size()
    if file_size > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.max_file_size_mb}MB"
        )
    
    # Save uploaded file
    video_id = f"vid_{int(time.time())}_{file.filename[:20]}"
    video_path = settings.temp_dir / f"{video_id}.mp4"
    
    try:
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")
    
    try:
        # Run prediction
        result = predictor.predict(video_path)
        
        # Generate detailed report
        report = generate_detailed_report(video_path, result, video_id)
        
        # Save report
        report_path = settings.reports_dir / f"{video_id}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Clean up
        background_tasks.add_task(lambda: os.remove(video_path))
        background_tasks.add_task(lambda: os.remove(report_path))  # Optional
        
        total_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            video_id=video_id,
            is_ai_generated=result['is_ai_generated'],
            confidence_score=result['confidence'],
            probability=result['probability'],
            prediction_time_ms=result['prediction_time_ms'],
            total_time_ms=total_time,
            report=report,
            status="success"
        )
        
    except Exception as e:
        background_tasks.add_task(lambda: os.remove(video_path))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report/{video_id}")
async def get_report(video_id: str):
    """Get detailed report for a video."""
    report_path = settings.reports_dir / f"{video_id}_report.json"
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    return report


@app.get("/report/download/{video_id}")
async def download_report(video_id: str):
    """Download report as JSON."""
    report_path = settings.reports_dir / f"{video_id}_report.json"
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        report_path,
        media_type="application/json",
        filename=f"{video_id}_report.json"
    )


def generate_detailed_report(video_path: Path, result: dict, video_id: str) -> dict:
    """
    Generate detailed report for a prediction.
    """
    # Get video metadata
    try:
        import ffmpeg
        probe = ffmpeg.probe(str(video_path))
        format_info = probe.get('format', {})
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        
        metadata = {
            'duration': float(format_info.get('duration', 0)),
            'size_bytes': int(format_info.get('size', 0)),
            'width': int(video_info.get('width', 0)),
            'height': int(video_info.get('height', 0)),
            'codec': video_info.get('codec_name', 'unknown'),
            'frame_rate': float(video_info.get('avg_frame_rate', '0/1').split('/')[0]) 
                          if '/' in video_info.get('avg_frame_rate', '') else 0
        }
    except:
        metadata = {
            'duration': 0,
            'size_bytes': 0,
            'width': 0,
            'height': 0,
            'codec': 'unknown',
            'frame_rate': 0
        }
    
    # Build report
    report = {
        'video_id': video_id,
        'timestamp': datetime.now().isoformat(),
        'prediction': {
            'is_ai_generated': result['is_ai_generated'],
            'probability': result['probability'],
            'confidence_score': result['confidence'],
            'raw_d3_score': result.get('raw_score', 0.5)
        },
        'metadata': metadata,
        'feature_breakdown': {
            'd3_score': result.get('raw_score', 0.5),
            'color_features': result.get('color_score', 0.5),
            'temporal_features': result.get('temporal_score', 0.5)
        },
        'interpretation': {
            'confidence_level': 'High' if result['confidence'] > 0.8 else 'Medium' if result['confidence'] > 0.6 else 'Low',
            'summary': _generate_summary(result)
        },
        'processing_time_ms': result.get('prediction_time_ms', 0)
    }
    
    return report


def _generate_summary(result: dict) -> str:
    """Generate human-readable summary."""
    if result['is_ai_generated']:
        if result['confidence'] > 0.8:
            return "With high confidence, this video appears to be AI-generated. The model detected strong artifacts consistent with synthetic generation."
        elif result['confidence'] > 0.6:
            return "This video shows signs of AI generation, but with moderate confidence. Additional analysis recommended."
        else:
            return "This video may be AI-generated, but confidence is low. Consider manual review."
    else:
        if result['confidence'] > 0.8:
            return "With high confidence, this video appears to be authentic. No significant AI generation artifacts detected."
        else:
            return "This video appears to be authentic, but with moderate confidence. Consider additional verification."


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            message=str(exc.detail),
            error_code=f"HTTP_{exc.status_code}"
        ).model_dump()
    )


if __name__ == "__main__":
    uvicorn.run(
        "api.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )