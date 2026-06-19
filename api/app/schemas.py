"""
api/app/schemas.py
Updated schemas with report structures.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class FeatureBreakdown(BaseModel):
    """Detailed feature breakdown."""
    d3_score: float = Field(..., ge=0, le=1)
    color_features: float = Field(..., ge=0, le=1)
    temporal_features: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    """Response model for video prediction."""
    video_id: str
    is_ai_generated: bool
    confidence_score: float = Field(..., ge=0, le=1)
    probability: float = Field(..., ge=0, le=1)
    prediction_time_ms: float
    total_time_ms: Optional[float] = None
    report: Optional[Dict[str, Any]] = None
    status: str = "success"
    message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class DetailedReportResponse(BaseModel):
    """Detailed report response."""
    video_id: str
    timestamp: str
    prediction: Dict[str, Any]
    metadata: Dict[str, Any]
    feature_breakdown: FeatureBreakdown
    interpretation: Dict[str, str]
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Response for batch processing."""
    results: List[PredictionResponse]
    total_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool
    gpu_available: bool


class ErrorResponse(BaseModel):
    """Error response model."""
    status: str = "error"
    message: str
    error_code: Optional[str] = None