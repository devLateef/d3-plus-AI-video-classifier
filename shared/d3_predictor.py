"""
D3+ Predictor for inference.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import time

from shared.video_processor import VideoProcessor
from shared.feature_extractor import FeatureExtractor
from research.models.d3_model import D3Model


class D3Predictor:
    """
    D3+ video predictor with confidence scoring.
    """
    
    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        threshold: float = 0.5
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model
            device: Device to use ('cuda' or 'cpu')
            threshold: Classification threshold
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        
        # Load model (auto-detect config from weights)
        self.model = D3Model(encoder_type='XCLIP-16', loss_type='l2')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize processors
        self.video_processor = VideoProcessor()
        self.feature_extractor = FeatureExtractor()
        
        print(f"✅ D3Predictor initialized on {self.device}")
    
    def predict(self, video_path: Path) -> Dict[str, float]:
        """
        Predict if video is AI-generated.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Process video
        frames = self.video_processor.process_video(video_path)
        frames = frames.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Run D3 inference
        with torch.no_grad():
            _, _, d3_score = self.model(frames)
            d3_score = d3_score.cpu().numpy().flatten()[0]
        
        # Extract additional features
        frames_np = frames[0].cpu().numpy()
        color_features = self.feature_extractor.extract_color_features(frames_np)
        temporal_features = self.feature_extractor.extract_temporal_features(frames_np)
        bitrate_features = self.feature_extractor.extract_bitrate_features(video_path)
        
        # Combine features for confidence computation
        combined_score = self._compute_combined_score(d3_score, color_features, temporal_features)
        
        # Compute confidence
        confidence = self._compute_confidence(d3_score, combined_score)
        
        prediction_time = (time.time() - start_time) * 1000
        
        return {
            'probability': float(combined_score),
            'confidence': float(confidence),
            'raw_score': float(d3_score),
            'prediction_time_ms': prediction_time,
            'is_ai_generated': combined_score > self.threshold
        }
    
    def _compute_combined_score(self, d3_score: float, color_features: np.ndarray, 
                                 temporal_features: np.ndarray) -> float:
        """Combine D3 score with additional features."""
        # Simple weighted combination
        # In practice, you'd train a classifier on these features
        d3_weight = 0.7
        color_weight = 0.15
        temporal_weight = 0.15
        
        # Normalize color and temporal features to [0, 1]
        color_norm = np.mean(color_features)  # Simplified
        temporal_norm = np.mean(temporal_features)  # Simplified
        
        combined = (d3_weight * d3_score + 
                   color_weight * color_norm + 
                   temporal_weight * temporal_norm)
        
        return np.clip(combined, 0, 1)
    
    def _compute_confidence(self, d3_score: float, combined_score: float) -> float:
        """Compute confidence score."""
        # Distance from decision boundary
        boundary_distance = abs(combined_score - 0.5) * 2
        
        # Scale to [0, 1]
        confidence = min(1.0, boundary_distance * 1.2)
        
        return float(confidence)
    
    def get_confidence_breakdown(self, video_path: Path) -> Dict[str, float]:
        """Get detailed confidence breakdown."""
        result = self.predict(video_path)
        
        return {
            'overall_confidence': result['confidence'],
            'prediction_probability': result['probability'],
            'raw_d3_score': result['raw_score'],
            'prediction_time_ms': result['prediction_time_ms']
        }


# Singleton for FastAPI
_predictor_instance = None


def get_predictor(model_path: Path = None) -> D3Predictor:
    """Get or create predictor instance."""
    global _predictor_instance
    
    if _predictor_instance is None:
        if model_path is None:
            model_path = Path("trained_models/d3_plus_model.pth")
        _predictor_instance = D3Predictor(model_path)
    
    return _predictor_instance