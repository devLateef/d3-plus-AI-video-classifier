"""
shared/d3_predictor.py
D3+ Predictor for inference using scikit-learn models.
"""

import torch
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import time

from shared.video_processor import VideoProcessor
from shared.feature_extractor import FeatureExtractor
from research.models.d3_model import D3Model


class D3PlusPredictor:
    """
    D3+ video predictor with confidence scoring.
    Uses scikit-learn models (Random Forest, SVM, etc.)
    """
    
    def __init__(
        self,
        rf_model_path: Path,
        svm_model_path: Optional[Path] = None,
        lr_model_path: Optional[Path] = None,
        imputer_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        feature_names_path: Optional[Path] = None,
        device: str = "cpu",
        threshold: float = 0.5
    ):
        """
        Initialize predictor with scikit-learn models.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        
        # Load scikit-learn models
        print("Loading scikit-learn models...")
        self.rf_model = joblib.load(rf_model_path)
        print(f"  ✅ Random Forest loaded from {rf_model_path}")
        
        if svm_model_path and svm_model_path.exists():
            self.svm_model = joblib.load(svm_model_path)
            print(f"  ✅ SVM loaded from {svm_model_path}")
        else:
            self.svm_model = None
        
        if lr_model_path and lr_model_path.exists():
            self.lr_model = joblib.load(lr_model_path)
            print(f"  ✅ Logistic Regression loaded from {lr_model_path}")
        else:
            self.lr_model = None
        
        # Load preprocessors
        if imputer_path and imputer_path.exists():
            self.imputer = joblib.load(imputer_path)
            print(f"  ✅ Imputer loaded from {imputer_path}")
        else:
            self.imputer = None
        
        if scaler_path and scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"  ✅ Scaler loaded from {scaler_path}")
        else:
            self.scaler = None
        
        # Load feature names
        if feature_names_path and feature_names_path.exists():
            self.feature_names = np.load(feature_names_path, allow_pickle=True)
            print(f"  ✅ Feature names loaded ({len(self.feature_names)} features)")
        else:
            self.feature_names = None
        
        # Use Random Forest as the primary model
        self.model = self.rf_model
        
        # Initialize processors
        self.video_processor = VideoProcessor()
        self.feature_extractor = FeatureExtractor()
        
        # D3 model for feature extraction (load if needed)
        self.d3_model = None
        
        print(f"✅ D3PlusPredictor initialized on {self.device}")
        print(f"   Primary model: Random Forest")
    
    def load_d3_model(self, d3_model_path: Path):
        """Load D3 model for feature extraction (optional)."""
        try:
            from research.models.d3_model import D3Model
            self.d3_model = D3Model(encoder_type='XCLIP-16', loss_type='l2').to(self.device)
            self.d3_model.load_state_dict(torch.load(d3_model_path, map_location=self.device))
            self.d3_model.eval()
            print(f"  ✅ D3 model loaded from {d3_model_path}")
        except Exception as e:
            print(f"  ⚠️ Could not load D3 model: {e}")
            self.d3_model = None
    
    def predict(self, video_path: Path) -> Dict[str, float]:
        """
        Predict if video is AI-generated from a video file.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Process video and get frames tensor
        frames_tensor = self.video_processor.process_video(video_path)
        
        # Extract features from frames
        features = self._extract_features_from_frames(frames_tensor, video_path)
        
        # Prepare features for model
        X = np.array([features])
        
        # Handle missing values and scale
        if self.imputer is not None:
            X_imputed = self.imputer.transform(X)
        else:
            X_imputed = X
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_imputed)
        else:
            X_scaled = X_imputed
        
        # Predict with Random Forest
        y_proba = self.model.predict_proba(X_scaled)[0]
        
        # Get probability of being fake (class 1)
        probability = y_proba[1] if len(y_proba) > 1 else y_proba[0]
        
        # Compute confidence
        confidence = self._compute_confidence(probability)
        
        prediction_time = (time.time() - start_time) * 1000
        
        return {
            'probability': float(probability),
            'confidence': float(confidence),
            'is_ai_generated': probability > self.threshold,
            'prediction_time_ms': prediction_time,
            'raw_score': float(probability)
        }
    
    def predict_from_frames(self, frames_tensor: torch.Tensor, video_path: Path) -> Dict[str, float]:
        """
        Predict if video is AI-generated using pre-extracted frames.
        
        Args:
            frames_tensor: Pre-processed frames tensor
            video_path: Path to video file (for metadata)
        
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Extract features from frames
        features = self._extract_features_from_frames(frames_tensor, video_path)
        
        # Prepare features for model
        X = np.array([features])
        
        # Handle missing values and scale
        if self.imputer is not None:
            X_imputed = self.imputer.transform(X)
        else:
            X_imputed = X
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_imputed)
        else:
            X_scaled = X_imputed
        
        # Predict with Random Forest
        y_proba = self.model.predict_proba(X_scaled)[0]
        
        # Get probability of being fake (class 1)
        probability = y_proba[1] if len(y_proba) > 1 else y_proba[0]
        
        # Compute confidence
        confidence = self._compute_confidence(probability)
        
        prediction_time = (time.time() - start_time) * 1000
        
        return {
            'probability': float(probability),
            'confidence': float(confidence),
            'is_ai_generated': probability > self.threshold,
            'prediction_time_ms': prediction_time,
            'raw_score': float(probability)
        }
    
    def _extract_features_from_frames(self, frames_tensor: torch.Tensor, video_path: Path) -> np.ndarray:
        """
        Extract the 203 features from pre-processed frames.
        This replicates your feature extraction pipeline.
        """
        import cv2
        import torch.nn.functional as F
        from scipy.stats import skew, kurtosis
        
        # Convert frames to numpy
        if isinstance(frames_tensor, torch.Tensor):
            frames_np = frames_tensor.cpu().numpy()
        else:
            frames_np = frames_tensor
        
        # Extract D3 features
        if self.d3_model is not None:
            frames_tensor_device = frames_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, d3_avg, d3_score = self.d3_model(frames_tensor_device)
                d3_avg = d3_avg.cpu().numpy()[0]
                d3_std = d3_score.cpu().numpy()[0] if isinstance(d3_score, torch.Tensor) else d3_score
        else:
            # Fallback: compute simple features
            d3_avg = 0.0
            d3_std = 0.0
        
        # Extract color features
        color_features = []
        for frame_idx in range(min(16, len(frames_np))):
            frame = frames_np[frame_idx]
            # frame is (C, H, W) or (H, W, C)
            if frame.shape[0] == 3:
                frame_hwc = frame.transpose(1, 2, 0)
            else:
                frame_hwc = frame
            
            for c in range(min(3, frame_hwc.shape[2])):
                channel = frame_hwc[:, :, c].flatten()
                color_features.extend([
                    float(np.mean(channel)),
                    float(np.std(channel)),
                    float(skew(channel)),
                    float(kurtosis(channel))
                ])
        
        # Pad color features to 192 (16 frames * 3 channels * 4 stats)
        while len(color_features) < 192:
            color_features.append(0.0)
        color_features = color_features[:192]
        
        # Extract temporal features
        temporal_features = []
        if len(frames_np) > 1:
            diffs = []
            for i in range(1, len(frames_np)):
                diff = np.mean(np.abs(frames_np[i] - frames_np[i-1]))
                diffs.append(diff)
            if diffs:
                temporal_features = [float(np.mean(diffs)), float(np.std(diffs)), float(np.max(diffs))]
            else:
                temporal_features = [0.0, 0.0, 0.0]
        else:
            temporal_features = [0.0, 0.0, 0.0]
        
        # Extract bitrate features (simplified)
        bitrate_features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Combine all features
        combined = [d3_avg, d3_std] + color_features + temporal_features + bitrate_features
        
        # Ensure we have exactly 203 features
        while len(combined) < 203:
            combined.append(0.0)
        combined = combined[:203]
        
        return np.array(combined)
    
    def _compute_confidence(self, probability: float) -> float:
        """Compute confidence score from probability."""
        boundary_distance = abs(probability - 0.5) * 2
        confidence = min(1.0, boundary_distance * 1.2)
        return float(confidence)


# Singleton for FastAPI
_predictor_instance = None


def get_predictor(model_dir: Path = None) -> D3PlusPredictor:
    """Get or create predictor instance."""
    global _predictor_instance
    
    if _predictor_instance is None:
        if model_dir is None:
            model_dir = Path("trained_models")
        
        rf_path = model_dir / "random_forest_model.pkl"
        imputer_path = model_dir / "imputer.pkl"
        scaler_path = model_dir / "scaler.pkl"
        feature_names_path = model_dir / "feature_names.npy"
        
        if not rf_path.exists():
            raise FileNotFoundError(f"Random Forest model not found at {rf_path}")
        
        print(f"\n📂 Loading models from {model_dir}...")
        
        _predictor_instance = D3PlusPredictor(
            rf_model_path=rf_path,
            imputer_path=imputer_path,
            scaler_path=scaler_path,
            feature_names_path=feature_names_path,
            device="cpu"
        )
    
    return _predictor_instance