"""
shared/d3_predictor.py
D3+ Predictor for inference using scikit-learn models.
"""

import torch
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import time
import pandas as pd
import cv2
import tempfile
import subprocess

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
        device: str = "cuda",
        threshold: float = 0.5
    ):
        """
        Initialize predictor with scikit-learn models.
        
        Args:
            rf_model_path: Path to Random Forest model (.pkl)
            svm_model_path: Path to SVM model (.pkl) - optional
            lr_model_path: Path to Logistic Regression model (.pkl) - optional
            imputer_path: Path to imputer (.pkl)
            scaler_path: Path to scaler (.pkl)
            feature_names_path: Path to feature names (.npy)
            device: Device to use
            threshold: Classification threshold
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
        
        # D3 model for feature extraction (needs to be loaded separately)
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
        Predict if video is AI-generated.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Process video and extract features
        features_df = self._extract_features(video_path)
        
        # Prepare features for model
        X = features_df.drop('video_name', axis=1) if 'video_name' in features_df.columns else features_df
        
        # Handle missing values and scale
        if self.imputer is not None:
            X_imputed = self.imputer.transform(X)
        else:
            X_imputed = X
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_imputed)
        else:
            X_scaled = X_imputed
        
        # Predict with Random Forest (primary model)
        y_pred = self.model.predict(X_scaled)[0]
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
    
    def _extract_features(self, video_path: Path) -> pd.DataFrame:
        """
        Extract 203 features from video.
        This should match your existing feature extraction pipeline.
        """
        # This is a simplified version - you should use your existing feature extraction
        from scripts.extract_all_features import extract_all_features_from_video
        
        # Extract features
        features = extract_all_features_from_video(video_path)
        
        # Create DataFrame with feature names
        if self.feature_names is not None:
            df = pd.DataFrame([features], columns=self.feature_names)
        else:
            df = pd.DataFrame([features])
        
        return df
    
    def _compute_confidence(self, probability: float) -> float:
        """Compute confidence score from probability."""
        # Distance from decision boundary (0.5)
        boundary_distance = abs(probability - 0.5) * 2
        confidence = min(1.0, boundary_distance * 1.2)
        return float(confidence)
    
    def get_confidence_breakdown(self, video_path: Path) -> Dict[str, float]:
        """Get detailed confidence breakdown."""
        result = self.predict(video_path)
        return {
            'overall_confidence': result['confidence'],
            'prediction_probability': result['probability'],
            'prediction_time_ms': result['prediction_time_ms']
        }


# Singleton for FastAPI
_predictor_instance = None


def get_predictor(model_dir: Path = None) -> D3PlusPredictor:
    """Get or create predictor instance."""
    global _predictor_instance
    
    if _predictor_instance is None:
        if model_dir is None:
            model_dir = Path("trained_models")
        
        # Determine paths
        rf_path = model_dir / "random_forest_model.pkl"
        imputer_path = model_dir / "imputer.pkl"
        scaler_path = model_dir / "scaler.pkl"
        feature_names_path = model_dir / "feature_names.npy"
        
        # Check if Random Forest model exists
        if not rf_path.exists():
            raise FileNotFoundError(f"Random Forest model not found at {rf_path}")
        
        print(f"\n📂 Loading models from {model_dir}...")
        
        _predictor_instance = D3PlusPredictor(
            rf_model_path=rf_path,
            imputer_path=imputer_path,
            scaler_path=scaler_path,
            feature_names_path=feature_names_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    return _predictor_instance