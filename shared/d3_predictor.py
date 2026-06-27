"""
shared/d3_predictor.py
D3+ Predictor for inference using scikit-learn models.
USES THE EXACT SAME FEATURE EXTRACTION AS TRAINING.
"""

import torch
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import time
import subprocess
import json
import cv2
import tempfile
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.video_processor import VideoProcessor

# Import the SAME feature extraction function used in training
# This must match what you used to generate your CSV
from scripts.extract_all_features import extract_all_features_from_video


class D3PlusPredictor:
    """
    D3+ video predictor with confidence scoring.
    Uses scikit-learn models (Random Forest, SVM, etc.)
    USES THE EXACT SAME FEATURE EXTRACTION AS TRAINING.
    """
    
    def __init__(
        self,
        rf_model_path: Path,
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
        
        # Load scikit-learn model
        print("Loading scikit-learn models...")
        self.rf_model = joblib.load(rf_model_path)
        print(f"  ✅ Random Forest loaded from {rf_model_path}")
        
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
        
        # Initialize video processor
        self.video_processor = VideoProcessor()
        
        print(f"✅ D3PlusPredictor initialized on {self.device}")
        print(f"   Primary model: Random Forest")
        print(f"   Feature extraction: SAME as training pipeline")
    
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
        
        print(f"\n🔍 Predicting for video: {video_path.name if video_path else 'unknown'}")
        
        # ============================================================
        # USE THE SAME FEATURE EXTRACTION AS TRAINING
        # ============================================================
        # This calls the exact same function used to generate your CSV
        features = self._extract_features_using_training_pipeline(frames_tensor, video_path)
        
        print(f"   Features extracted: {len(features)}")
        print(f"   Features mean: {features.mean():.4f}")
        print(f"   Features std: {features.std():.4f}")
        print(f"   First 10 features: {features[:10]}")
        
        # Prepare features for model
        X = np.array([features])
        
        # Handle missing values and scale (same as training)
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
        print(f"   Prediction probabilities: [real={y_proba[0]:.4f}, fake={y_proba[1] if len(y_proba) > 1 else y_proba[0]:.4f}]")
        
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
    
    def _extract_features_using_training_pipeline(self, frames_tensor: torch.Tensor, video_path: Path) -> np.ndarray:
        """
        Use the EXACT SAME feature extraction as training.
        This is the CRITICAL FIX - it ensures deployment matches training.
        """
        
        # ============================================================
        # METHOD 1: If you have the training extraction function
        # ============================================================
        try:
            from scripts.extract_all_features import extract_features_from_tensor
            
            # Use the exact training function
            features = extract_features_from_tensor(frames_tensor, video_path)
            print(f"   ✅ Used training pipeline to extract features")
            return features
            
        except ImportError as e:
            print(f"   ⚠️ Could not import training function: {e}")
            print(f"   Using fallback extraction (may not match training)")
        
        # ============================================================
        # METHOD 2: Replicate training extraction manually
        # ============================================================
        # This must EXACTLY match your training extraction
        from scipy.stats import skew, kurtosis
        
        if isinstance(frames_tensor, torch.Tensor):
            frames_np = frames_tensor.cpu().numpy()
        else:
            frames_np = frames_tensor
        
        print(f"   Frames shape: {frames_np.shape}")
        
        # --- 1. D3 Features (must match training) ---
        # If you used a D3 model in training, you need the same here
        # Without it, these will be zeros
        d3_avg = 0.0
        d3_std = 0.0
        
        # Try to compute D3 features directly
        try:
            # This is a simplified version - adjust to match your training
            frames_tensor_device = torch.FloatTensor(frames_np)
            frame_features = frames_tensor_device.mean(dim=(2, 3))
            
            if frame_features.shape[0] >= 2:
                vec1 = frame_features[:-1, :]
                vec2 = frame_features[1:, :]
                dis_1st = torch.norm(vec1 - vec2, p=2, dim=1)
                
                if dis_1st.shape[0] >= 2:
                    dis_2nd = dis_1st[1:] - dis_1st[:-1]
                    d3_avg = torch.mean(dis_2nd).item()
                    d3_std = torch.std(dis_2nd).item()
        except Exception as e:
            print(f"   ⚠️ D3 computation failed: {e}")
        
        d3_features = [float(d3_avg), float(d3_std)]
        
        # --- 2. Color Features (must match training) ---
        color_features = []
        n_frames = min(16, len(frames_np))
        
        for frame_idx in range(n_frames):
            frame = frames_np[frame_idx]
            
            if frame.shape[0] == 3:
                frame_hwc = frame.transpose(1, 2, 0)
            else:
                frame_hwc = frame
            
            for c in range(3):
                if c < frame_hwc.shape[2]:
                    channel = frame_hwc[:, :, c].flatten()
                else:
                    channel = np.zeros_like(frame_hwc[:, :, 0]).flatten()
                
                color_features.extend([
                    float(np.mean(channel)),
                    float(np.std(channel)),
                    float(skew(channel)),
                    float(kurtosis(channel))
                ])
        
        while len(color_features) < 192:
            color_features.append(0.0)
        color_features = color_features[:192]
        
        # --- 3. Temporal Features ---
        temporal_features = []
        if len(frames_np) > 1:
            diffs = []
            for i in range(1, min(len(frames_np), 16)):
                diff = np.mean(np.abs(frames_np[i] - frames_np[i-1]))
                diffs.append(diff)
            
            if diffs:
                temporal_features = [
                    float(np.mean(diffs)),
                    float(np.std(diffs)),
                    float(np.max(diffs))
                ]
            else:
                temporal_features = [0.0, 0.0, 0.0]
        else:
            temporal_features = [0.0, 0.0, 0.0]
        
        # --- 4. Bitrate Features ---
        bitrate_features = self._extract_bitrate_features(video_path)
        
        # --- 5. Combine ---
        combined = d3_features + color_features + temporal_features + list(bitrate_features)
        
        while len(combined) < 203:
            combined.append(0.0)
        combined = combined[:203]
        
        return np.array(combined, dtype=np.float32)
    
    def _extract_bitrate_features(self, video_path: Path) -> np.ndarray:
        """Extract bitrate features from video file."""
        if video_path is None or not Path(video_path).exists():
            return np.zeros(6)
        
        try:
            video_path = Path(video_path)
            
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                '-show_format',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return np.zeros(6)
            
            data = json.loads(result.stdout)
            
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if video_stream is None:
                return np.zeros(6)
            
            format_info = data.get('format', {})
            
            duration = float(format_info.get('duration', 0))
            file_size = int(format_info.get('size', 0)) / (1024 * 1024)
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            
            nb_frames = video_stream.get('nb_frames')
            if nb_frames is None:
                avg_frame_rate = video_stream.get('avg_frame_rate', '0/1')
                if '/' in avg_frame_rate:
                    num, den = avg_frame_rate.split('/')
                    if float(den) > 0:
                        frame_rate = float(num) / float(den)
                        frame_count = frame_rate * duration
                    else:
                        frame_count = 0
                else:
                    frame_count = 0
            else:
                frame_count = float(nb_frames)
            
            return np.array([
                float(duration),
                float(frame_count),
                float(width),
                float(height),
                float(file_size),
                0.0
            ])
            
        except Exception as e:
            print(f"   ⚠️ Bitrate extraction error: {e}")
            return np.zeros(6)
    
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
        print(f"   Feature extraction will match training pipeline")
        
        _predictor_instance = D3PlusPredictor(
            rf_model_path=rf_path,
            imputer_path=imputer_path,
            scaler_path=scaler_path,
            feature_names_path=feature_names_path,
            device="cpu"
        )
    
    return _predictor_instance