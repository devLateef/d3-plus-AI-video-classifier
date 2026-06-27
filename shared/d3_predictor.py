"""
shared/d3_predictor.py
D3+ Predictor for inference using scikit-learn models.
FIXED: Proper feature extraction matching training pipeline.
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
        d3_model_path: Optional[Path] = None,
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
        
        # Load D3 model for feature extraction
        self.d3_model = None
        if d3_model_path and d3_model_path.exists():
            self.load_d3_model(d3_model_path)
        else:
            print(f"  ⚠️ No D3 model provided. Some features will use fallback values.")
        
        print(f"✅ D3PlusPredictor initialized on {self.device}")
        print(f"   Primary model: Random Forest")
    
    def load_d3_model(self, d3_model_path: Path):
        """Load D3 model for feature extraction."""
        try:
            from research.models.d3_model import D3Model
            print(f"Loading D3 model from {d3_model_path}...")
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
        
        # DIAGNOSTIC LOGGING
        print(f"\n🔍 Predicting for video: {video_path.name if video_path else 'unknown'}")
        print(f"   Frames tensor shape: {frames_tensor.shape}")
        
        # Extract features from frames
        features = self._extract_features_from_frames(frames_tensor, video_path)
        
        # DIAGNOSTIC LOGGING
        print(f"   Features extracted: {len(features)}")
        print(f"   Features mean: {features.mean():.4f}")
        print(f"   Features std: {features.std():.4f}")
        print(f"   Features min: {features.min():.4f}")
        print(f"   Features max: {features.max():.4f}")
        print(f"   First 10 features: {features[:10]}")
        
        # Prepare features for model
        X = np.array([features])
        
        # Handle missing values and scale
        if self.imputer is not None:
            X_imputed = self.imputer.transform(X)
            print(f"   After imputation - mean: {X_imputed.mean():.4f}, std: {X_imputed.std():.4f}")
        else:
            X_imputed = X
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_imputed)
            print(f"   After scaling - mean: {X_scaled.mean():.4f}, std: {X_scaled.std():.4f}")
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
    
    def _extract_features_from_frames(self, frames_tensor: torch.Tensor, video_path: Path) -> np.ndarray:
        """
        Extract the 203 features from pre-processed frames.
        This MUST match your training feature extraction EXACTLY.
        """
        from scipy.stats import skew, kurtosis
        
        # Convert frames to numpy
        if isinstance(frames_tensor, torch.Tensor):
            frames_np = frames_tensor.cpu().numpy()
        else:
            frames_np = frames_tensor
        
        print(f"   Frames shape: {frames_np.shape}")
        
        # ============================================================
        # 1. D3 Features (2 features)
        # ============================================================
        if self.d3_model is not None:
            try:
                frames_tensor_device = frames_tensor.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, d3_avg, d3_score = self.d3_model(frames_tensor_device)
                    d3_avg = d3_avg.cpu().numpy()[0]
                    d3_std = d3_score.cpu().numpy()[0] if isinstance(d3_score, torch.Tensor) else d3_score
            except Exception as e:
                print(f"   ⚠️ D3 model inference failed: {e}")
                d3_avg = 0.0
                d3_std = 0.0
        else:
            print("   ⚠️ D3 model not loaded! Using fallback D3 features.")
            d3_avg = 0.0
            d3_std = 0.0
        
        d3_features = [float(d3_avg), float(d3_std)]
        print(f"   D3 features: {len(d3_features)}")
        
        # ============================================================
        # 2. Color Features (192 features: 16 frames * 3 channels * 4 stats)
        # ============================================================
        color_features = []
        
        # Process exactly 16 frames
        n_frames = min(16, len(frames_np))
        print(f"   Processing {n_frames} frames for color features")
        
        for frame_idx in range(n_frames):
            frame = frames_np[frame_idx]
            
            # Convert to (height, width, channels) for channel extraction
            if frame.shape[0] == 3:
                frame_hwc = frame.transpose(1, 2, 0)
            else:
                frame_hwc = frame
            
            # For each channel (R, G, B)
            for c in range(3):
                if c < frame_hwc.shape[2]:
                    channel = frame_hwc[:, :, c].flatten()
                else:
                    # If frame has fewer than 3 channels, pad with zeros
                    channel = np.zeros_like(frame_hwc[:, :, 0]).flatten()
                
                # Extract exactly the same statistics as training
                color_features.extend([
                    float(np.mean(channel)),
                    float(np.std(channel)),
                    float(skew(channel)),
                    float(kurtosis(channel))
                ])
        
        # Pad to exactly 192 features (16 frames * 3 channels * 4 stats)
        while len(color_features) < 192:
            color_features.append(0.0)
        color_features = color_features[:192]
        print(f"   Color features: {len(color_features)}")
        
        # ============================================================
        # 3. Temporal Features (3 features)
        # ============================================================
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
        
        print(f"   Temporal features: {len(temporal_features)}")
        
        # ============================================================
        # 4. Bitrate Features (6 features)
        # ============================================================
        bitrate_features = self._extract_bitrate_features(video_path)
        print(f"   Bitrate features: {len(bitrate_features)}")
        
        # ============================================================
        # 5. Combine ALL features
        # ============================================================
        combined = d3_features + color_features + temporal_features + list(bitrate_features)
        
        print(f"   Combined before padding: {len(combined)}")
        
        # Ensure exactly 203 features
        while len(combined) < 203:
            combined.append(0.0)
        combined = combined[:203]
        
        print(f"   Final features: {len(combined)}")
        
        # Convert to numpy array
        return np.array(combined, dtype=np.float32)
    
    def _extract_bitrate_features(self, video_path: Path) -> np.ndarray:
        """
        Extract bitrate features from video file.
        Matches training bitrate extraction.
        """
        print(f"   Extracting bitrate features for: {video_path}")
        
        # If video_path is None or doesn't exist, return zeros
        if video_path is None:
            print(f"   ⚠️ Video path is None")
            return np.zeros(6)
        
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"   ⚠️ Video path doesn't exist: {video_path}")
            return np.zeros(6)
        
        try:
            # Use ffprobe to get video metadata
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
                print(f"   ⚠️ ffprobe failed with return code {result.returncode}")
                return np.zeros(6)
            
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if video_stream is None:
                print(f"   ⚠️ No video stream found")
                return np.zeros(6)
            
            format_info = data.get('format', {})
            
            # Extract features (must match training)
            duration = float(format_info.get('duration', 0))
            
            # Get frame count
            nb_frames = video_stream.get('nb_frames')
            if nb_frames is None:
                # Try to calculate from duration and frame rate
                avg_frame_rate = video_stream.get('avg_frame_rate', '0/1')
                if '/' in avg_frame_rate:
                    num, den = avg_frame_rate.split('/')
                    if float(den) > 0:
                        frame_rate = float(num) / float(den)
                        duration_float = float(format_info.get('duration', 0))
                        frame_count = frame_rate * duration_float
                    else:
                        frame_count = 0
                else:
                    frame_count = 0
            else:
                frame_count = float(nb_frames)
            
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            file_size = int(format_info.get('size', 0)) / (1024 * 1024)  # MB
            is_gif = 0.0  # Not a GIF
            
            bitrate_features = np.array([
                float(duration),
                float(frame_count),
                float(width),
                float(height),
                float(file_size),
                float(is_gif)
            ])
            
            print(f"   Bitrate features: {bitrate_features}")
            return bitrate_features
            
        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON decode error: {e}")
            return np.zeros(6)
        except Exception as e:
            print(f"   ⚠️ Bitrate extraction error for {video_path}: {e}")
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
        d3_model_path = model_dir / "d3_plus_model.pth"
        
        if not rf_path.exists():
            raise FileNotFoundError(f"Random Forest model not found at {rf_path}")
        
        print(f"\n📂 Loading models from {model_dir}...")
        
        _predictor_instance = D3PlusPredictor(
            rf_model_path=rf_path,
            imputer_path=imputer_path,
            scaler_path=scaler_path,
            feature_names_path=feature_names_path,
            d3_model_path=d3_model_path if d3_model_path.exists() else None,
            device="cpu"
        )
    
    return _predictor_instance