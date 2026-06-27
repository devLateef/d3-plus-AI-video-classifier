"""
shared/d3_predictor.py
D3+ Predictor - Properly handles XCLIP model input shapes.
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
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.video_processor import VideoProcessor


class D3PlusPredictor:
    """
    D3+ video predictor with proper D3 model handling.
    """
    
    def __init__(
        self,
        rf_model_path: Path,
        imputer_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        feature_names_path: Optional[Path] = None,
        device: str = "cpu",
        threshold: float = 0.5,
        use_d3: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.use_d3 = use_d3
        
        # Load Random Forest
        print("Loading Random Forest...")
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
        
        if feature_names_path and feature_names_path.exists():
            self.feature_names = np.load(feature_names_path, allow_pickle=True)
            print(f"  ✅ Feature names loaded ({len(self.feature_names)} features)")
        else:
            self.feature_names = None
        
        self.model = self.rf_model
        self.video_processor = VideoProcessor()
        
        # Load D3 model from Hugging Face if requested
        self.d3_model = None
        if use_d3:
            try:
                print("Loading D3 model from Hugging Face...")
                from research.models.d3_model import D3Model
                self.d3_model = D3Model(encoder_type='XCLIP-16', loss_type='l2').to(self.device)
                self.d3_model.eval()
                print("  ✅ D3 model loaded from Hugging Face")
            except Exception as e:
                print(f"  ⚠️ Could not load D3 model: {e}")
                self.d3_model = None
        
        print(f"✅ D3PlusPredictor initialized on {self.device}")
        print(f"   D3 model: {'Loaded' if self.d3_model else 'Not available'}")
    
    def predict_from_frames(self, frames_tensor: torch.Tensor, video_path: Path) -> Dict[str, float]:
        start_time = time.time()
        
        print(f"\n🔍 Predicting: {video_path.name if video_path else 'unknown'}")
        print(f"   Frames shape: {frames_tensor.shape}")
        
        features = self._extract_features_from_frames(frames_tensor, video_path)
        
        print(f"   Features: {len(features)}")
        print(f"   Mean: {features.mean():.4f}, Std: {features.std():.4f}")
        
        X = np.array([features])
        
        if self.imputer is not None:
            X_imputed = self.imputer.transform(X)
        else:
            X_imputed = X
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_imputed)
        else:
            X_scaled = X_imputed
        
        y_proba = self.model.predict_proba(X_scaled)[0]
        print(f"   Probabilities: [real={y_proba[0]:.4f}, fake={y_proba[1] if len(y_proba) > 1 else y_proba[0]:.4f}]")
        
        probability = y_proba[1] if len(y_proba) > 1 else y_proba[0]
        confidence = self._compute_confidence(probability)
        
        return {
            'probability': float(probability),
            'confidence': float(confidence),
            'is_ai_generated': probability > self.threshold,
            'prediction_time_ms': (time.time() - start_time) * 1000,
            'raw_score': float(probability)
        }
    
    def _extract_features_from_frames(self, frames_tensor: torch.Tensor, video_path: Path) -> np.ndarray:
        from scipy.stats import skew, kurtosis
        
        if isinstance(frames_tensor, torch.Tensor):
            frames_np = frames_tensor.cpu().numpy()
        else:
            frames_np = frames_tensor
        
        print(f"   Frames shape (numpy): {frames_np.shape}")
        
        # ============================================================
        # 1. D3 Features (if available)
        # ============================================================
        d3_avg = 0.0
        d3_std = 0.0
        
        if self.d3_model is not None:
            try:
                # The D3 model expects: (batch, frames, channels, height, width)
                # We need to ensure we have the right number of frames
                # XCLIP expects 16 frames
                n_frames = 16
                if frames_tensor.shape[0] < n_frames:
                    # Pad with zeros
                    padded = torch.zeros(n_frames, frames_tensor.shape[1], frames_tensor.shape[2], frames_tensor.shape[3])
                    padded[:frames_tensor.shape[0]] = frames_tensor
                    frames_for_d3 = padded
                else:
                    # Take first 16 frames
                    frames_for_d3 = frames_tensor[:n_frames]
                
                # Add batch dimension: (1, frames, channels, height, width)
                frames_for_d3 = frames_for_d3.unsqueeze(0).to(self.device)
                print(f"   D3 model input shape: {frames_for_d3.shape}")
                
                with torch.no_grad():
                    _, d3_avg, d3_std = self.d3_model(frames_for_d3)
                    d3_avg = d3_avg.cpu().numpy()[0]
                    d3_std = d3_std.cpu().numpy()[0]
                
                print(f"   D3 features: avg={d3_avg:.4f}, std={d3_std:.4f}")
                
            except Exception as e:
                print(f"   ⚠️ D3 model failed: {e}")
                # Use fallback
                d3_avg = 0.0
                d3_std = 0.0
        
        # ============================================================
        # 2. Color features (192)
        # ============================================================
        color_features = []
        n_frames = min(16, len(frames_np))
        print(f"   Processing {n_frames} frames for color features")
        
        for frame_idx in range(n_frames):
            frame = frames_np[frame_idx]
            
            # Convert to (height, width, channels)
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
        print(f"   Color features: {len(color_features)}")
        
        # ============================================================
        # 3. Temporal features (3)
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
        # 4. Bitrate features (6)
        # ============================================================
        bitrate_features = self._extract_bitrate_features(video_path)
        print(f"   Bitrate features: {len(bitrate_features)}")
        
        # ============================================================
        # 5. Combine
        # ============================================================
        combined = np.concatenate([
            [float(d3_avg), float(d3_std)],
            np.array(color_features, dtype=np.float32),
            np.array(temporal_features, dtype=np.float32),
            np.array(bitrate_features, dtype=np.float32)
        ])
        
        # Ensure exactly 203 features
        if len(combined) < 203:
            combined = np.pad(combined, (0, 203 - len(combined)), 'constant')
        elif len(combined) > 203:
            combined = combined[:203]
        
        print(f"   Final features: {len(combined)}")
        
        return np.array(combined, dtype=np.float32)
    
    def _extract_bitrate_features(self, video_path: Path) -> np.ndarray:
        if video_path is None or not Path(video_path).exists():
            return np.zeros(6)
        
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format', str(video_path)]
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
            
            bitrate = float(video_stream.get('bit_rate', 0))
            frame_rate_str = video_stream.get('avg_frame_rate', '0/1')
            if '/' in frame_rate_str:
                num, den = frame_rate_str.split('/')
                frame_rate = float(num) / float(den) if float(den) > 0 else 0
            else:
                frame_rate = float(frame_rate_str)
            
            return np.array([
                bitrate,
                frame_rate,
                float(video_stream.get('width', 0)),
                float(video_stream.get('height', 0)),
                float(format_info.get('duration', 0)),
                1.0 if video_stream.get('codec_name') == 'h264' else 0.0
            ])
        except:
            return np.zeros(6)
    
    def _compute_confidence(self, probability: float) -> float:
        return min(1.0, abs(probability - 0.5) * 2.4)


def get_predictor(model_dir: Path = None) -> D3PlusPredictor:
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
            device="cpu",
            use_d3=True
        )
    
    return _predictor_instance


_predictor_instance = None