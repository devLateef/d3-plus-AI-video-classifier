"""
scripts/validate_new_dataset.py
Validate trained model on a new video dataset.
"""

import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import joblib
import os
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.models.d3_model import D3Model
from research.data.dataset import crop_center_by_percentage, get_preprocessing_pipeline
from scripts.extract_all_features import (
    extract_color_features, 
    extract_temporal_features,
    extract_bitrate_features_video
)


class FeatureExtractorForNewVideos:
    """
    Extract the exact same 203 features from new videos.
    """
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load D3 model for feature extraction
        if model_path:
            self.model = D3Model(encoder_type='XCLIP-16', loss_type='l2').to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            # Use pretrained model without loading weights
            self.model = D3Model(encoder_type='XCLIP-16', loss_type='l2').to(self.device)
        
        self.model.eval()
        self.trans = get_preprocessing_pipeline()
        
        print(f" Feature extractor ready on {self.device}")
    
    def load_frames_from_video(self, video_path: Path, n_frames: int = 16) -> torch.Tensor:
        """
        Extract and preprocess frames from a video.
        """
        import subprocess
        import tempfile
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            output_pattern = str(temp_dir / "%d.jpg")
            
            # Extract frames using ffmpeg
            cmd = [
                "ffmpeg",
                "-loglevel", "quiet",
                "-i", str(video_path),
                "-vf", "fps=8",
                "-t", "3",
                output_pattern
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except:
                return torch.zeros(n_frames, 3, 224, 224)
            
            # Load frames
            frame_files = sorted(temp_dir.glob("*.jpg"))
            if not frame_files:
                return torch.zeros(n_frames, 3, 224, 224)
            
            # Sample frames
            total_frames = len(frame_files)
            if total_frames < n_frames:
                indices = np.random.choice(total_frames, n_frames, replace=True)
            else:
                indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
            
            frames = []
            for idx in indices:
                frame_path = frame_files[idx]
                image = cv2.imread(str(frame_path))
                if image is None:
                    continue
                
                image = crop_center_by_percentage(image, 0.1)
                augmented = self.trans(image=image)
                image = augmented["image"]
                frames.append(image.transpose(2, 0, 1))
            
            while len(frames) < n_frames:
                frames.append(np.zeros((3, 224, 224)))
            
            return torch.from_numpy(np.stack(frames[:n_frames])).float()
    
    def extract_features_from_video(self, video_path: Path) -> np.ndarray:
        """
        Extract ALL 203 features from a single video.
        """
        # 1. Load frames
        frames = self.load_frames_from_video(video_path)
        frames_tensor = frames.unsqueeze(0).to(self.device)
        
        # 2. D3 Features
        with torch.no_grad():
            _, d3_avg, d3_std = self.model(frames_tensor)
        
        # 3. Color Features
        frames_cpu = frames
        color_feats = extract_color_features(frames_cpu)
        
        # 4. Temporal Features
        temporal_feats = extract_temporal_features(frames_cpu)
        
        # 5. Bitrate Features
        bitrate_feats = extract_bitrate_features_video(video_path)
        
        # 6. Combine ALL features
        combined = np.concatenate([
            [d3_avg.cpu().numpy()[0]],
            [d3_std.cpu().numpy()[0]],
            color_feats,
            temporal_feats,
            bitrate_feats
        ])
        
        return combined
    
    def extract_features_from_folder(self, folder_path: Path) -> pd.DataFrame:
        """
        Extract features from all videos in a folder.
        """
        video_paths = list(folder_path.rglob('*.mp4')) + list(folder_path.rglob('*.avi')) + list(folder_path.rglob('*.mov'))
        
        all_features = []
        video_names = []
        
        print(f"Processing {len(video_paths)} videos...")
        
        for video_path in tqdm(video_paths):
            try:
                features = self.extract_features_from_video(video_path)
                all_features.append(features)
                video_names.append(video_path.stem)
            except Exception as e:
                print(f"Error processing {video_path.name}: {e}")
        
        # Create DataFrame with feature names
        feature_names = []
        feature_names.extend(['d3_avg', 'd3_std'])
        
        for frame in range(16):
            for channel in ['R', 'G', 'B']:
                for stat in ['mean', 'std', 'skew', 'kurt']:
                    feature_names.append(f'frame{frame}_{channel}_{stat}')
        
        feature_names.extend(['temporal_mean_diff', 'temporal_std_diff', 'temporal_max_diff'])
        feature_names.extend(['duration', 'frame_count', 'width', 'height', 'size_mb', 'is_gif'])
        
        df = pd.DataFrame(all_features, columns=feature_names)
        df['video_name'] = video_names
        
        return df


def validate_model_on_new_dataset(
    features_df: pd.DataFrame,
    model_path: Path,
    output_dir: Path = Path("validation_results")
):
    """
    Validate trained model on extracted features.
    """
    # Load models
    rf_model = joblib.load(model_path / "random_forest_model.pkl")
    imputer = joblib.load(model_path / "imputer.pkl")
    scaler = joblib.load(model_path / "scaler.pkl")
    
    # Prepare features
    X = features_df.drop('video_name', axis=1)
    
    # Impute and scale
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    
    # Predict
    y_pred = rf_model.predict(X_scaled)
    y_proba = rf_model.predict_proba(X_scaled)[:, 1]
    
    # Create results
    results = features_df[['video_name']].copy()
    results['prediction'] = y_pred
    results['probability'] = y_proba
    results['label'] = 1  # All are fake (Text2Video-Zero is an AI generator)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "validation_results.csv", index=False)
    
    # Print summary
    print("\n" + "="*60)
    print(" VALIDATION RESULTS")
    print("="*60)
    print(f"Total videos: {len(results)}")
    print(f"Predicted as Fake: {results['prediction'].sum()}")
    print(f"Predicted as Real: {len(results) - results['prediction'].sum()}")
    print(f"Detection Rate: {results['prediction'].mean():.4f}")
    
    # Show sample predictions
    print("\nSample predictions:")
    print(results.head(10).to_string(index=False))
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-folder', type=str, required=True,
                       help='Path to folder containing videos')
    parser.add_argument('--model-dir', type=str, default='trained_models',
                       help='Directory containing trained models')
    parser.add_argument('--output', type=str, default='validation_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Step 1: Extract features from new videos
    print("="*60)
    print(" EXTRACTING FEATURES FROM NEW VIDEOS")
    print("="*60)
    
    extractor = FeatureExtractorForNewVideos()
    features_df = extractor.extract_features_from_folder(Path(args.video_folder))
    
    # Step 2: Validate
    results = validate_model_on_new_dataset(
        features_df,
        Path(args.model_dir),
        Path(args.output)
    )
    
    print(f"Validation complete! Results saved to {args.output}/")