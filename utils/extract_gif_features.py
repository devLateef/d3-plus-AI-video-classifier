import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy.stats import skew, kurtosis
import imageio
from PIL import Image
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.models.d3_model import D3Model


def get_preprocessing_pipeline():
    """Get preprocessing pipeline for images."""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])


def crop_center_by_percentage(image, percentage=0.1):
    """Crop center of image by percentage."""
    if image is None or image.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    if width > height:
        left_pixels = int(width * percentage)
        right_pixels = int(width * percentage)
        if left_pixels >= width:
            return image
        return image[:, left_pixels:width - right_pixels]
    else:
        up_pixels = int(height * percentage)
        down_pixels = int(height * percentage)
        if up_pixels >= height:
            return image
        return image[up_pixels:height - down_pixels, :]


def convert_frame_to_rgb(frame):
    """
    Convert any frame format to RGB.
    Returns a numpy array of shape (height, width, 3) or None on failure.
    """
    if frame is None:
        return None
    
    # Check if frame is empty
    if isinstance(frame, np.ndarray):
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            return None
    else:
        return None
    
    # Convert to PIL Image first for robust format handling
    try:
        if isinstance(frame, np.ndarray):
            if frame.dtype == np.uint8:
                # If frame is already uint8
                if frame.shape[-1] == 4:  # RGBA
                    img = Image.fromarray(frame, 'RGBA')
                    img = img.convert('RGB')
                    return np.array(img)
                elif frame.shape[-1] == 3:  # RGB
                    return frame
                elif len(frame.shape) == 2:  # Grayscale
                    return np.stack([frame] * 3, axis=-1)
                else:
                    # Try to convert
                    return np.array(Image.fromarray(frame).convert('RGB'))
            else:
                # Normalize to uint8
                frame_normalized = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255).astype(np.uint8)
                return convert_frame_to_rgb(frame_normalized)
        else:
            return None
    except Exception as e:
        return None


def load_gif_frames(gif_path, n_frames=16):
    """
    Load and preprocess frames from a GIF.
    Returns a tensor of shape (n_frames, 3, 224, 224).
    """
    try:
        reader = imageio.get_reader(gif_path)
        total_frames = reader.get_length()
        
        # If GIF has no frames, return zeros
        if total_frames == 0:
            return torch.zeros(n_frames, 3, 224, 224)
        
        # Sample frames evenly
        if total_frames < n_frames:
            indices = np.random.choice(total_frames, n_frames, replace=True)
        else:
            indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        
        trans = get_preprocessing_pipeline()
        frames = []
        
        for idx in indices:
            try:
                frame = reader.get_data(idx)
                
                # Convert to RGB
                frame_rgb = convert_frame_to_rgb(frame)
                if frame_rgb is None:
                    # If conversion fails, use a blank frame
                    frame_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
                
                # Ensure we have the right shape
                if frame_rgb.shape[0] == 0 or frame_rgb.shape[1] == 0:
                    frame_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
                
                # Crop and preprocess
                frame_cropped = crop_center_by_percentage(frame_rgb, 0.1)
                if frame_cropped.shape[0] == 0 or frame_cropped.shape[1] == 0:
                    frame_cropped = np.zeros((224, 224, 3), dtype=np.uint8)
                
                augmented = trans(image=frame_cropped)
                frame_processed = augmented["image"]
                frames.append(frame_processed.transpose(2, 0, 1))
                
            except Exception as e:
                # On any error, append a blank frame
                frames.append(np.zeros((3, 224, 224)))
        
        # Ensure we have the right number of frames
        while len(frames) < n_frames:
            frames.append(np.zeros((3, 224, 224)))
        
        return torch.from_numpy(np.stack(frames[:n_frames])).float()
        
    except Exception as e:
        print(f"Error loading GIF {gif_path}: {e}")
        return torch.zeros(n_frames, 3, 224, 224)


def extract_color_features(frames):
    """Extract color features (mean, std, skew, kurtosis) from frames."""
    color_features = []
    for frame in frames:
        for c in range(3):
            channel = frame[c, :, :].flatten().cpu().numpy()
            color_features.extend([
                np.mean(channel),
                np.std(channel),
                skew(channel),
                kurtosis(channel)
            ])
    return np.array(color_features)


def extract_temporal_features(frames):
    """Extract temporal features (mean diff, std diff, max diff)."""
    diffs = []
    for i in range(1, len(frames)):
        diff = torch.abs(frames[i] - frames[i-1]).mean().item()
        diffs.append(diff)
    
    if diffs:
        return np.array([np.mean(diffs), np.std(diffs), np.max(diffs)])
    else:
        return np.zeros(3)


def extract_gif_bitrate_features(gif_path):
    """
    Extract GIF-specific features (matching video format: 6 features).
    """
    try:
        reader = imageio.get_reader(gif_path)
        n_frames = reader.get_length()
        
        try:
            first_frame = reader.get_data(0)
            height, width = first_frame.shape[:2]
        except:
            height, width = 0, 0
        
        file_size = os.path.getsize(gif_path) / (1024 * 1024)
        
        try:
            metadata = reader.get_meta_data()
            duration = metadata.get('duration', 0) / 1000
        except:
            duration = 0
        
        # Return 6 features to match video format
        # [duration, frame_count, width, height, size_mb, is_gif]
        return np.array([
            float(duration),      # Duration in seconds
            float(n_frames),      # Number of frames
            float(width),         # Width
            float(height),        # Height
            float(file_size),     # File size in MB
            1.0                   # Is GIF flag
        ])
        
    except Exception as e:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def extract_gif_features(gif_path, model, device):
    """
    Extract ALL features from a single GIF.
    Returns a feature vector matching the video extraction format.
    """
    # 1. Load frames
    frames = load_gif_frames(gif_path, n_frames=16)
    frames = frames.unsqueeze(0).to(device)  # Add batch dimension
    
    # 2. D3 Features
    with torch.no_grad():
        _, d3_avg, d3_std = model(frames)
    
    # 3. Color Features
    frames_cpu = frames[0].cpu()
    color_feats = extract_color_features(frames_cpu)
    
    # 4. Temporal Features
    temporal_feats = extract_temporal_features(frames_cpu)
    
    # 5. GIF-specific bitrate features (7 features)
    bitrate_feats = extract_gif_bitrate_features(gif_path)
    
    # 6. Combine ALL features
    combined = np.concatenate([
        [d3_avg.cpu().numpy()[0]],      # D3 avg (1)
        [d3_std.cpu().numpy()[0]],      # D3 std (1)
        color_feats,                    # Color (192)
        temporal_feats,                 # Temporal (3)
        bitrate_feats                   # GIF Bitrate (7)
    ])
    
    return combined


def extract_all_gif_features(
    dataset_root,
    output_path="gif_features.npz",
    max_gifs=None
):
    """
    Extract features from all GIF files in the dataset.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset_root = Path(dataset_root)
    
    # Find all GIF files
    gif_files = list(dataset_root.rglob('*.gif'))
    
    # Limit if specified
    if max_gifs is not None:
        gif_files = gif_files[:max_gifs]
    
    print(f"Found {len(gif_files)} GIF files")
    
    if len(gif_files) == 0:
        print("No GIF files found!")
        return None, None
    
    # Load D3 model for feature extraction
    print("Loading XCLIP model (pretrained)...")
    model = D3Model(encoder_type='XCLIP-16', loss_type='l2').to(device)
    
    # Extract features
    all_features = []
    all_labels = []
    all_paths = []
    error_count = 0
    
    for gif_path in tqdm(gif_files, desc="Extracting GIF features"):
        # Determine label from path
        path_str = str(gif_path)
        if 'Real' in path_str:
            label = 0
        else:
            label = 1
        
        try:
            features = extract_gif_features(gif_path, model, device)
            all_features.append(features)
            all_labels.append(label)
            all_paths.append(str(gif_path))
        except Exception as e:
            error_count += 1
            if error_count <= 10:  # Only print first 10 errors
                print(f"Error processing {gif_path.name}: {e}")
    
    if len(all_features) == 0:
        print("No GIF features extracted!")
        return None, None
    
    # Convert to arrays
    features_arr = np.array(all_features)
    labels_arr = np.array(all_labels)
    
    # Save
    np.savez_compressed(output_path, 
                        features=features_arr, 
                        labels=labels_arr,
                        paths=np.array(all_paths))
    
    print(f"\n GIF features saved to: {output_path}")
    print(f"   Feature shape: {features_arr.shape}")
    print(f"   Labels shape: {labels_arr.shape}")
    print(f"   Feature vector length: {features_arr.shape[1]}")
    print(f"   Successful: {len(all_features)}")
    print(f"   Errors: {error_count}")
    
    return features_arr, labels_arr


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from GIF files')
    parser.add_argument('--dataset-root', type=str, required=True,
                       help='Path to the dataset root (where Real/ and Fake/ are)')
    parser.add_argument('--output', type=str, default='gif_features.npz',
                       help='Output path for GIF features')
    parser.add_argument('--max-gifs', type=int, default=None,
                       help='Maximum number of GIFs to process (for testing)')
    
    args = parser.parse_args()
    
    extract_all_gif_features(
        dataset_root=args.dataset_root,
        output_path=args.output,
        max_gifs=args.max_gifs
    )