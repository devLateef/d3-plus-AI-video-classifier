"""
scripts/video2csv.py
Create a single CSV with labels - Updated for Real/Fake folder structure.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def count_frames_in_directory(frame_dir):
    """Count number of frames in a directory."""
    frame_files = list(frame_dir.glob('*.jpg')) + list(frame_dir.glob('*.png'))
    return len(frame_files)


def generate_csv_single_file(dataset_path, output_csv=None, max_frames_threshold=8):
    """
    Generate a single CSV with labels for all videos.
    
    CSV columns:
        - frame_dir: Path to the frame directory
        - label: 0 for real, 1 for fake
        - generator: Name of the generator (for fake videos)
        - frame_count: Number of frames in the directory
        - video_id: Unique identifier for the video
    """
    dataset_path = Path(dataset_path)
    frames_root = dataset_path / "frames"
    
    if not frames_root.exists():
        print(f"Error: Frames directory not found: {frames_root}")
        print("Please run video2frame.py first.")
        return None
    
    if output_csv is None:
        output_csv = dataset_path / "csv" / "dataset.csv"
    
    # Create output directory
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating CSV from: {frames_root}")
    print(f"Output: {output_csv}")
    
    data = []
    
    # Iterate through all video frame directories
    # Structure: frames/Real/video1/ or frames/Fake/model1/video1/
    for video_dir in tqdm(list(frames_root.rglob('*')), desc="Scanning directories"):
        if not video_dir.is_dir():
            continue
        
        # Check if this is a leaf directory (contains frames)
        frame_files = list(video_dir.glob('*.jpg')) + list(video_dir.glob('*.png'))
        if len(frame_files) < max_frames_threshold:
            continue
        
        # Determine label from path
        rel_path = video_dir.relative_to(frames_root)
        path_parts = rel_path.parts
        
        # Check if path starts with 'Real' or 'Fake'
        if len(path_parts) > 0 and path_parts[0].lower() == 'real':
            label = 0
            generator = 'real'
            video_id = '_'.join(path_parts[1:])  # Skip 'Real'
        elif len(path_parts) > 0 and path_parts[0].lower() == 'fake':
            label = 1
            # generator is the subfolder (model name)
            generator = path_parts[1] if len(path_parts) >= 2 else 'unknown'
            video_id = '_'.join(path_parts[1:])  # Skip 'Fake'
        else:
            # Fallback: try to detect from parent folder
            # If we can't detect, assume fake (conservative approach)
            if any('real' in part.lower() for part in path_parts):
                label = 0
                generator = 'real'
            else:
                label = 1
                generator = path_parts[0] if len(path_parts) > 0 else 'unknown'
            video_id = '_'.join(path_parts)
        
        # Count frames
        frame_count = len(frame_files)
        
        data.append({
            'frame_dir': str(video_dir),
            'label': label,
            'generator': generator,
            'frame_count': frame_count,
            'video_id': video_id
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("⚠️ No valid frame directories found!")
        return None
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ CSV generated!")
    print(f"   Total samples: {len(df)}")
    print(f"   Real: {len(df[df['label'] == 0])}")
    print(f"   Fake: {len(df[df['label'] == 1])}")
    print(f"   Generators: {df['generator'].unique().tolist()}")
    print(f"   Saved to: {output_csv}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate single CSV with labels for D3 training'
    )
    parser.add_argument('--dataset-path', type=str, default='GenVideo',
                       help='Path to the dataset directory')
    parser.add_argument('--output-csv', type=str, default=None,
                       help='Path to output CSV file')
    parser.add_argument('--max-frames', type=int, default=8,
                       help='Minimum frames required per video')
    
    args = parser.parse_args()
    
    generate_csv_single_file(
        dataset_path=args.dataset_path,
        output_csv=args.output_csv,
        max_frames_threshold=args.max_frames
    )


if __name__ == '__main__':
    main()