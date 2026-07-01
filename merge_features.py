"""
scripts/merge_features.py
Merge geometric features with D3-X features.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def merge_features():
    """
    Merge geometric_features_enhanced.npz with full_features_with_gifs.csv
    """
    print("="*60)
    print("MERGING GEOMETRIC AND D3-X FEATURES")
    print("="*60)
    
    # Load geometric features
    print("\n📂 Loading geometric features from: geometric_features.npz")
    geo_data = np.load('geometric_features.npz')
    X_geo = geo_data['features']
    y_geo = geo_data['labels']
    print(f"   Geometric features shape: {X_geo.shape}")
    print(f"   Geometric labels: {len(y_geo)}")
    
    # Create feature names
    def get_feature_names(n_features):
        return [f'geo_{i}' for i in range(n_features)]
    
    feature_names = get_feature_names(X_geo.shape[1])
    geo_df = pd.DataFrame(X_geo, columns=feature_names)
    geo_df['label'] = y_geo
    print(f"   Geometric DataFrame shape: {geo_df.shape}")
    
    # Load D3-X features
    print("\n📂 Loading D3-X features from: full_features_with_gifs.csv")
    d3x_df = pd.read_csv('data_csv/full_features_with_gifs.csv')
    print(f"   D3-X features shape: {d3x_df.shape}")
    
    # Check if sample counts match
    if len(d3x_df) != len(geo_df):
        print(f"\n⚠️ Sample count mismatch:")
        print(f"   D3-X samples: {len(d3x_df)}")
        print(f"   Geometric samples: {len(geo_df)}")
        print(f"   Using minimum count: {min(len(d3x_df), len(geo_df))}")
        
        # Trim to minimum
        min_samples = min(len(d3x_df), len(geo_df))
        d3x_df = d3x_df.iloc[:min_samples]
        geo_df = geo_df.iloc[:min_samples]
    
    # Drop label from D3-X
    d3x_features = d3x_df.drop('label', axis=1)
    d3x_label = d3x_df['label']
    
    # Check if video_name exists and preserve it
    video_name_col = None
    if 'video_name' in d3x_features.columns:
        video_name_col = d3x_features['video_name']
        d3x_features = d3x_features.drop('video_name', axis=1)
    
    # Concatenate features
    print("\n🔗 Merging features...")
    merged_features = pd.concat([
        d3x_features.reset_index(drop=True), 
        geo_df.drop('label', axis=1).reset_index(drop=True)
    ], axis=1)
    
    # Add label
    merged_features['label'] = d3x_label.values
    
    # Add video_name back if it existed
    if video_name_col is not None:
        merged_features.insert(0, 'video_name', video_name_col.values)
    
    # Save
    output_path = 'full_features_merged.csv'
    print(f"\n💾 Saving merged features to: {output_path}")
    merged_features.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "="*60)
    print("✅ MERGING COMPLETE")
    print("="*60)
    print(f"   Merged shape: {merged_features.shape}")
    print(f"   D3-X features: {d3x_features.shape[1]}")
    print(f"   Geometric features: {geo_df.shape[1] - 1}")  # -1 for label
    print(f"   Total features (excluding label): {merged_features.shape[1] - 1}")
    print(f"   Saved to: {output_path}")
    
    # Feature breakdown
    print("\n📊 FEATURE BREAKDOWN:")
    print(f"   D3-X (203): 2 (D3) + 192 (Color) + 3 (Temporal) + 6 (Bitrate)")
    print(f"   Geometric (154): 77 (Mean) + 77 (Std)")
    print(f"   Total: 357 features")
    
    # Check for missing values
    missing = merged_features.isna().sum().sum()
    if missing > 0:
        print(f"\n⚠️ Warning: {missing} missing values detected!")
    else:
        print(f"\n✅ No missing values detected.")
    
    return merged_features

if __name__ == "__main__":
    merge_features()