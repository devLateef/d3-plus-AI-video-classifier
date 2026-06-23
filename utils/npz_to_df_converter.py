import numpy as np
import pandas as pd

# Load your combined data
data = np.load('full_features_with_gifs.npz')
X = data['features']
y = data['labels']

# --- Define all 203 column names ---
feature_names = []

# 1. D3 Features (2)
feature_names.extend(['d3_avg', 'd3_std'])

# 2. Color Features (192)
channels = ['R', 'G', 'B']
stats = ['mean', 'std', 'skew', 'kurt']

for frame in range(16):
    for channel in channels:
        for stat in stats:
            feature_names.append(f'frame{frame}_{channel}_{stat}')

# 3. Temporal Features (3)
feature_names.extend(['temporal_mean_diff', 'temporal_std_diff', 'temporal_max_diff'])

# 4. Bitrate Features (6)
feature_names.extend(['duration', 'frame_count', 'width', 'height', 'size_mb', 'is_gif'])

# Verify
print(f"Total features: {len(feature_names)}")
print(f"Data shape: {X.shape}")

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['label'] = y

# Save as CSV
df.to_csv('full_features_with_gifs.csv', index=False)

print(f"✅ CSV saved with {len(df)} rows and {len(df.columns)} columns")
print(f"   Features: {len(df.columns) - 1}")  # -1 for label column

# Show first few rows
print("\nFirst 5 rows:")
print(df.head())