import numpy as np

# Load video features (203 features)
video_data = np.load('full_features.npz')
X_video = video_data['features']
y_video = video_data['labels']

# Load new GIF features (203 features)
gif_data = np.load('gif_features.npz')
X_gif = gif_data['features']
y_gif = gif_data['labels']

# They should now match
print(f"Video features: {X_video.shape[1]}")
print(f"GIF features: {X_gif.shape[1]}")

# Combine
X_combined = np.vstack([X_video, X_gif])
y_combined = np.concatenate([y_video, y_gif])

# Save combined
np.savez_compressed('full_features_with_gifs.npz', 
                    features=X_combined, 
                    labels=y_combined)

print(f"\nCombined: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
print(f"   Video: {X_video.shape[0]}, GIF: {X_gif.shape[0]}")