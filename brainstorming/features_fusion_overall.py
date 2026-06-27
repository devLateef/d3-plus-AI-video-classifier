import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =====================================================================
# STEP 1: Define the Dynamic Feature Extractor (Residual -> FFT -> Causal)
# =====================================================================
def extract_dynamic_numerical_features(video_frames):
    """
    Processes raw video frames through the Causal Residual -> FFT pipeline
    and collapses the spatial maps into flat numerical statistics.
    Input: video_frames tensor of shape (Frames, Channels, Height, Width)
    Output: A 1D numpy array of structural temporal features
    """
    # 1. Compute Causal Interframe Residuals
    residuals = torch.abs(video_frames[1:] - video_frames[:-1])
    
    # 2. Apply FFT to the residuals
    fft_complex = torch.fft.fft2(residuals, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
    
    magnitude = torch.abs(fft_shifted)
    phase = torch.angle(fft_shifted)
    
    # 3. Simulate Causal Filtering (Exponential Moving Average across Time)
    alpha = 0.3
    filtered_magnitude = magnitude.clone()
    for t in range(1, filtered_magnitude.shape[0]):
        filtered_magnitude[t] = alpha * magnitude[t] + (1 - alpha) * filtered_magnitude[t-1]
        
    # 4. Statistical Pooling: Convert the maps into flat scalar values
    # We take the mean and variance across the temporal and spatial dimensions
    mean_mag = torch.mean(filtered_magnitude).item()
    std_mag = torch.std(filtered_magnitude).item()
    mean_phase = torch.mean(phase).item()
    std_phase = torch.std(phase).item()
    residual_variance = torch.var(residuals).item()
    
    # Return these as a flat numerical array (5 features)
    return np.array([mean_mag, std_mag, mean_phase, std_phase, residual_variance])

# =====================================================================
# STEP 2: Simulate Your Complete Fused Dataset Construction
# =====================================================================
num_samples = 100  # Total videos in your dataset
existing_handcrafted_dim = 203  # Your original feature count (RGB stats, etc.)
bitrate_dim = 2  # e.g., average bitrate, peak bitrate

# Placeholders for our final design matrix
X_list = []
y_list = []

print("Extracting and fusing multimodal features...")
for i in range(num_samples):
    # Simulate a raw video sequence for this sample: 10 frames, RGB, 64x64 resolution
    simulated_video = torch.rand(10, 3, 64, 64)
    
    # A. Extract the dynamic/wave features from scratch (5 numerical features)
    dynamic_feats = extract_dynamic_numerical_features(simulated_video)
    
    # B. Generate your 203 existing handcrafted statistics + bitrate traits
    handcrafted_feats = np.random.normal(loc=0.0, scale=1.0, size=(existing_handcrafted_dim,))
    bitrate_feats = np.random.uniform(1500, 5000, size=(bitrate_dim,))
    
    # C. Feature Fusion: Concatenate them horizontally into a single flat vector
    # Total dim = 5 (dynamic) + 203 (handcrafted) + 2 (bitrate) = 210 features
    fused_vector = np.concatenate([dynamic_feats, handcrafted_feats, bitrate_feats])
    
    # Assign binary label (0 = Real, 1 = Fake)
    label = np.random.choice([0, 1])
    
    X_list.append(fused_vector)
    y_list.append(label)

# Convert lists to final structural numpy matrices for Scikit-Learn
X = np.array(X_list)
y = np.array(y_list)

print(f"Final Dataset Shape: {X.shape} (Samples, Total Fused Features)")

# =====================================================================
# STEP 3: Train and Evaluate the Machine Learning Classifier
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest
# Random Forests handle mixed multi-modal statistical distributions exceptionally well
clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
clf.fit(X_train, y_train)

# Evaluate model output integrity
predictions = clf.predict(X_test)
print("\n--- Model Classification Report ---")
print(classification_report(y_test, predictions, target_names=["Real (0)", "Fake (1)"]))
