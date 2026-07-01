"""
scripts/extract_geometric_features_enhanced.py
Extract 77 geometric features from 81-point Dlib landmarks.
WITH:
- Kalman filtering for temporal smoothing
- Face detection confidence
- Optical flow fallback for lost faces
- Mahalanobis distance for anomaly detection
- Inter-frame motion consistency
- Face size normalization
- Confidence weighting
"""

import torch
import sys
import numpy as np
import cv2
import dlib
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.data.dataset import D3Dataset


# ============================================================
# 1. KALMAN FILTER FOR LANDMARK SMOOTHING
# ============================================================

class LandmarkKalmanFilter:
    """
    Kalman filter for smoothing landmark trajectories over time.
    """
    
    def __init__(self, n_landmarks=81, process_noise=1e-4, measurement_noise=1e-2):
        self.n_landmarks = n_landmarks
        self.n_dim = n_landmarks * 2
        self.state_dim = self.n_dim * 2
        
        self.kf = cv2.KalmanFilter(self.state_dim, self.n_dim, 0)
        
        # State transition matrix (constant velocity)
        self.kf.transitionMatrix = np.eye(self.state_dim, dtype=np.float32)
        for i in range(self.n_dim):
            self.kf.transitionMatrix[i, i + self.n_dim] = 1.0
        
        # Measurement matrix
        self.kf.measurementMatrix = np.hstack([
            np.eye(self.n_dim, dtype=np.float32),
            np.zeros((self.n_dim, self.n_dim), dtype=np.float32)
        ])
        
        self.kf.processNoiseCov = np.eye(self.state_dim, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(self.n_dim, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(self.state_dim, dtype=np.float32)
        
        self.initialized = False
    
    def update(self, landmarks):
        if landmarks is None:
            return None
        
        z = landmarks.flatten().astype(np.float32)
        
        if not self.initialized:
            self.kf.statePost[:self.n_dim] = z
            self.kf.statePost[self.n_dim:] = 0
            self.initialized = True
            return landmarks
        
        self.kf.predict()
        corrected = self.kf.correct(z)
        smoothed = corrected[:self.n_dim].reshape(self.n_landmarks, 2)
        
        return smoothed
    
    def reset(self):
        self.initialized = False
        self.kf.statePost = np.zeros((self.state_dim, 1), dtype=np.float32)


# ============================================================
# 2. OPTICAL FLOW FALLBACK
# ============================================================

class OpticalFlowFallback:
    """
    Estimate landmarks using optical flow when face detection fails.
    """
    
    def __init__(self):
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        self.prev_frame = None
        self.prev_landmarks = None
    
    def compute_flow(self, prev_frame, curr_frame):
        """Compute dense optical flow between two frames."""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **self.flow_params)
        return flow
    
    def estimate_landmarks(self, curr_frame, prev_frame, prev_landmarks):
        """Estimate current landmarks using optical flow."""
        if prev_landmarks is None or prev_frame is None:
            return None
        
        flow = self.compute_flow(prev_frame, curr_frame)
        estimated = np.zeros_like(prev_landmarks)
        
        for i, (x, y) in enumerate(prev_landmarks):
            x_int, y_int = int(round(x)), int(round(y))
            
            if (0 <= x_int < flow.shape[1] and 0 <= y_int < flow.shape[0]):
                flow_at_point = flow[y_int, x_int]
                estimated[i] = [x + flow_at_point[0], y + flow_at_point[1]]
            else:
                estimated[i] = [x, y]
        
        return estimated
    
    def update(self, curr_frame, prev_frame, prev_landmarks):
        """Update with current frame and previous landmarks."""
        self.prev_frame = curr_frame
        self.prev_landmarks = prev_landmarks
        return prev_landmarks


# ============================================================
# 3. FACE DETECTION WITH CONFIDENCE (FIXED)
# ============================================================

def detect_face_with_confidence(frame, detector, predictor, min_confidence=0.7):
    """
    Detect face and return landmarks with confidence score.
    
    Returns:
        (landmarks, confidence) where confidence is based on detection quality
    """
    # ============================================================
    # FIX: Handle different channel formats
    # ============================================================
    if len(frame.shape) == 2:
        # Already grayscale
        gray = frame
    elif len(frame.shape) == 3 and frame.shape[2] == 1:
        # Single channel image (grayscale with shape)
        gray = frame[:, :, 0]
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        # BGR image (3 channels)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        # Fallback: try to convert
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            # If all else fails, convert to uint8 and use first channel
            gray = frame.astype(np.uint8)
            if len(gray.shape) > 2:
                gray = gray[:, :, 0]
    
    faces = detector(gray)
    
    if len(faces) == 0:
        return None, 0.0
    
    # Use the largest face
    largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
    
    try:
        landmarks = predictor(gray, largest_face)
        points = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # Calculate confidence based on face size and detection quality
        face_area = largest_face.width() * largest_face.height()
        img_area = frame.shape[0] * frame.shape[1]
        size_ratio = face_area / img_area
        
        # Higher confidence for larger faces
        confidence = min(1.0, size_ratio * 5.0)
        
        return points, confidence
    except:
        return None, 0.0


# ============================================================
# 4. FACE SIZE NORMALIZATION
# ============================================================

def normalize_face(landmarks, target_eye_distance=60):
    """
    Normalize face size to make features scale-invariant.
    """
    if landmarks is None:
        return None
    
    # Compute eye distance
    left_eye_center = np.mean(landmarks[36:42], axis=0)
    right_eye_center = np.mean(landmarks[42:48], axis=0)
    eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
    
    if eye_distance < 1e-6:
        return landmarks
    
    # Compute scaling factor
    scale = target_eye_distance / eye_distance
    
    # Center the face
    face_center = np.mean(landmarks, axis=0)
    normalized = (landmarks - face_center) * scale + face_center
    
    return normalized


# ============================================================
# 5. LANDMARK INTERPOLATION FOR MISSING FRAMES
# ============================================================

def interpolate_missing_landmarks(landmarks_list):
    """
    Interpolate missing landmarks from surrounding frames.
    """
    n_frames = len(landmarks_list)
    valid_indices = [i for i, lm in enumerate(landmarks_list) if lm is not None]
    
    if len(valid_indices) == 0:
        return [None] * n_frames
    
    if len(valid_indices) == 1:
        idx = valid_indices[0]
        return [landmarks_list[idx].copy() for _ in range(n_frames)]
    
    interpolated = [None] * n_frames
    
    for i in range(n_frames):
        if landmarks_list[i] is not None:
            interpolated[i] = landmarks_list[i].copy()
        else:
            before = max([j for j in valid_indices if j < i], default=None)
            after = min([j for j in valid_indices if j > i], default=None)
            
            if before is not None and after is not None:
                alpha = (i - before) / (after - before)
                a = landmarks_list[before]
                b = landmarks_list[after]
                interpolated[i] = (1 - alpha) * a + alpha * b
            elif before is not None:
                interpolated[i] = landmarks_list[before].copy()
            elif after is not None:
                interpolated[i] = landmarks_list[after].copy()
    
    return interpolated


# ============================================================
# 6. MAHALANOBIS DISTANCE COMPUTATION
# ============================================================

class MahalanobisDetector:
    """
    Compute Mahalanobis distance for anomaly detection.
    """
    
    def __init__(self):
        self.mu = None
        self.S_inv = None
        self.is_fitted = False
    
    def fit(self, landmark_vectors):
        """
        Fit reference distribution from real faces.
        
        Args:
            landmark_vectors: List of flattened landmark vectors (162,)
        """
        if len(landmark_vectors) < 2:
            return
        
        X = np.array(landmark_vectors)
        self.mu = np.mean(X, axis=0)
        S = np.cov(X, rowvar=False)
        
        # Pseudo-inverse for stability
        self.S_inv = np.linalg.pinv(S + np.eye(S.shape[0]) * 1e-6)
        self.is_fitted = True
    
    def compute_distance(self, landmarks):
        """Compute Mahalanobis distance for a face."""
        if not self.is_fitted or landmarks is None:
            return 0.0
        
        L = landmarks.flatten()
        return mahalanobis(L, self.mu, self.S_inv)


# ============================================================
# 7. FEATURE COMPUTATION FUNCTIONS
# ============================================================

def compute_distances(landmarks):
    """Compute inter-landmark distances."""
    distances = []
    
    # Eye widths
    distances.append(np.linalg.norm(landmarks[36] - landmarks[39]))
    distances.append(np.linalg.norm(landmarks[42] - landmarks[45]))
    
    # Eye heights
    distances.append(np.linalg.norm(landmarks[37] - landmarks[41]))
    distances.append(np.linalg.norm(landmarks[43] - landmarks[47]))
    
    # Mouth width and height
    distances.append(np.linalg.norm(landmarks[48] - landmarks[54]))
    distances.append(np.linalg.norm(landmarks[51] - landmarks[57]))
    
    # Nose width
    distances.append(np.linalg.norm(landmarks[31] - landmarks[35]))
    
    # Eye-to-nose
    left_eye_center = np.mean(landmarks[36:42], axis=0)
    nose_tip = landmarks[30]
    distances.append(np.linalg.norm(left_eye_center - nose_tip))
    
    # Jaw width
    distances.append(np.linalg.norm(landmarks[2] - landmarks[14]))
    
    # Forehead
    distances.append(np.linalg.norm(landmarks[69] - landmarks[75]))
    distances.append(np.linalg.norm(landmarks[69] - landmarks[81]))
    
    # Eye spacing
    right_eye_center = np.mean(landmarks[42:48], axis=0)
    distances.append(np.linalg.norm(left_eye_center - right_eye_center))
    
    return np.array(distances)


def compute_ratios(landmarks):
    """Compute geometric ratios."""
    ratios = []
    
    def ear(eye_indices):
        p1, p2, p3, p4, p5, p6 = eye_indices
        vertical_1 = np.linalg.norm(landmarks[p2] - landmarks[p6])
        vertical_2 = np.linalg.norm(landmarks[p3] - landmarks[p5])
        horizontal = np.linalg.norm(landmarks[p1] - landmarks[p4])
        return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)
    
    ratios.append(ear([36, 37, 38, 39, 40, 41]))
    ratios.append(ear([42, 43, 44, 45, 46, 47]))
    
    # Mouth ratio
    vertical_1 = np.linalg.norm(landmarks[51] - landmarks[57])
    vertical_2 = np.linalg.norm(landmarks[52] - landmarks[56])
    horizontal = np.linalg.norm(landmarks[48] - landmarks[54])
    ratios.append((vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6))
    
    # Eye spacing ratio
    face_width = np.linalg.norm(landmarks[0] - landmarks[16])
    ratios.append(np.linalg.norm(left_eye_center - right_eye_center) / (face_width + 1e-6))
    
    # Forehead ratio
    face_height = np.linalg.norm(landmarks[8] - landmarks[27])
    ratios.append(np.linalg.norm(landmarks[69] - landmarks[81]) / (face_height + 1e-6))
    
    # Eye symmetry
    ratios.append(np.linalg.norm(landmarks[36] - landmarks[39]) / 
                  (np.linalg.norm(landmarks[42] - landmarks[45]) + 1e-6))
    
    return np.array(ratios)


def compute_angles(landmarks):
    """Compute angles between facial landmarks."""
    angles = []
    
    def angle_between(a, b, c):
        v1 = a - b
        v2 = c - b
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
    
    # Head tilt
    left_eye_center = np.mean(landmarks[36:42], axis=0)
    right_eye_center = np.mean(landmarks[42:48], axis=0)
    head_tilt = np.arctan2(right_eye_center[1] - left_eye_center[1],
                           right_eye_center[0] - left_eye_center[0]) * 180 / np.pi
    angles.append(head_tilt)
    
    # Jaw angle
    angles.append(angle_between(landmarks[2], landmarks[8], landmarks[14]))
    
    # Nose angle
    angles.append(angle_between(landmarks[27], landmarks[30], landmarks[33]))
    
    # Eye corner angles
    angles.append(angle_between(landmarks[36], landmarks[39], landmarks[37]))
    angles.append(angle_between(landmarks[42], landmarks[45], landmarks[43]))
    
    # Mouth angles
    angles.append(angle_between(landmarks[48], landmarks[51], landmarks[54]))
    angles.append(angle_between(landmarks[48], landmarks[57], landmarks[54]))
    
    # Forehead angle
    angles.append(angle_between(landmarks[69], landmarks[75], landmarks[81]))
    
    return np.array(angles)


def compute_forehead_features(landmarks):
    """Extract forehead features (points 69-81)."""
    forehead = []
    forehead_points = landmarks[69:82]
    
    width = np.linalg.norm(forehead_points[0] - forehead_points[6])
    height = np.linalg.norm(forehead_points[0] - forehead_points[12])
    forehead.append(width)
    forehead.append(height)
    forehead.append(width / (height + 1e-6))
    forehead.append(np.std(forehead_points[:, 1]))
    
    center = np.mean(forehead_points, axis=0)
    forehead.append(np.mean(np.linalg.norm(forehead_points - center, axis=1)))
    
    return np.array(forehead)


def compute_temporal_features(landmarks, previous_landmarks):
    """Compute frame-to-frame changes."""
    if previous_landmarks is None:
        return np.zeros(15)
    
    temporal = []
    key_indices = [36, 42, 48, 54, 30, 8, 69, 81]
    
    for idx in key_indices:
        if idx < len(landmarks) and idx < len(previous_landmarks):
            temporal.append(np.linalg.norm(landmarks[idx] - previous_landmarks[idx]))
    
    all_displacements = np.linalg.norm(landmarks - previous_landmarks, axis=1)
    temporal.append(np.mean(all_displacements))
    temporal.append(np.std(all_displacements))
    temporal.append(np.max(all_displacements))
    
    # Pad to 15 features
    while len(temporal) < 15:
        temporal.append(0.0)
    
    return np.array(temporal[:15])


def extract_geometric_features(landmarks, previous_landmarks=None, confidence=1.0):
    """Extract 77 geometric features with confidence weighting."""
    if landmarks is None:
        return np.zeros(77)
    
    features = []
    features.extend(compute_distances(landmarks))
    features.extend(compute_ratios(landmarks))
    features.extend(compute_angles(landmarks))
    features.extend(compute_forehead_features(landmarks))
    
    temporal = compute_temporal_features(landmarks, previous_landmarks)
    features.extend(temporal)
    
    # Add confidence as a feature
    features.append(confidence)
    
    # Pad to exactly 77 features
    while len(features) < 77:
        features.append(0.0)
    
    return np.array(features[:77])


# ============================================================
# 8. MAIN EXTRACTION FUNCTION
# ============================================================

def extract_geometric_features_enhanced(csv_path, dataset_root, output_path="geometric_features.npz"):
    """
    Extract geometric features with all robustness enhancements.
    """
    print("Loading Dlib model...")
    detector = dlib.get_frontal_face_detector()
    dlib_dat_path = '/home/cuab/Documents/shape_predictor_81_face_landmarks/shape_predictor_81_face_landmarks.dat'
    predictor = dlib.shape_predictor(dlib_dat_path)
    
    # Initialize components
    kalman = LandmarkKalmanFilter()
    optical_flow = OpticalFlowFallback()
    mahalanobis_detector = MahalanobisDetector()
    
    # Load dataset
    dataset = D3Dataset(csv_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    all_features = []
    all_labels = []
    all_confidence_scores = []
    
    print(f"Processing {len(dataset)} videos...")
    print("Note: Features include Mahalanobis distance (for anomaly detection)")
    
    # First pass: collect landmarks for Mahalanobis fitting
    real_landmarks = []
    
    for idx, (frames, label) in enumerate(tqdm(loader, desc="Collecting reference landmarks")):
        if label.numpy()[0] == 0:  # Real video
            frames_np = frames.cpu().numpy()
            for frame in frames_np[:5]:  # Use first 5 frames
                if frame.shape[0] == 3:
                    frame_hwc = frame.transpose(1, 2, 0)
                else:
                    frame_hwc = frame
                landmarks, _ = detect_face_with_confidence(frame_hwc, detector, predictor)
                if landmarks is not None and len(landmarks) == 81:
                    real_landmarks.append(landmarks.flatten())
        
        if len(real_landmarks) >= 100:  # Enough samples
            break
    
    # Fit Mahalanobis detector
    if len(real_landmarks) > 10:
        mahalanobis_detector.fit(real_landmarks)
        print(f"✅ Mahalanobis detector fitted with {len(real_landmarks)} faces")
    else:
        print("⚠️ Not enough real faces for Mahalanobis fitting")
    
    # Main extraction loop
    for idx, (frames, label) in enumerate(tqdm(loader, desc="Extracting features")):
        frames_np = frames.cpu().numpy()
        
        # Extract landmarks with confidence
        raw_landmarks = []
        confidences = []
        prev_frame = None
        
        for frame in frames_np:
            if frame.shape[0] == 3:
                frame_hwc = frame.transpose(1, 2, 0)
            else:
                frame_hwc = frame
            
            landmarks, confidence = detect_face_with_confidence(frame_hwc, detector, predictor)
            confidences.append(confidence)
            
            # If face not detected, use optical flow fallback
            if landmarks is None and prev_frame is not None and len(raw_landmarks) > 0:
                prev_landmarks = raw_landmarks[-1]
                if prev_landmarks is not None:
                    landmarks = optical_flow.estimate_landmarks(
                        frame_hwc, prev_frame, prev_landmarks
                    )
                    confidences[-1] = 0.3  # Lower confidence for estimated
            
            raw_landmarks.append(landmarks)
            prev_frame = frame_hwc
        
        # Interpolate missing landmarks
        raw_landmarks = interpolate_missing_landmarks(raw_landmarks)
        
        # Apply Kalman filtering
        kalman.reset()
        smoothed_landmarks = []
        for landmarks in raw_landmarks:
            smoothed = kalman.update(landmarks)
            smoothed_landmarks.append(smoothed)
        
        # Normalize face size
        for i, landmarks in enumerate(smoothed_landmarks):
            if landmarks is not None:
                smoothed_landmarks[i] = normalize_face(landmarks)
        
        # Extract geometric features
        video_features = []
        previous_landmarks = None
        
        for i, landmarks in enumerate(smoothed_landmarks):
            confidence = confidences[i] if i < len(confidences) else 0.5
            features = extract_geometric_features(landmarks, previous_landmarks, confidence)
            video_features.append(features)
            previous_landmarks = landmarks
        
        # Aggregate over frames
        video_features = np.array(video_features)
        
        # Check if any frames had face detection
        if np.all(video_features == 0):
            aggregated = np.zeros(154)
            mahalanobis_score = 0.0
        else:
            # Compute Mahalanobis distance for each frame
            mahalanobis_scores = []
            for landmarks in smoothed_landmarks:
                if landmarks is not None:
                    dist = mahalanobis_detector.compute_distance(landmarks)
                    mahalanobis_scores.append(dist)
            
            # Aggregate features
            geo_mean = video_features.mean(axis=0)
            geo_std = video_features.std(axis=0)
            aggregated = np.concatenate([geo_mean, geo_std])
            
            # Add Mahalanobis statistics
            if mahalanobis_scores:
                mahalanobis_score = np.mean(mahalanobis_scores)
            else:
                mahalanobis_score = 0.0
        
        # Append features
        all_features.append(aggregated)
        all_labels.append(label.numpy()[0])
        all_confidence_scores.append(mahalanobis_score)
    
    # Save features and labels
    features_arr = np.array(all_features)
    labels_arr = np.array(all_labels)
    
    np.savez_compressed(output_path, 
                        features=features_arr, 
                        labels=labels_arr,
                        confidence_scores=np.array(all_confidence_scores))
    
    print(f"\n✅ Geometric features saved to: {output_path}")
    print(f"   Feature shape: {features_arr.shape}")
    print(f"   Feature vector length: {features_arr.shape[1]}")
    print(f"   Mahalanobis scores included as confidence measure")
    
    return features_arr, labels_arr


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--dataset-root', type=str, required=True)
    parser.add_argument('--output', type=str, default='geometric_features_enhanced.npz')
    
    args = parser.parse_args()
    
    extract_geometric_features_enhanced(
        csv_path=args.csv,
        dataset_root=args.dataset_root,
        output_path=args.output
    )