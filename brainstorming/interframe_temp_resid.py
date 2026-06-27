import torch

def compute_causal_residuals(video_tensor):
    """
    Computes temporal residuals causally.
    Input shape: (Batch, Frames, Channels, Height, Width)
    Output shape: (Batch, Frames-1, Channels, Height, Width)
    """
    # Frame t (from frame 1 to the end)
    frames_current = video_tensor[:, 1:, :, :, :]
    
    # Frame t-1 (from frame 0 to the second-to-last frame)
    frames_past = video_tensor[:, :-1, :, :, :]
    
    # Absolute difference isolates the changes over time
    residuals = torch.abs(frames_current - frames_past)
    
    return residuals

# Quick Verification
# 1 Video, 5 Frames, 3 Channels, 128x128 resolution
sample_video = torch.rand(1, 5, 3, 128, 128)
video_residuals = compute_causal_residuals(sample_video)

print(f"Original Video Shape: {sample_video.shape}")
print(f"Residuals Video Shape: {video_residuals.shape}") 
# Notice the frame dimension drops by 1 because the first frame has no past.