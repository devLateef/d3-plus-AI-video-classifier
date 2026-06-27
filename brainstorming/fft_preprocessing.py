import torch
import torch.nn as nn
import torch.fft

class DFTPreprocessingLayer(nn.Module):
    """
    A PyTorch preprocessing layer that converts spatial image frames 
    into their normalized 2D Frequency Magnitude Spectrum.
    
    Input shape:  (Batch, Channels, Height, Width) - e.g., (B, 3, H, W)
    Output shape: (Batch, Channels, Height, Width) - Frequency representation
    """
    def __init__(self, eps=1e-7):
        super(DFTPreprocessingLayer, self).__init__()
        self.eps = eps

    def forward(self, x):
        # 1. Compute the 2D Real Fast Fourier Transform (RFFT) across spatial dimensions (H, W)
        # We use torch.fft.fft2 to handle full complex output for all channels
        fft_complex = torch.fft.fft2(x, dim=(-2, -1))
        
        # 2. Shift the zero-frequency component to the center of the spectrum
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
        
        # 3. Calculate the Magnitude Spectrum (removes phase, isolates frequency power)
        magnitude = torch.abs(fft_shifted)
        
        # 4. Apply Log-Scaling to compress the massive dynamic range of the spectrum
        # Without this, the DC component (center) completely overpowers high frequencies
        log_magnitude = torch.log(magnitude + self.eps)
        
        # 5. Min-Max Normalization per image/channel to scale values safely between [0, 1]
        # Reshape to easily extract min/max across the spatial dimensions
        b, c, h, w = log_magnitude.shape
        flat_magnitude = log_magnitude.view(b, c, -1)
        
        min_vals = flat_magnitude.min(dim=-1, keepdim=True)[0].view(b, c, 1, 1)
        max_vals = flat_magnitude.max(dim=-1, keepdim=True)[0].view(b, c, 1, 1)
        
        normalized_magnitude = (log_magnitude - min_vals) / (max_vals - min_vals + self.eps)
        
        return normalized_magnitude

# Example Usage & Verification Pipeline
if __name__ == "__main__":
    # Simulate a batch of 4 RGB video frames of size 256x256
    # Batch Size = 4, Channels = 3 (RGB), Height = 256, Width = 256
    dummy_video_frames = torch.rand(4, 3, 256, 256)
    
    # Initialize the preprocessing layer
    dft_layer = DFTPreprocessingLayer()
    
    # Run the layer
    with torch.no_grad():
        frequency_maps = dft_layer(dummy_video_frames)
        
    print("--- Pipeline Verification ---")
    print(Input shape:       {dummy_video_frames.shape})
    print(Output frequency shape: {frequency_maps.shape})
    print(Output value range:    Min={frequency_maps.min().item():.4f}, Max={frequency_maps.max().item():.4f})
    
    # Check shape preservation for downstream model ingestion
    assert dummy_video_frames.shape == frequency_maps.shape, "Shape mismatch!"
    print("\n[Success] Frequency feature map is ready to feed into standard 2D CNN backbones (ResNet/ViT).")