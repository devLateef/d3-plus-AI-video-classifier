import torch
import torch.nn as nn
import torch.fft

class DualChannelFourierExtractor(nn.Module):
    """
    A PyTorch module built from scratch to extract and normalize both 
    Magnitude and Phase Spectrums simultaneously from an input video frame/residual.
    
    Input shape:  (Batch, Channels, Height, Width) -> e.g., (B, C, H, W)
    Output shape: (Batch, Channels * 2, Height, Width) -> Channel 0: Mag, Channel 1: Phase
    """
    def __init__(self, eps=1e-7):
        super(DualChannelFourierExtractor, self).__init__()
        self.eps = eps

    def _normalize_spectrum(self, spectrum):
        """Applies instance min-max normalization across spatial dimensions."""
        b, c, h, w = spectrum.shape
        flat = spectrum.view(b, c, -1)
        
        min_vals = flat.min(dim=-1, keepdim=True).view(b, c, 1, 1)
        max_vals = flat.max(dim=-1, keepdim=True).view(b, c, 1, 1)
        
        return (spectrum - min_vals) / (max_vals - min_vals + self.eps)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 1. Compute the 2D Fast Fourier Transform
        fft_complex = torch.fft.fft2(x, dim=(-2, -1))
        
        # 2. Shift low frequencies to the center (critical for up-sampling artifacts)
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
        
        # 3. Extract Magnitude & apply log-scaling to manage dynamic range
        magnitude = torch.abs(fft_shifted)
        log_magnitude = torch.log(magnitude + self.eps)
        
        # 4. Extract Phase Angle (from scratch using real and imaginary components)
        # torch.angle returns values in radians between [-pi, pi]
        phase_angle = torch.angle(fft_shifted)
        
        # 5. Individually normalize both representations to [0, 1]
        norm_magnitude = self._normalize_spectrum(log_magnitude)
        norm_phase = self._normalize_spectrum(phase_angle)
        
        # 6. Interleave channels to build a clean dual-channel feature map
        # Expected output order for an RGB frame: [R_mag, R_phase, G_mag, G_phase, B_mag, B_phase]
        dual_channel_output = torch.empty(b, c * 2, h, w, device=x.device, dtype=x.dtype)
        dual_channel_output[:, 0::2, :, :] = norm_magnitude
        dual_channel_output[:, 1::2, :, :] = norm_phase
        
        return dual_channel_output

# ==========================================
# Operational Test and Output Verification
# ==========================================
if __name__ == "__main__":
    # Simulate a single-channel Grayscale Interframe Temporal Residual (e.g., extracted causally)
    # Batch Size = 2, Channels = 1, Height = 128, Width = 128
    simulated_residual = torch.rand(2, 1, 128, 128)
    
    # Initialize the extractor
    extractor = DualChannelFourierExtractor()
    
    # Execute extraction
    with torch.no_grad():
        spatio_fourier_features = extractor(simulated_residual)
        
    print("--- Spectrum Extraction Pipeline ---")
    print(f"Input Shape (Temporal Residual): {simulated_residual.shape}")
    print(f"Output Feature Map Shape:         {spatio_fourier_features.shape}")
    
    # Verify exact splitting of output channels
    mag_channel = spatio_fourier_features[:, 0, :, :]
    phase_channel = spatio_fourier_features[:, 1, :, :]
    
    print("\n--- Value Range Normalization Integrity Check ---")
    print(f"Magnitude Channel -> Min: {mag_channel.min().item():.4f}, Max: {mag_channel.max().item():.4f}")
    print(f"Phase Channel     -> Min: {phase_channel.min().item():.4f}, Max: {phase_channel.max().item():.4f}")
    
    # Assert sanity check
    assert spatio_fourier_features.shape[1] == simulated_residual.shape[1] * 2, "Channel stacking failure!"
    print("\n[Success] Dual-channel feature tensor generated successfully. Ready for sequential Causal TCN blocks.")
