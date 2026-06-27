import torch
import torch.nn as nn

class DCTvsFFTExtractor(nn.Module):
    """
    Compares the mathematical output layout of DCT and FFT.
    """
    def __init__(self):
        super(DCTvsFFTExtractor, self).__init__()

    def compute_dct_2d(self, x):
        """
        Computes a Type-II 2D DCT from scratch using an FFT matrix trick.
        Outputs purely real numbers.
        """
        b, c, h, w = x.shape
        # Pad and mirror the input to simulate pure cosine boundaries
        x_padded = torch.cat([x, x.flip(-1)], dim=-1)
        x_padded = torch.cat([x_padded, x_padded.flip(-2)], dim=-2)
        
        fft_complex = torch.fft.fft2(x_padded, dim=(-2, -1))
        # Extract the real part and crop back to original size
        dct_out = fft_complex.real[:, :, :h, :w]
        return dct_out

    def forward(self, x):
        # 1. DCT Pipeline (Purely Real)
        dct_features = self.compute_dct_2d(x)
        
        # 2. FFT Pipeline (Complex -> Requires explicit real/imag separation)
        fft_complex = torch.fft.fft2(x, dim=(-2, -1))
        
        return dct_features, fft_complex

# Quick structural shape check
sample_frame = torch.rand(1, 1, 64, 64) # Grayscale frame
extractor = DCTvsFFTExtractor()
dct_res, fft_res = extractor(sample_frame)

print(f"DCT Feature Output Tensor Type: {dct_res.dtype} -> Shape: {dct_res.shape}")
print(f"FFT Feature Output Tensor Type: {fft_res.dtype} -> Shape: {fft_res.shape}")
