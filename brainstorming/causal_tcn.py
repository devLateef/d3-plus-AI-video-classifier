import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    """
    A 1D Convolutional layer that strictly enforces causality by 
    padding only the past (left side) of the sequence.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        # Calculate padding needed on the left to ensure causality
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, dilation=dilation
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Sequence_Length)
        # Manually pad only the left side of the time dimension
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class CausalDeepfakeDetector(nn.Module):
    """
    A Causal Temporal Model for real-time deepfake classification.
    Processes frame features sequentially without looking ahead.
    """
    def __init__(self, feature_dim=512, hidden_dim=128):
        super(CausalDeepfakeDetector, self).__init__()
        
        # 1. Causal temporal feature refiner (TCN Layer)
        # Processes local temporal dependencies causally
        self.causal_tcn = nn.Sequential(
            CausalConv1d(in_channels=feature_dim, out_channels=hidden_dim, kernel_size=3, dilation=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            CausalConv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 2. Unidirectional Recurrent Layer (Strictly Causal)
        # batch_first=True expects input: (Batch, Sequence, Features)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, 
                          num_layers=1, batch_first=True, bidirectional=False)
        
        # 3. Output classifier (Real vs Fake frame score)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Input shape from preprocessing: (Batch, Sequence_Length, Feature_Dim)
        
        # Reshape for Conv1D: (Batch, Feature_Dim, Sequence_Length)
        x_t = x.transpose(1, 2)
        
        # Apply causal convolutions
        tcn_out = self.causal_tcn(x_t)
        
        # Reshape back for GRU: (Batch, Sequence_Length, Hidden_Dim)
        gru_in = tcn_out.transpose(1, 2)
        
        # Process sequentially through GRU
        gru_out, _ = self.gru(gru_in)
        
        # Generate predictions for every frame up to the current point
        # Output shape: (Batch, Sequence_Length, 1) -> Sigmoid for probability
        predictions = torch.sigmoid(self.classifier(gru_out))
        
        return predictions

# ==========================================
# Simulated Live Streaming Test Pipeline
# ==========================================
if __name__ == "__main__":
    # Simulate a batch of 2 live video streams
    # 30 frames long, where each frame has a 512-dim spatial feature vector
    batch_size = 2
    sequence_length = 30
    feature_dimension = 512
    
    simulated_video_stream = torch.rand(batch_size, sequence_length, feature_dimension)
    
    # Initialize the causal model
    model = CausalDeepfakeDetector(feature_dim=feature_dimension)
    
    # Forward pass
    frame_by_frame_predictions = model(simulated_video_stream)
    
    print("--- Causal Pipeline Execution ---")
    print(f"Input Stream Shape:        {simulated_video_stream.shape} (Batch, Frames, Features)")
    print(f"Output Predictions Shape:  {frame_by_frame_predictions.shape} (Batch, Frames, Real/Fake Score)")
    
    # Verification of Causality: 
    # Altering frame 25 should NOT affect the model's prediction for frame 10.
    stream_altered = simulated_video_stream.clone()
    stream_altered[:, 25:, :] += 10.0 # Inject heavy manipulation far in the future
    
    pred_original = model(simulated_video_stream)
    pred_altered = model(stream_altered)
    
    # Measure difference at frame index 10
    difference_at_frame_10 = torch.abs(pred_original[:, :11] - pred_altered[:, :11]).max().item()
    print(f"Max prediction variance at frame 10 after altering frame 25: {difference_at_frame_10:.6f}")
    
    if difference_at_frame_10 == 0.0:
        print("[Success] Enforced strict temporal causality! Future frames cannot leak into past decisions.")