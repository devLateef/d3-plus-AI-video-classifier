"""
research/models/d3_model.py
WORKING VERSION: Uses CLIPVisionModel directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import warnings
warnings.filterwarnings('ignore')

from transformers import CLIPVisionModel


class D3Model(nn.Module):
    """
    D3 Model using CLIPVisionModel (no AutoModel issues).
    """
    
    def __init__(self, encoder_type: str = 'CLIP-16', loss_type: str = 'l2'):
        super(D3Model, self).__init__()
        self.loss_type = loss_type
        self.encoder_type = encoder_type
        
        # Initialize CLIPVisionModel directly
        if encoder_type == 'CLIP-16':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        elif encoder_type == 'CLIP-32':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        elif encoder_type == 'ResNet-18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        else:
            # Default to CLIP-16
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        
        # FREEZE encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Classifier head (trainable)
        self.classifier = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)      
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model: {total_params:,} params ({trainable_params:,} trainable)")
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass with proper gradient flow.
        """
        b, t, c, h, w = x.shape
        
        # Reshape for encoder
        images = x.reshape(-1, c, h, w)
        
        # Extract features using CLIPVisionModel
        if self.encoder_type.startswith('CLIP'):
            outputs = self.encoder(images, output_hidden_states=True)
            features = outputs.pooler_output
        else:
            # ResNet
            features = self.encoder(images)
            features = features.view(features.size(0), -1)
        
        # Reshape to (batch, frames, features)
        features = features.reshape(b, t, -1)
        
        # If not enough frames, return zeros
        if t < 2:
            score = torch.zeros(b, device=x.device)
            return features, torch.zeros(b, device=x.device), score
        
        # First-order differences
        vec1 = features[:, :-1, :]
        vec2 = features[:, 1:, :]
        
        if self.loss_type == 'cos':
            dis_1st = F.cosine_similarity(vec1, vec2, dim=-1)
        else:
            dis_1st = torch.norm(vec1 - vec2, p=2, dim=-1)
        
        # Second-order differences
        if dis_1st.size(1) < 2:
            score = torch.zeros(b, device=x.device)
            return features, torch.zeros(b, device=x.device), score
        
        dis_2nd = dis_1st[:, 1:] - dis_1st[:, :-1]
        
        # Aggregate
        dis_2nd_avg = torch.mean(dis_2nd, dim=1)
        dis_2nd_std = torch.std(dis_2nd, dim=1)
        
        combined = torch.stack([dis_2nd_avg, dis_2nd_std], dim=1)
        score = self.classifier(combined).squeeze(1)
        
        return features, dis_2nd_avg, score