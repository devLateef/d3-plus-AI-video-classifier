"""
research/models/d3_model.py
FIXED: Proper classifier input size matching combined features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import (
    CLIPVisionModel, 
    XCLIPVisionModel, 
    AutoModel,
)
import torchvision.models as models
import warnings
warnings.filterwarnings("ignore")


class D3Model(nn.Module):
    """
    D3 Model with proper gradient flow.
    """
    
    def __init__(self, encoder_type: str = 'XCLIP-16', loss_type: str = 'l2'):
        super(D3Model, self).__init__()
        self.loss_type = loss_type
        self.encoder_type = encoder_type
        
        # Initialize encoder
        if encoder_type == 'CLIP-16':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        elif encoder_type == 'CLIP-32':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        elif encoder_type == 'XCLIP-16':
            self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16")
        elif encoder_type == 'XCLIP-32':
            self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")
        elif encoder_type == 'DINO-base':
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-base")
        elif encoder_type == 'DINO-large':
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-large")
        elif encoder_type == 'ResNet-18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        elif encoder_type == 'VGG-16':
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(vgg.children())[:-1])
        elif encoder_type == 'EfficientNet-b4':
            effnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(effnet.children())[:-1])
        elif encoder_type == 'MobileNet-v3':
            mobilenet = timm.create_model('mobilenetv3_large_100', pretrained=True)
            self.encoder = nn.Sequential(*list(mobilenet.children())[:-1])
        else:
            raise ValueError(f"Unsupported encoder: {encoder_type}")
        
        # FREEZE encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # ============================================================
        # FIXED: Classifier for combined features (batch, 2)
        # ============================================================
        # combined = [dis_2nd_avg, dis_2nd_std] -> shape (batch, 2)
        self.classifier = nn.Sequential(
            nn.Linear(2, 32),      # Input: 2 features from D3
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)       # Output: single score
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model: {total_params:,} params ({trainable_params:,} trainable)")
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass with proper gradient flow.
        
        Returns:
            features: Frame features (for analysis)
            dis_2nd_avg: Average difference (for analysis)
            score: Prediction score (trainable)
        """
        b, t, c, h, w = x.shape
        
        # Reshape for encoder
        images = x.reshape(-1, c, h, w)
        
        # Extract features
        if self.encoder_type in ['CLIP-16', 'CLIP-32', 'XCLIP-16', 'XCLIP-32']:
            outputs = self.encoder(images, output_hidden_states=True)
            features = outputs.pooler_output
        elif self.encoder_type in ['DINO-base', 'DINO-large']:
            outputs = self.encoder(images)
            features = outputs.pooler_output
        else:
            features = self.encoder(images)
            if features.dim() > 2:
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
        
        # Aggregate - these maintain gradient flow
        dis_2nd_avg = torch.mean(dis_2nd, dim=1)
        dis_2nd_std = torch.std(dis_2nd, dim=1)
        
        # Stack as (batch, 2) - THIS IS THE CORRECT SHAPE
        combined = torch.stack([dis_2nd_avg, dis_2nd_std], dim=1)
        
        # Classify
        score = self.classifier(combined).squeeze(1)
        
        return features, dis_2nd_avg, score