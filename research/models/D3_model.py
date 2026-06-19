"""
D3+ Model Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import (
    CLIPVisionModel, 
    XCLIPVisionModel, 
    AutoModel,
    ViTModel
)
import torchvision.models as models


class D3Model(nn.Module):
    """
    D3 (Detection by Difference of Differences) Model.
    """
    
    SUPPORTED_ENCODERS = [
        'CLIP-16', 'CLIP-32',
        'XCLIP-16', 'XCLIP-32',
        'DINO-base', 'DINO-large',
        'ResNet-18', 'VGG-16',
        'EfficientNet-b4', 'MobileNet-v3'
    ]
    
    def __init__(self, encoder_type: str = 'XCLIP-16', loss_type: str = 'l2'):
        """
        Initialize D3 model.
        
        Args:
            encoder_type: Type of encoder to use
            loss_type: 'l2' or 'cos' for distance computation
        """
        super(D3Model, self).__init__()
        self.loss_type = loss_type
        self.encoder_type = encoder_type
        
        # Initialize encoder based on type
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
            resnet = models.resnet18(pretrained=True)
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        elif encoder_type == 'VGG-16':
            vgg = models.vgg16(pretrained=True)
            self.encoder = nn.Sequential(*list(vgg.children())[:-1])
        elif encoder_type == 'EfficientNet-b4':
            effnet = models.efficientnet_b4(pretrained=True)
            self.encoder = nn.Sequential(*list(effnet.children())[:-1])
        elif encoder_type == 'MobileNet-v3':
            mobilenet = timm.create_model('mobilenetv3_large_100', pretrained=True)
            self.encoder = nn.Sequential(*list(mobilenet.children())[:-1])
        else:
            raise ValueError(f"Unsupported encoder: {encoder_type}")
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, frames, channels, height, width)
        
        Returns:
            features: Extracted features
            dis_2nd_avg: Average of second-order differences
            dis_2nd_std: Standard deviation of second-order differences
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
            features = features.view(features.size(0), -1)
        
        # Reshape to (batch, frames, features)
        features = features.reshape(b, t, -1)
        
        # Compute first-order differences
        vec1 = features[:, :-1, :]  # (batch, n-1, features)
        vec2 = features[:, 1:, :]   # (batch, n-1, features)
        
        if self.loss_type == 'cos':
            dis_1st = F.cosine_similarity(vec1, vec2, dim=-1)  # (batch, n-1)
        else:  # l2
            dis_1st = torch.norm(vec1 - vec2, p=2, dim=-1)     # (batch, n-1)
        
        # Compute second-order differences
        dis_2nd = dis_1st[:, 1:] - dis_1st[:, :-1]  # (batch, n-2)
        
        # Aggregate
        dis_2nd_avg = torch.mean(dis_2nd, dim=1)    # (batch)
        dis_2nd_std = torch.std(dis_2nd, dim=1)     # (batch)
        
        return features, dis_2nd_avg, dis_2nd_std
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict score for video."""
        _, _, score = self.forward(x)
        return score