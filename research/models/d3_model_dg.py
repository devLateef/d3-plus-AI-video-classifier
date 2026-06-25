"""
research/models/d3_model_dg.py
D3 Model with Domain Generalization (SWA-DAL style).
Adds support for:
- Feature extraction for adversarial training
- Domain classifier integration
- Self-regularization with strong/weak augmentations
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
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loss import get_sr_loss, get_ffvit_loss
from lossZoo import ConsistencyCos, adv, adv_local, im

warnings.filterwarnings("ignore")


class D3ModelDG(nn.Module):
    """
    D3 Model with Domain Generalization capabilities.
    Extends the original D3 model with adversarial and self-regularization losses.
    """
    
    def __init__(self, 
                 encoder_type: str = 'XCLIP-16', 
                 loss_type: str = 'l2',
                 num_classes: int = 2,
                 domain_classifier_hidden: int = 256,
                 sr_epsilon: float = 0.4,
                 sr_loss_p: float = 0.5,
                 sr_alpha: float = 0.3):
        super(D3ModelDG, self).__init__()
        self.loss_type = loss_type
        self.encoder_type = encoder_type
        self.num_classes = num_classes
        
        # === Original D3 encoder (frozen) ===
        self._init_encoder()
        
        # FREEZE encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # === Trainable D3 classifier (original) ===
        self.classifier = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)  # Binary classification
        )
        
        # === Domain Classifier (for adversarial training) ===
        self.domain_classifier = nn.Sequential(
            nn.Linear(self._get_feature_dim(), domain_classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(domain_classifier_hidden, domain_classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(domain_classifier_hidden, 1),  # Binary domain prediction
            nn.Sigmoid()
        )
        
        # === Self-regularization parameters ===
        self.sr_epsilon = sr_epsilon
        self.sr_loss_p = sr_loss_p
        self.sr_alpha = sr_alpha
        
        # === Loss functions from SWA-DAL ===
        self.cos_loss = ConsistencyCos()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model: {total_params:,} params ({trainable_params:,} trainable)")
    
    def _get_feature_dim(self) -> int:
        """Get the feature dimension from the encoder."""
        if self.encoder_type in ['CLIP-16', 'CLIP-32', 'XCLIP-16', 'XCLIP-32']:
            return 768
        elif self.encoder_type in ['DINO-base', 'DINO-large']:
            return 768
        else:
            return 512
    
    def _init_encoder(self):
        """Initialize encoder based on type."""
        if self.encoder_type == 'CLIP-16':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        elif self.encoder_type == 'CLIP-32':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        elif self.encoder_type == 'XCLIP-16':
            self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16")
        elif self.encoder_type == 'XCLIP-32':
            self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")
        elif self.encoder_type == 'DINO-base':
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-base")
        elif self.encoder_type == 'DINO-large':
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-large")
        elif self.encoder_type == 'ResNet-18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        elif self.encoder_type == 'VGG-16':
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(vgg.children())[:-1])
        elif self.encoder_type == 'EfficientNet-b4':
            effnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(effnet.children())[:-1])
        elif self.encoder_type == 'MobileNet-v3':
            mobilenet = timm.create_model('mobilenetv3_large_100', pretrained=True)
            self.encoder = nn.Sequential(*list(mobilenet.children())[:-1])
        else:
            raise ValueError(f"Unsupported encoder: {self.encoder_type}")
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from encoder."""
        b, t, c, h, w = x.shape
        images = x.reshape(-1, c, h, w)
        
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
        return features
    
    def compute_d3_scores(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute D3 scores from features."""
        b, t, _ = features.shape
        device = features.device
        
        if t < 2:
            return features, torch.zeros(b, device=device), torch.zeros(b, device=device)
        
        # First-order differences
        vec1 = features[:, :-1, :]
        vec2 = features[:, 1:, :]
        
        if self.loss_type == 'cos':
            dis_1st = F.cosine_similarity(vec1, vec2, dim=-1)
        else:
            dis_1st = torch.norm(vec1 - vec2, p=2, dim=-1)
        
        # Second-order differences
        if dis_1st.size(1) < 2:
            return features, torch.zeros(b, device=device), torch.zeros(b, device=device)
        
        dis_2nd = dis_1st[:, 1:] - dis_1st[:, :-1]
        dis_2nd_avg = torch.mean(dis_2nd, dim=1)
        dis_2nd_std = torch.std(dis_2nd, dim=1)
        
        # D3 score (trainable)
        combined = torch.stack([dis_2nd_avg, dis_2nd_std], dim=1)
        score = self.classifier(combined).squeeze(1)
        
        return features, dis_2nd_avg, score
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> tuple:
        """
        Forward pass.
        
        Returns:
            features: Frame features
            dis_2nd_avg: Average difference (for analysis)
            score: Prediction score
            (optional) d3_features: D3 features for adversarial training
        """
        features = self.forward_features(x)
        d3_features, dis_2nd_avg, score = self.compute_d3_scores(features)
        
        if return_features:
            return features, dis_2nd_avg, score, d3_features
        
        return features, dis_2nd_avg, score
    
    def compute_domain_loss(self, features_source: torch.Tensor, 
                           features_target: torch.Tensor) -> torch.Tensor:
        """
        Compute domain adversarial loss.
        """
        # Concatenate features from both domains
        combined_features = torch.cat([features_source, features_target], dim=0)
        
        # Get domain predictions
        domain_logits = self.domain_classifier(combined_features)
        
        # Create domain labels
        batch_size = features_source.size(0)
        domain_labels = torch.cat([
            torch.ones(batch_size, 1),  # Source domain = 1
            torch.zeros(batch_size, 1)  # Target domain = 0
        ]).to(combined_features.device)
        
        # Binary cross-entropy loss
        domain_loss = F.binary_cross_entropy(domain_logits, domain_labels)
        
        return domain_loss
    
    def compute_sr_loss(self, logits_weak: torch.Tensor, 
                       logits_strong: torch.Tensor) -> torch.Tensor:
        """
        Compute self-regularization loss between weak and strong augmentations.
        """
        return get_sr_loss(logits_weak, logits_strong, 
                          self.sr_epsilon, self.sr_loss_p)
    
    def compute_ffvit_loss(self, logits_weak: torch.Tensor,
                          logits_strong: torch.Tensor,
                          logits_fused: torch.Tensor) -> torch.Tensor:
        """
        Compute Feature Fusion ViT loss.
        """
        return get_ffvit_loss(logits_weak, logits_strong, logits_fused,
                             self.sr_epsilon, self.sr_loss_p, self.sr_alpha)