"""
research/scripts/train_dg.py
Domain Generalization training with SWA-DAL losses.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.data.dataset import D3Dataset
from research.models.d3_model_dg import D3ModelDG
from loss import get_sr_loss, get_ffvit_loss
from lossZoo import ConsistencyCos, adv, im


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_video_augmentation(frames, aug_type='weak'):
    """
    Apply consistent augmentation to all frames in a video clip.
    
    Args:
        frames: Tensor of shape (frames, channels, height, width)
        aug_type: 'weak' or 'strong'
    
    Returns:
        Augmented frames tensor
    """
    import albumentations as A
    import cv2
    
    if aug_type == 'weak':
        # Weak augmentation: slight rotation, scaling, and color jitter
        aug = A.Compose([
            A.Rotate(limit=5, p=0.5),
            A.RandomScale(scale_limit=0.05, p=0.3),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.5),
        ])
    else:
        # Strong augmentation: more aggressive transformations
        aug = A.Compose([
            A.Rotate(limit=15, p=0.7),
            A.RandomScale(scale_limit=0.15, p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.7),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ])
    
    # Convert to numpy for augmentation
    frames_np = frames.cpu().numpy()
    augmented = []
    
    for frame in frames_np:
        # Transpose from (C, H, W) to (H, W, C) for albumentations
        frame_hwc = frame.transpose(1, 2, 0)
        # Convert to uint8 if needed
        if frame_hwc.max() <= 1.0:
            frame_hwc = (frame_hwc * 255).astype(np.uint8)
        else:
            frame_hwc = frame_hwc.astype(np.uint8)
        
        augmented_frame = aug(image=frame_hwc)['image']
        # Convert back to (C, H, W) and normalize
        augmented_frame = augmented_frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        augmented.append(augmented_frame)
    
    return torch.FloatTensor(np.stack(augmented))


class DGAugmentationWrapper:
    """
    Wrapper to apply weak and strong augmentations to video clips.
    """
    def __init__(self):
        self.weak_aug = None
        self.strong_aug = None
    
    def __call__(self, frames, apply_weak=True, apply_strong=True):
        frames_weak = apply_video_augmentation(frames, 'weak') if apply_weak else frames
        frames_strong = apply_video_augmentation(frames, 'strong') if apply_strong else frames
        return frames_weak, frames_strong


def train_d3_model_dg(
    csv_path: Path,
    model_save_path: Path = Path("trained_models/d3_plus_model_dg.pth"),
    encoder_type: str = 'XCLIP-16',
    loss_type: str = 'l2',
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    epochs: int = 50,
    val_split: float = 0.2,
    patience: int = 10,
    max_samples: int = 9999999,
    device: str = 'cuda',
    domain_loss_weight: float = 0.1,
    sr_loss_weight: float = 0.5,
    ffvit_loss_weight: float = 0.3
):
    """
    Train D3 model with domain generalization (SWA-DAL style).
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"CSV: {csv_path}")
    
    # Load dataset
    dataset = D3Dataset(csv_path=csv_path, max_samples=max_samples)
    
    # Create domain labels (assuming CSV has a 'domain' column, or use generator info)
    # If not, we can use the label as a proxy for domain (real=0, fake=1)
    # For better domain generalization, you'd want more fine-grained domains
    if 'generator' in dataset.df.columns:
        # Use generator as domain
        unique_generators = dataset.df['generator'].unique()
        generator_to_domain = {g: i for i, g in enumerate(unique_generators)}
        dataset.df['domain'] = dataset.df['generator'].map(generator_to_domain)
    else:
        # Use label as domain (real=0, fake=1)
        dataset.df['domain'] = dataset.df['label']
    
    # Split
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = D3ModelDG(
        encoder_type=encoder_type, 
        loss_type=loss_type,
        num_classes=2
    ).to(device)
    
    # Optimizer - only trainable parameters (encoder is frozen)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=learning_rate,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Loss functions
    classifier_criterion = nn.BCEWithLogitsLoss()
    domain_criterion = nn.BCELoss()
    
    # Cosine consistency loss
    cos_loss = ConsistencyCos()
    
    # Augmentation wrapper
    aug_wrapper = DGAugmentationWrapper()
    
    # Training
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [],
        'domain_loss': [], 'sr_loss': [], 'ffvit_loss': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_domain_loss = 0.0
        epoch_sr_loss = 0.0
        epoch_ffvit_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for frames, labels in pbar:
            frames = frames.to(device)
            labels = labels.float().to(device)
            
            # Apply weak and strong augmentations
            frames_weak, frames_strong = aug_wrapper(frames)
            frames_weak = frames_weak.to(device)
            frames_strong = frames_strong.to(device)
            
            # Forward pass for weak and strong augmentations
            _, _, score_weak, d3_features_weak = model(frames_weak, return_features=True)
            _, _, score_strong, d3_features_strong = model(frames_strong, return_features=True)
            
            # === 1. Classification Loss ===
            cls_loss = classifier_criterion(score_weak, labels)
            
            # === 2. Self-Regularization Loss ===
            sr_loss = model.compute_sr_loss(score_weak, score_strong)
            epoch_sr_loss += sr_loss.item()
            
            # === 3. Feature Fusion Loss ===
            # Use weak and strong as individual views, fused as combination
            # For simplicity, we use weak as fused representation
            ffvit_loss = model.compute_ffvit_loss(score_weak, score_strong, score_weak)
            epoch_ffvit_loss += ffvit_loss.item()
            
            # === 4. Domain Adversarial Loss ===
            # Use domain labels from dataset
            # For simplicity, use label as domain (0=real, 1=fake)
            domain_loss = model.compute_domain_loss(d3_features_weak, d3_features_strong)
            epoch_domain_loss += domain_loss.item()
            
            # Total loss
            loss = cls_loss + sr_loss_weight * sr_loss + ffvit_loss_weight * ffvit_loss + domain_loss_weight * domain_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            preds = (torch.sigmoid(score_weak) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)
            
            pbar.set_postfix({
                'loss': loss.item(),
                'cls': cls_loss.item(),
                'sr': sr_loss.item(),
                'ffvit': ffvit_loss.item(),
                'dom': domain_loss.item()
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['domain_loss'].append(epoch_domain_loss / len(train_loader))
        history['sr_loss'].append(epoch_sr_loss / len(train_loader))
        history['ffvit_loss'].append(epoch_ffvit_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                frames = frames.to(device)
                labels = labels.float().to(device)
                
                _, _, score = model(frames)
                loss = classifier_criterion(score, labels)
                
                val_loss += loss.item()
                preds = (torch.sigmoid(score) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Domain Loss: {epoch_domain_loss/len(train_loader):.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
            print(f"  ✅ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹ Early stopping at epoch {epoch+1}")
                break
    
    # Save config
    config_path = model_save_path.parent / "model_config_dg.json"
    with open(config_path, 'w') as f:
        json.dump({
            'encoder_type': encoder_type,
            'loss_type': loss_type,
            'best_val_loss': best_val_loss,
            'best_val_acc': max(history['val_acc']),
            'epochs_trained': len(history['train_loss']),
            'domain_loss_weight': domain_loss_weight,
            'sr_loss_weight': sr_loss_weight,
            'ffvit_loss_weight': ffvit_loss_weight
        }, f, indent=2)
    
    print(f"✅ Training complete! Best val loss: {best_val_loss:.4f}")
    
    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'best_val_acc': max(history['val_acc'])
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('trained_models/d3_plus_model_dg.pth'))
    parser.add_argument('--encoder', type=str, default='XCLIP-16')
    parser.add_argument('--loss', type=str, default='l2')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--domain-weight', type=float, default=0.1)
    parser.add_argument('--sr-weight', type=float, default=0.5)
    parser.add_argument('--ffvit-weight', type=float, default=0.3)
    
    args = parser.parse_args()
    
    train_d3_model_dg(
        csv_path=args.csv,
        model_save_path=args.output,
        encoder_type=args.encoder,
        loss_type=args.loss,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        domain_loss_weight=args.domain_weight,
        sr_loss_weight=args.sr_weight,
        ffvit_loss_weight=args.ffvit_weight
    )