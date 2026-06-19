"""
research/scripts/train.py
WORKING VERSION: Proper training loop.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from research.data.dataset import D3Dataset
from research.models.d3_model import D3Model


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """Stack frames and labels."""
    frames_list = []
    labels_list = []
    
    for frames, label in batch:
        frames_list.append(frames)
        labels_list.append(label)
    
    frames_batch = torch.stack(frames_list)
    return frames_batch, torch.tensor(labels_list, dtype=torch.float32)


def train_d3_model(
    csv_path: Path,
    model_save_path: Path = Path("trained_models/d3_plus_model.pth"),
    encoder_type: str = 'XCLIP-16',
    loss_type: str = 'l2',
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    epochs: int = 50,
    val_split: float = 0.2,
    patience: int = 10,
    max_samples: int = 9999999
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"CSV: {csv_path}")
    
    # Load dataset
    dataset = D3Dataset(csv_path=csv_path, max_samples=max_samples)
    
    # Split
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = D3Model(encoder_type=encoder_type, loss_type=loss_type).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=learning_rate,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # Training
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for frames, labels in pbar:
            frames = frames.to(device)
            labels = labels.to(device)
            
            _, _, score = model(frames)
            loss = criterion(score, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(score) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                frames = frames.to(device)
                labels = labels.to(device)
                
                _, _, score = model(frames)
                loss = criterion(score, labels)
                
                val_loss += loss.item()
                preds = (torch.sigmoid(score) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
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
    config_path = model_save_path.parent / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'encoder_type': encoder_type,
            'loss_type': loss_type,
            'best_val_loss': best_val_loss,
            'best_val_acc': max(history['val_acc']),
            'epochs_trained': len(history['train_loss'])
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
    parser.add_argument('--output', type=Path, default=Path('trained_models/d3_plus_model.pth'))
    parser.add_argument('--encoder', type=str, default='XCLIP-16')
    parser.add_argument('--loss', type=str, default='l2')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max-samples', type=int, default=9999999)
    
    args = parser.parse_args()
    
    train_d3_model(
        csv_path=args.csv,
        model_save_path=args.output,
        encoder_type=args.encoder,
        loss_type=args.loss,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        max_samples=args.max_samples
    )