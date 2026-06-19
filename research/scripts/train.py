"""
research/scripts/train.py
Updated training with early stopping and model saving.
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
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_d3_model(
    csv_path: Path,
    model_save_path: Path = Path("trained_models/d3_plus_model.pth"),
    encoder_type: str = 'XCLIP-16',
    loss_type: str = 'l2',
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    epochs: int = 50,
    val_split: float = 0.2,
    patience: int = 10,
    max_samples: int = 9999999,
    device: str = 'cuda'
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"CSV: {csv_path}")
    
    # Load dataset
    dataset = D3Dataset(csv_path=csv_path, max_samples=max_samples)
    
    # Split
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model
    model = D3Model(encoder_type=encoder_type, loss_type=loss_type).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    criterion = nn.MSELoss()
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            frames = frames.to(device)
            labels = labels.float().to(device)
            
            _, _, scores = model(frames)
            loss = criterion(scores, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (scores > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)
        
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
                labels = labels.float().to(device)
                
                _, _, scores = model(frames)
                loss = criterion(scores, labels)
                val_loss += loss.item()
                
                preds = (scores > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
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
    
    # Save training history
    history_path = model_save_path.parent / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
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
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=10)
    
    args = parser.parse_args()
    
    train_d3_model(
        csv_path=args.csv,
        model_save_path=args.output,
        encoder_type=args.encoder,
        loss_type=args.loss,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience
    )