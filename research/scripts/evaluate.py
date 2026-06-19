"""
Evaluation script for D3+ model.
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score

from research.data.dataset import D3Dataset
from research.models.d3_model import D3Model


def evaluate_d3_model(
    model_path: Path,
    real_dir: Optional[Path] = None,
    fake_dir: Optional[Path] = None,
    real_csv: Optional[Path] = None,
    fake_csv: Optional[Path] = None,
    encoder_type: str = 'XCLIP-16',
    loss_type: str = 'l2',
    batch_size: int = 8,
    device: str = 'cuda',
    max_samples: int = 1000
) -> dict:
    """
    Evaluate D3+ model.
    
    Returns:
        Dictionary of metrics
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = D3Dataset(
        real_dir=real_dir,
        fake_dir=fake_dir,
        real_csv=real_csv,
        fake_csv=fake_csv,
        max_samples=max_samples
    )
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Evaluation samples: {len(dataset)}")
    
    # Load model
    model = D3Model(encoder_type=encoder_type, loss_type=loss_type).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Evaluation
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Evaluating"):
            frames = frames.to(device)
            labels = labels.numpy()
            
            _, _, scores = model(frames)
            
            y_true.extend(labels)
            y_pred.extend(scores.cpu().numpy().flatten())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute metrics
    ap_score = average_precision_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_true, pred_binary)
    
    results = {
        'ap_score': ap_score,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'n_samples': len(y_true),
        'n_real': int(np.sum(y_true == 0)),
        'n_fake': int(np.sum(y_true == 1))
    }
    
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"AP Score: {ap_score:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Samples: {len(y_true)} (Real: {results['n_real']}, Fake: {results['n_fake']})")
    print("="*50)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate D3+ model')
    parser.add_argument('--model', type=Path, required=True, help='Model path')
    parser.add_argument('--real-dir', type=Path, help='Directory with real videos')
    parser.add_argument('--fake-dir', type=Path, help='Directory with fake videos')
    parser.add_argument('--real-csv', type=Path, help='CSV with real videos')
    parser.add_argument('--fake-csv', type=Path, help='CSV with fake videos')
    parser.add_argument('--encoder', type=str, default='XCLIP-16')
    parser.add_argument('--loss', type=str, default='l2')
    parser.add_argument('--max-samples', type=int, default=1000)
    
    args = parser.parse_args()
    
    evaluate_d3_model(
        model_path=args.model,
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        real_csv=args.real_csv,
        fake_csv=args.fake_csv,
        encoder_type=args.encoder,
        loss_type=args.loss,
        max_samples=args.max_samples
    )