"""
research/scripts/train_fast.py
FAST training using pre-extracted features.
Runs in minutes, not hours!
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def train_on_features(features_path="full_features.npz", output_dir="trained_models"):
    """
    Train classifiers on pre-extracted features.
    """
    # Load features
    data = np.load(features_path)
    X = data['features']
    y = data['labels']
    
    print(f"Loaded features: {X.shape}")
    print(f"Labels: {np.unique(y)}")
    print(f"Real: {np.sum(y == 0)}, Fake: {np.sum(y == 1)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    print(f"Features per sample: {X.shape[1]}")
    
    # ============================================================
    # Random Forest
    # ============================================================
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Test Accuracy: {acc_rf:.4f}")
    
    # Feature importance
    importance = rf.feature_importances_
    print(f"Top 5 features: {np.argsort(importance)[-5:][::-1]}")
    
    # ============================================================
    # SVM
    # ============================================================
    print("\n" + "="*50)
    print("Training SVM...")
    print("="*50)
    
    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced',
        random_state=42,
        probability=True
    )
    
    # Cross-validation
    cv_scores_svm = cross_val_score(svm, X_train, y_train, cv=5)
    print(f"CV Accuracy: {cv_scores_svm.mean():.4f} ± {cv_scores_svm.std():.4f}")
    
    # Train
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"Test Accuracy: {acc_svm:.4f}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Random Forest Test Accuracy: {acc_rf:.4f}")
    print(f"SVM Test Accuracy: {acc_svm:.4f}")
    
    # Save models
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(rf, output_dir / "random_forest_model.pkl")
    joblib.dump(svm, output_dir / "svm_model.pkl")
    print(f"\n✅ Models saved to {output_dir}/")
    
    # Save results
    with open(output_dir / "fast_training_results.txt", 'w') as f:
        f.write(f"Random Forest Accuracy: {acc_rf:.4f}\n")
        f.write(f"SVM Accuracy: {acc_svm:.4f}\n")
        f.write(f"Features: {X.shape[1]}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, (name, y_pred) in zip(axes, [('RF', y_pred_rf), ('SVM', y_pred_svm)]):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Real', 'Fake'],
                    yticklabels=['Real', 'Fake'])
        ax.set_title(f'{name} - Accuracy: {accuracy_score(y_test, y_pred):.4f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=150)
    plt.close()
    
    print(f"✅ Confusion matrices saved to {output_dir}/confusion_matrices.png")
    
    return {'rf': acc_rf, 'svm': acc_svm}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='full_features.npz',
                       help='Path to features file')
    parser.add_argument('--output', type=str, default='trained_models',
                       help='Output directory')
    
    args = parser.parse_args()
    
    train_on_features(
        features_path=args.features,
        output_dir=args.output
    )