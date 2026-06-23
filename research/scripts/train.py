"""
scripts/train_fast.py
FAST training using pre-extracted features.
No GPU needed! Runs in minutes.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def train_on_features(file_path="full_features_with_gifs.csv", output_dir="trained_models"):
    """
    Train classifiers on pre-extracted features.
    """
    print("="*60)
    print("D3+ FAST TRAINING")
    print("="*60)
    
    # Load features
    print(f"\n📂 Loading features from: {file_path}")
    data = pd.read_csv(file_path)
    
    # Check for NaN values
    nan_count = data.isna().sum().sum()
    print(f"   Total NaN values in dataset: {nan_count}")
    
    if nan_count > 0:
        print("   ⚠️ NaN values detected! Will impute with median.")
    
    X = data.drop('label', axis=1)
    y = data['label']
    
    print(f"   Samples: {X.shape[0]}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {np.unique(y)}")
    print(f"   Real (0): {np.sum(y == 0)}, Fake (1): {np.sum(y == 1)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # ============================================================
    # Impute missing values with MEDIAN (robust to outliers)
    # ============================================================
    print("\n🔧 Imputing missing values with median...")
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features for SVM and Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    print(f"   Features per sample: {X_train_scaled.shape[1]}")
    print(f"   Imputation complete! ✅")
    
    # ============================================================
    # Random Forest
    # ============================================================
    print("\n" + "="*50)
    print("🌲 Training Random Forest...")
    print("="*50)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)
    print(f"   CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"   Test Accuracy: {acc_rf:.4f}")
    
    # Feature importance (Random Forest only)
    importance = rf.feature_importances_
    top_indices = np.argsort(importance)[-10:][::-1]
    print(f"   Top 5 features: {top_indices[:5]}")
    
    # ============================================================
    # SVM
    # ============================================================
    print("\n" + "="*50)
    print("⚡ Training SVM...")
    print("="*50)
    
    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced',
        random_state=42,
        probability=True
    )
    
    cv_scores_svm = cross_val_score(svm, X_train_scaled, y_train, cv=5)
    print(f"   CV Accuracy: {cv_scores_svm.mean():.4f} ± {cv_scores_svm.std():.4f}")
    
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"   Test Accuracy: {acc_svm:.4f}")
    
    # ============================================================
    # Logistic Regression (Baseline)
    # ============================================================
    print("\n" + "="*50)
    print("📊 Training Logistic Regression...")
    print("="*50)
    
    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
    cv_scores_lr = cross_val_score(lr, X_train_scaled, y_train, cv=5)
    print(f"   CV Accuracy: {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")
    
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"   Test Accuracy: {acc_lr:.4f}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*50)
    print("📊 SUMMARY")
    print("="*50)
    print(f"   Random Forest Test Accuracy: {acc_rf:.4f}")
    print(f"   SVM Test Accuracy: {acc_svm:.4f}")
    print(f"   Logistic Regression Test Accuracy: {acc_lr:.4f}")
    
    # Best model
    best_name = max([('RF', acc_rf), ('SVM', acc_svm), ('LR', acc_lr)], key=lambda x: x[1])
    print(f"\n🏆 Best Model: {best_name[0]} with {best_name[1]:.4f} accuracy")
    
    # ============================================================
    # Save Models, Imputer, and Scaler
    # ============================================================
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(rf, output_dir / "random_forest_model.pkl")
    joblib.dump(svm, output_dir / "svm_model.pkl")
    joblib.dump(lr, output_dir / "logistic_regression_model.pkl")
    joblib.dump(imputer, output_dir / "imputer.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")
    
    # Save feature names
    feature_names = X.columns.tolist()
    np.save(output_dir / "feature_names.npy", feature_names)
    
    print(f"\n✅ Models saved to {output_dir}/")
    
    # ============================================================
    # Save Results
    # ============================================================
    with open(output_dir / "fast_training_results.txt", 'w') as f:
        f.write("D3+ Feature Extraction Results\n")
        f.write("="*50 + "\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Features per sample: {X.shape[1]}\n")
        f.write(f"Real: {np.sum(y == 0)}, Fake: {np.sum(y == 1)}\n\n")
        f.write(f"Random Forest Test Accuracy: {acc_rf:.4f}\n")
        f.write(f"SVM Test Accuracy: {acc_svm:.4f}\n")
        f.write(f"Logistic Regression Test Accuracy: {acc_lr:.4f}\n")
        f.write(f"\nBest Model: {best_name[0]} ({best_name[1]:.4f})\n")
    
    # ============================================================
    # Confusion Matrices
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (name, y_pred, acc) in zip(
        axes, 
        [('RF', y_pred_rf, acc_rf), ('SVM', y_pred_svm, acc_svm), ('LR', y_pred_lr, acc_lr)]
    ):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Real', 'Fake'],
                    yticklabels=['Real', 'Fake'])
        ax.set_title(f'{name}\nAccuracy: {acc:.4f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=150)
    plt.close()
    print(f"✅ Confusion matrices saved to {output_dir}/confusion_matrices.png")
    
    # ============================================================
    # Feature Importance Plot (Random Forest)
    # ============================================================
    if acc_rf > 0.5:
        plt.figure(figsize=(10, 8))
        top_k = min(20, len(importance))
        top_indices = np.argsort(importance)[-top_k:]
        top_names = [feature_names[i] for i in top_indices]
        top_importance = importance[top_indices]
        
        plt.barh(range(top_k), top_importance, color='steelblue')
        plt.yticks(range(top_k), top_names)
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features (Random Forest)')
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=150)
        plt.close()
        print(f"✅ Feature importance plot saved to {output_dir}/feature_importance.png")
    
    print("\n✅ Training complete!")
    
    return {'rf': acc_rf, 'svm': acc_svm, 'lr': acc_lr, 'best': best_name[0], 'best_acc': best_name[1]}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='full_features_with_gifs.csv',
                       help='Path to features file')
    parser.add_argument('--output', type=str, default='trained_models',
                       help='Output directory')
    
    args = parser.parse_args()
    
    train_on_features(
        file_path=args.features,
        output_dir=args.output
    )