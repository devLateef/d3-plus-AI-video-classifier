"""
scripts/ablation.py
Ablation study with optimized SVM.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC  
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AblationStudy:
    """
    Run ablation study on pre-extracted features.
    """
    
    def __init__(self, csv_path: str, output_dir: str = "results"):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("Loading data...")
        self.data = pd.read_csv(csv_path)
        self.X = self.data.drop('label', axis=1)
        self.y = self.data['label']
        
        # Define feature groups
        self.feature_groups = self._define_feature_groups()
        
        # Define ablation configurations
        self.configs = {
            'Baseline (D3 only)': ['d3_avg', 'd3_std'],
            'D3 + Color': ['d3_avg', 'd3_std'] + self.feature_groups['color'],
            'D3 + Color + Temporal': ['d3_avg', 'd3_std'] + self.feature_groups['color'] + self.feature_groups['temporal'],
            'Full D3+': self.X.columns.tolist()
        }
        
        # Classifiers
        self.classifiers = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'SVM (Linear)': LinearSVC(
                C=10,
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                dual='auto'
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        }
        
        print("="*60)
        print("🔬 ABLATION STUDY (Optimized)")
        print("="*60)
        print(f"Total samples: {len(self.X)}")
        print(f"Total features: {len(self.X.columns)}")
        print(f"Configurations: {len(self.configs)}")
        print(f"Classifiers: {len(self.classifiers)}")
        
    def _define_feature_groups(self):
        """Define feature groups based on column names."""
        all_cols = self.X.columns.tolist()
        
        # D3 features
        d3 = ['d3_avg', 'd3_std']
        
        # Color features
        color = [col for col in all_cols if 'frame' in col and ('_mean' in col or '_std' in col or '_skew' in col or '_kurt' in col)]
        
        # Temporal features
        temporal = [col for col in all_cols if col.startswith('temporal_')]
        
        # Bitrate features
        bitrate = [col for col in all_cols if col in ['duration', 'frame_count', 'width', 'height', 'size_mb', 'is_gif']]
        
        return {
            'd3': d3,
            'color': color,
            'temporal': temporal,
            'bitrate': bitrate
        }
    
    def run(self):
        """Run the ablation study."""
        results = {}
        
        for config_name, feature_list in self.configs.items():
            print(f"\n{'='*60}")
            print(f"📊 Testing: {config_name}")
            print(f"   Features: {len(feature_list)}")
            print('='*60)
            
            # Select features
            X_subset = self.X[feature_list]
            
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X_subset)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            results[config_name] = {}
            
            for clf_name, clf in self.classifiers.items():
                print(f"\n  Training {clf_name}...")
                metrics = self._evaluate_classifier(X_scaled, self.y, clf, clf_name)
                results[config_name][clf_name] = metrics
        
        # Save and plot results
        self._save_results(results)
        self._generate_figures(results)
        
        return results
    
    def _evaluate_classifier(self, X, y, clf, clf_name):
        """Evaluate a single classifier with cross-validation."""
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced from 5 to 3
        
        metrics_list = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
            'roc_auc': [], 'ap_score': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx], y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]
            
            # For SVM, use a subset if the dataset is large
            if clf_name == 'SVM (Linear)' and len(X_train) > 10000:
                sample_idx = np.random.choice(len(X_train), 8000, replace=False)
                X_train_small = X_train[sample_idx]
                y_train_small = y_train.iloc[sample_idx] if isinstance(y_train, pd.Series) else y_train[sample_idx]
                clf.fit(X_train_small, y_train_small)
            else:
                clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            
            try:
                y_proba = clf.decision_function(X_test)
                # Convert decision function to probability-like [0,1]
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
            except:
                y_proba = y_pred
            
            metrics_list['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics_list['precision'].append(precision_score(y_test, y_pred))
            metrics_list['recall'].append(recall_score(y_test, y_pred))
            metrics_list['f1'].append(f1_score(y_test, y_pred))
            metrics_list['roc_auc'].append(roc_auc_score(y_test, y_proba))
            metrics_list['ap_score'].append(average_precision_score(y_test, y_proba))
            
            print(f"    Fold {fold+1}/3 - Accuracy: {metrics_list['accuracy'][-1]:.4f}")
        
        mean_metrics = {
            'accuracy': np.mean(metrics_list['accuracy']),
            'precision': np.mean(metrics_list['precision']),
            'recall': np.mean(metrics_list['recall']),
            'f1': np.mean(metrics_list['f1']),
            'roc_auc': np.mean(metrics_list['roc_auc']),
            'ap_score': np.mean(metrics_list['ap_score'])
        }
        
        print(f"\n  {clf_name} Results:")
        for metric, value in mean_metrics.items():
            print(f"    {metric}: {value:.4f}")
        
        return mean_metrics
    
    def _save_results(self, results):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        json_path = self.output_dir / f"ablation_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        report_path = self.output_dir / f"ablation_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("ABLATION STUDY REPORT\n")
            f.write(f"Total samples: {len(self.X)}\n")
            f.write(f"Total features: {len(self.X.columns)}\n\n")
            
            for config_name, config_results in results.items():
                f.write(f"\n{config_name}:\n")
                for clf_name, metrics in config_results.items():
                    f.write(f"\n  {clf_name}:\n")
                    for metric, value in metrics.items():
                        f.write(f"    {metric}: {value:.4f}\n")
        
        print(f"\n Results saved to:\n   {json_path}\n   {report_path}")
    
    def _generate_figures(self, results):
        """Generate ablation figures."""
        self._plot_performance_comparison(results, "ablation_comparison")
        self._plot_feature_contribution(results, "feature_contribution")
        self._plot_confusion_matrices(results, "ablation_confusion")
        print(f"\n Figures saved to {self.output_dir}")
    
    def _plot_performance_comparison(self, results, filename):
        """Plot performance comparison across configurations."""
        configs = list(results.keys())
        metrics = ['accuracy', 'f1', 'roc_auc', 'ap_score']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, clf_name in enumerate(['Random Forest', 'SVM (Linear)', 'Logistic Regression']):
            ax = axes[0] if idx == 0 else axes[1]
            
            if clf_name not in results[configs[0]]:
                continue
            
            x = np.arange(len(configs))
            width = 0.2
            multiplier = 0
            
            for metric in metrics:
                values = [results[config][clf_name][metric] for config in configs]
                offset = width * multiplier
                rects = ax.bar(x + offset, values, width, label=metric)
                ax.bar_label(rects, padding=2, fmt='%.3f', fontsize=8)
                multiplier += 1
            
            ax.set_ylabel('Score')
            ax.set_title(f'{clf_name} Performance')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(configs, rotation=30, ha='right')
            ax.legend(loc='lower right')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Ablation Study - Performance Comparison', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_contribution(self, results, filename):
        """Plot feature contribution heatmap."""
        configs = list(results.keys())
        classifiers = list(results[configs[0]].keys())
        metric = 'accuracy'
        
        data = []
        for clf in classifiers:
            row = [results[config][clf][metric] for config in configs]
            data.append(row)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=0.9)
        
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([c.replace(' ', '\n')[:15] for c in configs])
        ax.set_yticks(range(len(classifiers)))
        ax.set_yticklabels(classifiers)
        
        for i in range(len(classifiers)):
            for j in range(len(configs)):
                text = ax.text(j, i, f'{data[i][j]:.3f}', ha='center', va='center', color='black' if data[i][j] > 0.6 else 'white')
        
        plt.colorbar(im, ax=ax, label='Accuracy')
        plt.title('Feature Contribution Heatmap')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, results, filename):
        """Plot confusion matrices for the best configuration."""
        best_config = max(
            results.keys(),
            key=lambda x: results[x]['Random Forest']['accuracy']
        )
        
        feature_list = self.configs[best_config]
        X_subset = self.X[feature_list]
        
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_subset)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_scaled, self.y)
        y_pred = rf.predict(X_scaled)
        
        cm = confusion_matrix(self.y, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real', 'Fake'],
                    yticklabels=['Real', 'Fake'], ax=ax)
        ax.set_title(f'Confusion Matrix - Best Config: {best_config}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation study on CSV features')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV features file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    study = AblationStudy(
        csv_path=args.csv,
        output_dir=args.output
    )
    
    results = study.run()