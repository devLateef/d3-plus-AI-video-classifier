"""
scripts/check_outliers.py
Check for outliers in the columns with missing values (skewness and kurtosis).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def check_outliers(data: pd.DataFrame, columns: list = None):
    """
    Check for outliers in specified columns.
    
    Args:
        data: DataFrame containing your features
        columns: List of column names to check (if None, checks all skew/kurt columns)
    """
    
    # If no columns specified, find all skewness and kurtosis columns
    if columns is None:
        columns = [col for col in data.columns if '_skew' in col or '_kurt' in col]
    
    print("="*60)
    print("📊 OUTLIER ANALYSIS FOR COLUMNS WITH MISSING VALUES")
    print("="*60)
    print(f"Checking {len(columns)} columns...")
    print()
    
    results = {}
    
    for col in columns:
        # Get non-null values
        values = data[col].dropna().values
        
        if len(values) == 0:
            print(f"⚠️ {col}: All values are missing!")
            continue
        
        # Calculate statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        # Define outlier boundaries
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outliers
        outliers = values[(values < lower_bound) | (values > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(values)) * 100
        
        # Z-score method (alternative)
        z_scores = np.abs(stats.zscore(values))
        z_outliers = values[z_scores > 3]
        z_outlier_count = len(z_outliers)
        
        results[col] = {
            'mean': mean_val,
            'std': std_val,
            'q1': q1,
            'median': np.median(values),
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage,
            'z_outlier_count': z_outlier_count,
            'min': np.min(values),
            'max': np.max(values),
            'missing_count': data[col].isna().sum()
        }
    
    # Display results
    print("="*60)
    print("📊 SUMMARY STATISTICS")
    print("="*60)
    print(f"{'Column':<25} {'Missing':<10} {'Outliers':<10} {'Outlier%':<10} {'Range'}")
    print("-"*80)
    
    for col, stats_dict in results.items():
        if stats_dict:
            print(f"{col:<25} {stats_dict['missing_count']:<10} {stats_dict['outlier_count']:<10} {stats_dict['outlier_percentage']:<10.2f} [{stats_dict['min']:.4f}, {stats_dict['max']:.4f}]")
    
    print()
    print("="*60)
    print("📊 DETAILED STATISTICS FOR EACH COLUMN")
    print("="*60)
    
    for col, stats_dict in results.items():
        if stats_dict:
            print(f"\n{col}:")
            print(f"  Mean: {stats_dict['mean']:.4f}")
            print(f"  Std: {stats_dict['std']:.4f}")
            print(f"  Median: {stats_dict['median']:.4f}")
            print(f"  Q1: {stats_dict['q1']:.4f}")
            print(f"  Q3: {stats_dict['q3']:.4f}")
            print(f"  IQR: {stats_dict['iqr']:.4f}")
            print(f"  Range: [{stats_dict['min']:.4f}, {stats_dict['max']:.4f}]")
            print(f"  Outliers (IQR): {stats_dict['outlier_count']} ({stats_dict['outlier_percentage']:.2f}%)")
            print(f"  Outliers (Z-score): {stats_dict['z_outlier_count']}")
            print(f"  Missing values: {stats_dict['missing_count']}")
    
    return results


def plot_outliers(data: pd.DataFrame, columns: list = None, n_cols: int = 4):
    """
    Plot boxplots and histograms for outlier visualization.
    """
    if columns is None:
        columns = [col for col in data.columns if '_skew' in col or '_kurt' in col]
    
    # Limit to first 8 columns to keep plots manageable
    if len(columns) > 8:
        columns = columns[:8]
    
    fig, axes = plt.subplots(len(columns), 2, figsize=(12, 4 * len(columns)))
    
    if len(columns) == 1:
        axes = [axes]
    
    for i, col in enumerate(columns):
        values = data[col].dropna().values
        
        if len(values) == 0:
            continue
        
        # Boxplot
        ax1 = axes[i][0]
        ax1.boxplot(values)
        ax1.set_title(f'{col} - Boxplot')
        ax1.set_ylabel('Value')
        
        # Histogram
        ax2 = axes[i][1]
        ax2.hist(values, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(values), color='red', linestyle='--', label='Mean')
        ax2.axvline(np.median(values), color='green', linestyle='--', label='Median')
        ax2.set_title(f'{col} - Histogram')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('outlier_plots.png', dpi=150)
    plt.show()
    print("✅ Plots saved to 'outlier_plots.png'")


def analyze_missing_patterns(data: pd.DataFrame):
    """
    Analyze if missing values follow a pattern.
    """
    # Identify all skewness and kurtosis columns
    skew_cols = [col for col in data.columns if '_skew' in col]
    kurt_cols = [col for col in data.columns if '_kurt' in col]
    
    print("="*60)
    print("📊 MISSING VALUE PATTERN ANALYSIS")
    print("="*60)
    
    # Check if missing values occur in the same rows
    if skew_cols:
        # Create a mask for rows with any missing skewness
        skew_missing_mask = data[skew_cols].isna().any(axis=1)
        skew_missing_count = skew_missing_mask.sum()
        print(f"Rows with any missing skewness: {skew_missing_count} ({skew_missing_count/len(data)*100:.2f}%)")
    
    if kurt_cols:
        # Create a mask for rows with any missing kurtosis
        kurt_missing_mask = data[kurt_cols].isna().any(axis=1)
        kurt_missing_count = kurt_missing_mask.sum()
        print(f"Rows with any missing kurtosis: {kurt_missing_count} ({kurt_missing_count/len(data)*100:.2f}%)")
    
    # Check if the same rows have missing skewness and kurtosis
    if skew_cols and kurt_cols:
        both_missing = (data[skew_cols].isna().any(axis=1)) & (data[kurt_cols].isna().any(axis=1))
        both_missing_count = both_missing.sum()
        print(f"Rows with both missing skewness and kurtosis: {both_missing_count} ({both_missing_count/len(data)*100:.2f}%)")
    
    # Check if missing values are limited to specific frame numbers
    print("\nMissing values by frame (if frame number is in column name):")
    for col in skew_cols + kurt_cols:
        missing_count = data[col].isna().sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} missing")


if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('full_features_with_gifs.csv')
    print(f"Loaded {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Run outlier analysis
    results = check_outliers(data)
    
    # Analyze missing patterns
    analyze_missing_patterns(data)
    
    # Plot outliers
    plot_outliers(data)