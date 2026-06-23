import pandas as pd
import numpy as np
from typing import Dict, List

# Load your data
data = pd.read_csv('full_features_with_gifs.csv')

print("="*60)
print("📊 DATA OVERVIEW")
print("="*60)
print(f"Shape: {data.shape}")
print(f"Rows: {data.shape[0]}")
print(f"Columns: {data.shape[1]}")
print()

print("="*60)
print("📊 DESCRIPTIVE STATISTICS")
print("="*60)
print(data.describe())
print()


def get_missing_columns(data: pd.DataFrame) -> Dict[str, int]:
    """Get dictionary of columns with missing values."""
    missing_counts = data.isna().sum()
    return missing_counts[missing_counts > 0].to_dict()

missing_dict = get_missing_columns(data)

print("="*60)
print("📊 COLUMNS WITH MISSING VALUES (DICT)")
print("="*60)
print(missing_dict)
print(list(missing_dict.values()))