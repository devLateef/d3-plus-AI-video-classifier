import pandas as pd
from pathlib import Path

# Load CSV
df = pd.read_csv('../GenVideo/csv/dataset.csv')

# Check if any GIF entries exist in CSV
gif_csv_entries = df[df['frame_dir'].str.contains('.gif', case=False)]
print(f"GIF entries in CSV: {len(gif_csv_entries)}")

# Check what video_id values look like for GIFs (if any)
if len(gif_csv_entries) > 0:
    print(gif_csv_entries[['frame_dir', 'video_id']].head())