"""
verify_setup.py
Check if everything is ready to run.
"""

import os
import sys
from pathlib import Path

def check_ffmpeg():
    try:
        import subprocess
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

def check_requirements():
    try:
        import torch, fastapi, streamlit, cv2, numpy, sklearn, pandas
        return True
    except ImportError as e:
        print(f"Missing: {e}")
        return False

def check_model():
    model_path = Path("trained_models/d3_plus_model.pth")
    if model_path.exists():
        return True
    print("Model not found. You need to train one first.")
    return False

def check_folders():
    required = ["tmp", "results", "reports"]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        print(f"Missing folders: {missing}")
        return False
    return True

print("🔍 D3+ Setup Verification")
print("="*40)

print(f"FFmpeg: {'✅' if check_ffmpeg() else '❌ (Install ffmpeg)'}")
print(f"Python packages: {'✅' if check_requirements() else '❌ (Run: pip install -r requirements.txt)'}")
print(f"Model weights: {'✅' if check_model() else '❌ (Train or download model)'}")
print(f"Required folders: {'✅' if check_folders() else '❌ (Create missing folders)'}")

print("\nIf all checks pass, run: python run_app.py")