"""
run_app.py
Launcher for both FastAPI and Streamlit.
"""

import os
import sys
import subprocess
import time
import threading
import webbrowser
import signal
from pathlib import Path


def run_api():
    """Run FastAPI server."""
    # Use 127.0.0.1 instead of localhost
    cmd = "uvicorn api.app.main:app --host 127.0.0.1 --port 8000 --reload"
    print(f"Starting API: {cmd}")
    os.system(cmd)


def run_streamlit():
    """Run Streamlit UI."""
    time.sleep(3)  # Wait for API to start
    cmd = "streamlit run ui/streamlit_app.py --server.port 8501 --server.address 127.0.0.1"
    print(f"Starting Streamlit: {cmd}")
    os.system(cmd)


def check_health():
    """Check if API is healthy."""
    import requests
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def main():
    print("="*60)
    print("🚀 Starting D3+ AI Video Detector")
    print("="*60)
    print("\nStarting FastAPI server on http://127.0.0.1:8000")
    print("Starting Streamlit UI on http://127.0.0.1:8501")
    print("\nPress Ctrl+C to stop both services\n")
    
    # Check if API is already running
    if check_health():
        print("✅ API is already running on port 8000")
    else:
        # Run API in a separate thread
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        print("⏳ Waiting for API to start...")
        
        # Wait for API to be ready
        for i in range(30):
            time.sleep(1)
            if check_health():
                print("✅ API is ready!")
                break
            if i % 5 == 0:
                print(f"⏳ Waiting for API... ({i+1}s)")
        else:
            print("⚠️ API may not be ready. Check logs.")
    
    # Run Streamlit in a separate thread
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()
    
    # Wait for Streamlit to start
    time.sleep(3)
    
    # Open browser
    print("\n🌐 Opening browser...")
    webbrowser.open("http://127.0.0.1:8501")
    
    print("\n" + "="*60)
    print("✅ Services running!")
    print("   API:     http://127.0.0.1:8000")
    print("   API Docs: http://127.0.0.1:8000/docs")
    print("   UI:      http://127.0.0.1:8501")
    print("="*60)
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        sys.exit(0)


if __name__ == "__main__":
    main()