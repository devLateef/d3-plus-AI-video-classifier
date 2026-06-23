"""
run_app.py
Launcher for both FastAPI and Streamlit.
"""

import os
import sys
import subprocess
import time
import signal
import threading
import webbrowser
from pathlib import Path

def run_api():
    """Run FastAPI server."""
    os.system("uvicorn api.app.main:app --host 0.0.0.0 --port 8000 --reload")

def run_streamlit():
    """Run Streamlit UI."""
    time.sleep(2)  # Wait for API to start
    os.system("streamlit run ui/streamlit_app.py --server.port 8501")

def main():
    print("-----------------------")
    print(" Starting D3+ AI Video Detector")
    print("-----------------------")
    print("\nStarting FastAPI server on http://localhost:8000")
    print("Starting Streamlit UI on http://localhost:8501")
    print("\nPress Ctrl+C to stop both services\n")
    
    # Run both in separate threads
    api_thread = threading.Thread(target=run_api)
    streamlit_thread = threading.Thread(target=run_streamlit)
    
    api_thread.start()
    streamlit_thread.start()
    
    # Open browser
    time.sleep(3)
    webbrowser.open("http://localhost:8501")
    
    # Wait for threads
    try:
        api_thread.join()
        streamlit_thread.join()
    except KeyboardInterrupt:
        print("\n\n Shutting down services...")
        os.kill(os.getpid(), signal.SIGTERM)

if __name__ == "__main__":
    main()