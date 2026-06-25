"""
run_app.py
Launcher for both FastAPI and Streamlit with email prompt bypass.
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
    cmd = "uvicorn api.app.main:app --host 127.0.0.1 --port 8000 --reload"
    print(f"Starting API: {cmd}")
    os.system(cmd)


def run_streamlit():
    """Run Streamlit UI with all prompts disabled."""
    time.sleep(2)
    
    # Set environment variables to disable all prompts
    env = os.environ.copy()
    env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    env['STREAMLIT_SERVER_ENABLE_FILE_WATCHER'] = 'false'
    env['STREAMLIT_TELEMETRY_ENABLED'] = 'false'
    env['STREAMLIT_CLIENT_SHOW_ERROR_DETAILS'] = 'false'
    
    # Full command with all flags
    cmd = [
        "streamlit", "run", "ui/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false",
        "--logger.level", "error",
        "--global.developmentMode", "false"
    ]
    
    print(f"Starting Streamlit: {' '.join(cmd)}")
    print("Waiting for Streamlit to start...")
    
    # Run with env
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n🛑 Streamlit stopped by user")
    except Exception as e:
        print(f"Error running Streamlit: {e}")


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
    print("Starting Streamlit UI on http://localhost:8501")
    print("Press Ctrl+C to stop all services\n")
    
    # Start API
    if check_health():
        print("✅ API is already running on port 8000")
    else:
        print("⏳ Starting API...")
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        
        # Wait for API to start
        for i in range(30):
            time.sleep(1)
            if check_health():
                print("✅ API is ready!")
                break
            if i % 5 == 0:
                print(f"⏳ Waiting for API... ({i+1}s)")
    
    # Give API a moment to fully initialize
    time.sleep(2)
    
    # Start Streamlit
    print("\n" + "="*60)
    print("Starting Streamlit UI...")
    print("This will open a browser window automatically.")
    print("="*60 + "\n")
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(3)
        try:
            webbrowser.open("http://localhost:8501")
            print("\n🌐 Browser opened to http://localhost:8501")
        except:
            print("\n🌐 Please open http://localhost:8501 manually")
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run Streamlit (this blocks until Ctrl+C)
    run_streamlit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down all services...")
        sys.exit(0)