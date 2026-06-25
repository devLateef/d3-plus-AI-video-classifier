"""
ui/streamlit_app.py
Complete Streamlit UI for D3+ AI Video Detector.
"""

import streamlit as st
import requests
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime

# Set Streamlit config options (also set via environment)
st.set_page_config(
    page_title="D3+ AI Video Detector",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .report-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .report-card.warning {
        border-left-color: #ffa500;
    }
    .report-card.danger {
        border-left-color: #dc3545;
    }
    .report-card.success {
        border-left-color: #28a745;
    }
    .confidence-bar {
        height: 24px;
        border-radius: 10px;
        background: #e0e0e0;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .verdict-badge {
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        display: inline-block;
    }
    .verdict-ai {
        background: #dc3545;
        color: white;
    }
    .verdict-real {
        background: #28a745;
        color: white;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# Helper Functions
# =============================================================================

def check_api_health(api_url: str) -> tuple:
    """Check if API is healthy."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": f"API returned {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return False, {"error": f"Cannot connect to API at {api_url}"}
    except Exception as e:
        return False, {"error": str(e)}


def display_verdict(is_ai: bool):
    """Display verdict badge."""
    if is_ai:
        st.markdown('<span class="verdict-badge verdict-ai">🚨 AI-GENERATED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="verdict-badge verdict-real">✅ AUTHENTIC</span>', unsafe_allow_html=True)


def display_confidence_bar(confidence: float):
    """Display confidence bar."""
    color = "#28a745" if confidence > 0.8 else "#ffa500" if confidence > 0.5 else "#dc3545"
    st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:{confidence*100:.1f}%; background:{color};">
                {confidence:.1%}
            </div>
        </div>
    """, unsafe_allow_html=True)


def display_full_report(report: dict):
    """Display full report."""
    if not report:
        return
    
    prediction = report.get('prediction', {})
    is_ai = prediction.get('is_ai_generated', False)
    confidence = prediction.get('confidence_score', 0)
    
    # Verdict
    card_class = "danger" if is_ai else "success"
    st.markdown(f"""
        <div class="report-card {card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin:0;">{'🚨 AI-GENERATED' if is_ai else '✅ AUTHENTIC'}</h3>
                    <p style="margin:0; color:#666;">Verdict based on multi-dimensional analysis</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2rem; font-weight: 700;">{confidence:.1%}</div>
                    <div style="color:#666; font-size: 0.8rem;">Confidence Score</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature breakdown
    if 'feature_breakdown' in report:
        st.markdown("#### 🔬 Feature Breakdown")
        fb = report['feature_breakdown']
        cols = st.columns(3)
        features = [
            ("D3 Score", fb.get('d3_score', 0)),
            ("Color Features", fb.get('color_features', 0)),
            ("Temporal Features", fb.get('temporal_features', 0))
        ]
        for col, (name, value) in zip(cols, features):
            with col:
                st.metric(name, f"{value:.1%}")
                st.progress(value)
    
    # Interpretation
    if 'interpretation' in report:
        st.markdown("#### 📝 Interpretation")
        st.info(report['interpretation'].get('summary', 'No interpretation available.'))
        st.caption(f"Confidence Level: {report['interpretation'].get('confidence_level', 'N/A')}")
    
    # Metadata
    if 'metadata' in report:
        st.markdown("#### 📋 Video Metadata")
        md = report['metadata']
        cols = st.columns(4)
        with cols[0]:
            st.metric("Duration", f"{md.get('duration', 0):.1f}s")
        with cols[1]:
            st.metric("Resolution", f"{md.get('width', 0)}x{md.get('height', 0)}")
        with cols[2]:
            st.metric("Codec", md.get('codec', 'N/A'))
        with cols[3]:
            st.metric("Frame Rate", f"{md.get('frame_rate', 0):.1f} fps")

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/video.png", width=80)
    st.title("D3+ Detector")
    st.markdown("---")
    
    # API configuration
    api_url = st.text_input("API URL", value="http://localhost:8000")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Check Health", use_container_width=True):
            with st.spinner("Checking API health..."):
                is_healthy, data = check_api_health(api_url)
                if is_healthy:
                    st.success(f"✅ API is healthy")
                    st.caption(f"Version: {data.get('version', 'N/A')}")
                else:
                    st.error(f"❌ {data.get('error', 'API not reachable')}")
    
    with col2:
        if st.button("🧹 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared")
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info(
        "D3+ uses multi-dimensional features to detect AI-generated videos. "
        "Upload a video to get a detailed report with confidence scores."
    )

# =============================================================================
# Main Content
# =============================================================================

st.markdown('<div class="main-header">🎬 D3+ AI Video Detector</div>', unsafe_allow_html=True)
st.markdown("Upload a video to receive a detailed, explainable report on whether it is AI-generated or authentic.")

# Tabs
tab1, tab2 = st.tabs(["📤 Upload & Analyze", "📊 Report"])

# =============================================================================
# Tab 1: Upload
# =============================================================================

with tab1:
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP4, AVI, MOV, MKV. Max size: 500MB."
    )
    
    if uploaded_file is not None:
        # Display video preview
        video_bytes = uploaded_file.read()
        st.video(video_bytes)
        uploaded_file.seek(0)
        
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Check API health
                status_text.text("🔄 Checking API health...")
                progress_bar.progress(10)
                
                is_healthy, health_data = check_api_health(api_url)
                if not is_healthy:
                    st.error(f"❌ {health_data.get('error', 'API not reachable')}")
                    st.info("Make sure the FastAPI server is running with: python run_app.py")
                    progress_bar.progress(0)
                    st.stop()
                
                # Upload video
                status_text.text("📤 Uploading video...")
                progress_bar.progress(20)
                
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                start_time = time.time()
                
                # Process
                status_text.text("🧠 Analyzing video... (this may take 10-30 seconds)")
                progress_bar.progress(40)
                
                response = requests.post(f"{api_url}/predict", files=files, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    result['video_name'] = uploaded_file.name
                    result['analysis_time'] = time.time() - start_time
                    
                    st.session_state.result = result
                    st.success("✅ Analysis complete! View the report in the 'Report' tab.")
                    st.rerun()
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")
                    progress_bar.progress(0)
                    
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to API at {api_url}")
                st.info("Make sure the FastAPI server is running with: python run_app.py")
                progress_bar.progress(0)
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The video may be too large or the server is busy.")
                progress_bar.progress(0)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                progress_bar.progress(0)
            finally:
                uploaded_file.seek(0)

# =============================================================================
# Tab 2: Report
# =============================================================================

with tab2:
    if "result" in st.session_state and st.session_state.result:
        result = st.session_state.result
        report = result.get('report', {})
        
        # Video info
        st.markdown(f"### 📹 {result.get('video_name', 'Uploaded Video')}")
        st.caption(f"Video ID: `{result.get('video_id', 'N/A')}`")
        if "analysis_time" in result:
            st.caption(f"Analysis time: {result['analysis_time']:.2f} seconds")
        
        # Display report
        if report:
            display_full_report(report)
            
            # JSON download
            report_json = json.dumps(report, indent=2)
            st.download_button(
                label="📥 Download Report (JSON)",
                data=report_json,
                file_name=f"report_{result.get('video_id', 'unknown')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        if st.button("🔄 New Analysis", use_container_width=True):
            del st.session_state.result
            st.rerun()
    else:
        st.info("👆 Upload a video and click 'Analyze Video' to see the report here.")

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        D3+ AI Video Detector v1.0.0 | Powered by FastAPI + Streamlit
    </div>
""", unsafe_allow_html=True)