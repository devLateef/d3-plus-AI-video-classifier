"""
streamlit_app.py
Complete Streamlit UI for D3+ AI Video Detector.
Displays a comprehensive report directly in the UI.
"""

import streamlit as st
import requests
import json
import time
import os
from pathlib import Path
import base64
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="D3+ AI Video Detector",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better report display
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
    .feature-box {
        background: white;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.25rem 0;
        border: 1px solid #e0e0e0;
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

# Helper Functions for Report Display
def display_verdict(is_ai_generated: bool):
    """Display the verdict badge."""
    if is_ai_generated:
        st.markdown('<span class="verdict-badge verdict-ai"> AI-GENERATED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="verdict-badge verdict-real"> AUTHENTIC</span>', unsafe_allow_html=True)

def display_confidence_bar(confidence: float):
    """Display a styled confidence bar."""
    color = "#28a745" if confidence > 0.8 else "#ffa500" if confidence > 0.5 else "#dc3545"
    st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:{confidence*100:.1f}%; background:{color};">
                {confidence:.1%}
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_feature_breakdown(feature_breakdown: dict):
    """Display feature breakdown as a horizontal bar chart."""
    if not feature_breakdown:
        return
    
    st.markdown("#### Feature Breakdown")
    
    # Using columns for a cleaner look
    cols = st.columns(3)
    features = [
        ("D3 Score", feature_breakdown.get('d3_score', 0)),
        ("Color Features", feature_breakdown.get('color_features', 0)),
        ("Temporal Features", feature_breakdown.get('temporal_features', 0))
    ]
    
    for col, (name, value) in zip(cols, features):
        with col:
            st.markdown(f"**{name}**")
            st.progress(value, text=f"{value:.1%}")

def display_metadata(metadata: dict):
    """Display video metadata."""
    if not metadata:
        return
    
    st.markdown("#### Video Metadata")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Duration", f"{metadata.get('duration', 0):.1f}s")
    with col2:
        st.metric("Resolution", f"{metadata.get('width', 0)}x{metadata.get('height', 0)}")
    with col3:
        st.metric("Codec", metadata.get('codec', 'N/A'))
    with col4:
        st.metric("Frame Rate", f"{metadata.get('frame_rate', 0):.1f} fps")

def display_interpretation(interpretation: dict):
    """Display the AI-generated interpretation."""
    if not interpretation:
        return
    
    st.markdown("#### Interpretation")
    
    confidence_level = interpretation.get('confidence_level', 'Medium')
    emoji = "✅" if confidence_level == "High" else "⚠️" if confidence_level == "Medium" else "❓"
    
    st.info(f"{emoji} **Confidence Level:** {confidence_level}")
    st.write(interpretation.get('summary', 'No interpretation available.'))

def display_full_report(report: dict):
    """
    Display the full report in a structured, readable format.
    """
    if not report:
        st.warning("No detailed report available.")
        return
    
    # --- 1. Verdict Header ---
    prediction = report.get('prediction', {})
    is_ai = prediction.get('is_ai_generated', False)
    confidence = prediction.get('confidence_score', 0)
    
    # Show a prominent verdict card
    card_class = "danger" if is_ai else "success"
    st.markdown(f"""
        <div class="report-card {card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin:0;">{' AI-GENERATED' if is_ai else ' AUTHENTIC'}</h3>
                    <p style="margin:0; color:#666;">Verdict based on multi-dimensional analysis</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2rem; font-weight: 700;">{confidence:.1%}</div>
                    <div style="color:#666; font-size: 0.8rem;">Confidence Score</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # --- 2. Confidence Visualization ---
    st.markdown("#### Confidence Breakdown")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("**Prediction Probability**")
        display_confidence_bar(prediction.get('probability', 0))
    
    with col2:
        st.metric("Processing Time", f"{report.get('processing_time_ms', 0):.0f}ms")
    
    # --- 3. Feature Breakdown ---
    display_feature_breakdown(report.get('feature_breakdown', {}))
    
    # --- 4. Interpretation ---
    display_interpretation(report.get('interpretation', {}))
    
    # --- 5. Video Metadata ---
    display_metadata(report.get('metadata', {}))
    
    # --- 6. Raw Data (Collapsible for transparency) ---
    with st.expander(" View Full Report Data (JSON)"):
        st.json(report)

def display_prediction_result(result: dict):
    """
    Display the prediction result, including the report.
    """
    video_name = result.get('video_name', 'Uploaded Video')
    video_id = result.get('video_id', 'N/A')
    status = result.get('status', 'unknown')
    
    if status != 'success':
        st.error(f" Error: {result.get('message', 'Unknown error')}")
        return
    
    # Show video info header
    st.markdown(f"### {video_name}")
    st.caption(f"Video ID: `{video_id}`")
    
    # Display the report
    report = result.get('report', {})
    if report:
        display_full_report(report)
    else:
        # Fallback: display basic info if report is missing
        st.warning("Detailed report not available. Showing basic prediction.")
        is_ai = result.get('is_ai_generated', False)
        confidence = result.get('confidence_score', 0)
        
        col1, col2 = st.columns(2)
        with col1:
            display_verdict(is_ai)
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/video.png", width=80)
    st.title("D3+ Detector")
    st.markdown("---")
    
    # API configuration
    api_url = st.text_input("API URL", value="http://localhost:8000")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Che ck Health", use_container_width=True):
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    st.success(" API is healthy")
                else:
                    st.error(f" API error: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error(" Cannot connect to API")
            except Exception as e:
                st.error(f" Error: {e}")
    
    with col2:
        if st.button(" Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared")
    
    st.markdown("---")
    st.markdown("###  About")
    st.info(
        "D3+ uses multi-dimensional features to detect AI-generated videos. "
        "Upload a video to get a detailed report with confidence scores."
    )
    
    st.markdown("---")
    st.markdown("### Report Sections")
    st.markdown("""
    -  **Verdict** (AI/Real)
    -  **Confidence Score**
    -  **Feature Breakdown**
    -  **Interpretation**
    -  **Video Metadata**
    """)

# Main Content
st.markdown('<div class="main-header">🎬 D3+ AI Video Detector</div>', unsafe_allow_html=True)
st.markdown("Upload a video to receive a detailed, explainable report on whether it is AI-generated or authentic.")

# Tabs
tab1, tab2, tab3 = st.tabs([" Upload & Analyze", " Report", " About"])

# Tab 1: Upload & Analyze
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
        
        # Reset file pointer for upload
        uploaded_file.seek(0)
        
        # Analyze button
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Upload
                status_text.text(" Uploading video...")
                progress_bar.progress(15)
                
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                start_time = time.time()
                
                # Step 2: Process
                status_text.text(" Analyzing video... (this may take 10-30 seconds)")
                progress_bar.progress(40)
                
                response = requests.post(f"{api_url}/predict", files=files, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    progress_bar.progress(100)
                    status_text.text(" Analysis complete!")
                    
                    # Add video name to result
                    result['video_name'] = uploaded_file.name
                    
                    # Store result in session state for report tab
                    st.session_state.result = result
                    st.session_state.analysis_time = time.time() - start_time
                    
                    st.success(" Analysis complete! View the report in the 'Report' tab.")
                    st.rerun()
                else:
                    st.error(f" Error {response.status_code}: {response.text}")
                    progress_bar.progress(0)
                    
            except requests.exceptions.ConnectionError:
                st.error(" Cannot connect to API. Make sure the server is running.")
            except requests.exceptions.Timeout:
                st.error(" Request timed out. The video may be too large or the server is busy.")
            except Exception as e:
                st.error(f" Error: {str(e)}")
            finally:
                uploaded_file.seek(0)

# Tab 2: Report
with tab2:
    if "result" in st.session_state and st.session_state.result:
        result = st.session_state.result
        
        # Display the report
        display_prediction_result(result)
        
        # Show processing time
        if "analysis_time" in st.session_state:
            st.caption(f"Total analysis time: {st.session_state.analysis_time:.2f} seconds")
        
        # Reset button
        if st.button(" New Analysis", use_container_width=True):
            del st.session_state.result
            if "analysis_time" in st.session_state:
                del st.session_state.analysis_time
            st.rerun()
    else:
        st.info(" Upload a video and click 'Analyze Video' to see the report here.")

# Tab 3: About
with tab3:
    st.markdown("### 📖 About D3+ AI Video Detector")
    
    st.markdown("""
    **D3+ (Detection by Difference of Differences)** is a multi-dimensional framework for detecting AI-generated videos.
    
    #### How It Works
    1. **Temporal Analysis**: Analyzes second-order differences in video frames to detect unnatural motion patterns.
    2. **Color Distribution**: Examines RGB and HSV color statistics for AI-generated artifacts.
    3. **Temporal Channel Relationships**: Tracks how color relationships evolve over time.
    4. **Bitrate Analysis**: Analyzes compression patterns for signs of synthetic generation.
    
    #### Report Sections
    - **Verdict**: AI-generated or Authentic
    - **Confidence Score**: How confident the model is in its prediction
    - **Feature Breakdown**: Contribution of each feature dimension
    - **Interpretation**: Human-readable explanation of the result
    - **Video Metadata**: Technical details of the uploaded video
    
    #### Reference
    - Method based on "D3: Training-Free AI-Generated Video Detection Using Second-Order Features"
    - Extended with multi-dimensional feature analysis
    """)
    
    st.markdown("---")
    st.markdown("**Version:** 1.0.0 | **Powered by:** FastAPI + Streamlit")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        D3+ AI Video Detector v1.0.0 | Report displayed directly in UI
    </div>
""", unsafe_allow_html=True)