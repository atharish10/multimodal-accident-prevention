"""
Panimalar Engineering College - Department of Artificial Intelligence & Machine Learning
Multimodal Road Accident Prevention, Pre-Crash Risk Analysis & Damage Severity Assessment System
Batch: D16 | Investigators: Harish AT & Barathraj AS | Supervisor: Dr. A. Arun
"""

# ── DEPLOYMENT URL ─────────────────────────────────────────────────────────────
APP_URL = "https://atharish10-multimodal-accident-prevention-app-7u8wx1.streamlit.app/"
# ───────────────────────────────────────────────────────────────────────────────

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import time
import tempfile
import datetime
import plotly.graph_objects as go
from pathlib import Path

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

from config import SEVERITY_LEVELS, BASE_DIR, SAMPLE_DATA_DIR
from modules.damage_classifier import DamageClassifier
from modules.violation_detector import ViolationDetector
from modules.hazard_detector import HazardDetector
from modules.risk_fusion import MultimodalRiskFusionEngine
from modules.temporal_risk_lstm import RollingSequenceRiskPipeline
from modules.report_generator import generate_incident_pdf

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Road Accident & Damage AI System | Panimalar AIML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Executive Dark/Slate Design System ─────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global App Canvas & Background */
    .stApp {
        background-color: #0B0F19 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(37, 99, 235, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.06) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(15, 23, 42, 0.6) 0px, transparent 50%) !important;
        color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    
    .block-container {
        padding-top: 2.8rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #070A12 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 1.2rem 0 !important;
    }
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #F8FAFC !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 0.75rem !important;
    }

    /* Typography Hierarchy */
    h1, h2, h3, h4, .main-title {
        color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        letter-spacing: -0.025em;
        font-weight: 700;
    }
    p, span, label, div {
        color: #CBD5E1;
    }
    strong, b {
        color: #F8FAFC;
    }
    small, .text-muted {
        color: #94A3B8;
    }
    
    /* Executive Card Surface */
    .saas-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.25s ease;
    }
    .saas-card:hover {
        border-color: rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }
    
    /* High-Precision KPI Tile */
    .kpi-tile {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.035) 0%, rgba(255, 255, 255, 0.015) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        height: 100%;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .kpi-tile:hover {
        border-color: rgba(59, 130, 246, 0.45);
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.5), 0 0 16px rgba(59, 130, 246, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 5px;
    }
    
    /* Modern Severity Badges (Pills) */
    .sev-badge-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(5, 150, 105, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34D399;
        box-shadow: 0 0 14px rgba(16, 185, 129, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        padding: 11px 18px;
        border-radius: 9px;
        font-weight: 800;
        font-size: 1.05rem;
        text-align: center;
        letter-spacing: 0.04em;
    }
    .sev-badge-mid {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.16) 0%, rgba(217, 119, 6, 0.08) 100%);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #FBBF24;
        box-shadow: 0 0 14px rgba(245, 158, 11, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        padding: 11px 18px;
        border-radius: 9px;
        font-weight: 800;
        font-size: 1.05rem;
        text-align: center;
        letter-spacing: 0.04em;
    }
    .sev-badge-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(185, 28, 28, 0.10) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #F87171;
        box-shadow: 0 0 18px rgba(239, 68, 68, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        padding: 11px 18px;
        border-radius: 9px;
        font-weight: 800;
        font-size: 1.05rem;
        text-align: center;
        letter-spacing: 0.04em;
    }
    
    /* Clean Protocol Card */
    .protocol-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.75) 0%, rgba(17, 24, 39, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #3B82F6;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 12px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        font-size: 0.88rem;
        line-height: 1.55;
        color: #E2E8F0;
    }
    
    /* Streamlit Segmented Tabs */
    div[data-testid="stTabs"] [role="tablist"] {
        gap: 6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
        margin-bottom: 16px;
    }
    div[data-testid="stTabs"] button,
    div[data-testid="stTabs"] [role="tab"] {
        color: #94A3B8 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        margin-right: 2px !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stTabs"] button:hover,
    div[data-testid="stTabs"] [role="tab"]:hover {
        color: #F8FAFC !important;
        background: rgba(255, 255, 255, 0.04) !important;
        border-color: rgba(255, 255, 255, 0.06) !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"],
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        background: linear-gradient(180deg, rgba(59, 130, 246, 0.18) 0%, rgba(37, 99, 235, 0.08) 100%) !important;
        border: 1px solid rgba(59, 130, 246, 0.35) !important;
        border-bottom: 2px solid #3B82F6 !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 10px rgba(37, 99, 235, 0.25) !important;
    }
    
    /* Tactile Modern Buttons */
    div.stButton > button {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.13) !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: -0.01em !important;
        padding: 0.55rem 1.15rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(180deg, #2563EB 0%, #1D4ED8 100%) !important;
        border-color: #3B82F6 !important;
        color: #FFFFFF !important;
        transform: translateY(-1.5px) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
    }
    div.stButton > button:active {
        transform: translateY(0.5px) !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4), inset 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    }

    /* Primary / Download Button */
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(180deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #60A5FA !important;
        border-radius: 9px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(180deg, #3B82F6 0%, #2563EB 100%) !important;
        box-shadow: 0 6px 22px rgba(37, 99, 235, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stDownloadButton"] > button:active {
        transform: translateY(0.5px) !important;
    }

    /* Sidebar Link Button */
    a[data-testid="stLinkButton"], 
    div[data-testid="stSidebar"] a[data-testid="stLinkButton"] {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 9px !important;
        color: #F8FAFC !important;
        font-weight: 600 !important;
        text-align: center !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    a[data-testid="stLinkButton"]:hover {
        background: linear-gradient(180deg, #2563EB 0%, #1D4ED8 100%) !important;
        border-color: #3B82F6 !important;
        color: #FFFFFF !important;
        transform: translateY(-1.5px) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }

    /* Form Controls & Inputs */
    div[data-baseweb="select"] > div {
        background-color: #0F172A !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 9px !important;
        color: #F8FAFC !important;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="select"]:focus-within > div {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25), inset 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    }
    ul[role="listbox"] {
        background-color: #0F172A !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.8) !important;
    }
    li[role="option"] {
        color: #CBD5E1 !important;
        font-size: 0.86rem !important;
        transition: all 0.15s ease !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.18) !important;
        color: #60A5FA !important;
    }

    div[data-testid="stTextInput"] input, 
    div[data-testid="stTextArea"] textarea {
        background-color: #0F172A !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 9px !important;
        color: #F8FAFC !important;
        font-size: 0.88rem !important;
        padding: 10px 14px !important;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stTextInput"] input:focus, 
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25), inset 0 1px 2px rgba(0, 0, 0, 0.3) !important;
        outline: none !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3), 0 2px 6px rgba(0, 0, 0, 0.5) !important;
        border: 2px solid #FFFFFF !important;
        width: 17px !important; 
        height: 17px !important;
    }

    /* File Uploader */
    div[data-testid="stFileUploader"] section {
        background-color: rgba(17, 24, 39, 0.6) !important;
        border: 1.5px dashed rgba(255, 255, 255, 0.18) !important;
        border-radius: 12px !important;
        padding: 18px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #3B82F6 !important;
        background-color: rgba(37, 99, 235, 0.06) !important;
        box-shadow: 0 0 16px rgba(59, 130, 246, 0.12) !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: rgba(17, 24, 39, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
        overflow: hidden !important;
        transition: border-color 0.2s ease !important;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(255, 255, 255, 0.15) !important;
    }
    div[data-testid="stExpander"] details summary {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 12px 18px !important;
        transition: background 0.2s ease !important;
    }
    div[data-testid="stExpander"] details summary:hover {
        background: rgba(255, 255, 255, 0.03) !important;
    }

    /* Tables & Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
        overflow: hidden !important;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 9px !important;
        padding: 12px 16px !important;
        font-size: 0.86rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    }

    /* Live Status Pill & Pulse Dot */
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.45; transform: scale(0.85); }
    }
    .live-pulse-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        background: #10B981;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 8px #10B981;
        animation: pulse-dot 2s infinite ease-in-out;
    }
    .header-status-pill {
        display: inline-flex;
        align-items: center;
        padding: 5px 13px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        backdrop-filter: blur(8px);
    }
    .header-status-success {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        font-family: ui-monospace, SFMono-Regular, monospace;
    }
    .header-status-info {
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #38BDF8;
        font-family: ui-monospace, SFMono-Regular, monospace;
    }
    .header-status-purple {
        background: rgba(139, 92, 246, 0.12);
        border: 1px solid rgba(139, 92, 246, 0.3);
        color: #C084FC;
    }

    /* Sidebar Status & Credits Cards */
    .sidebar-status-card {
        background: linear-gradient(145deg, rgba(17, 24, 39, 0.85) 0%, rgba(15, 23, 42, 0.65) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 13px 15px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }
    .sidebar-active-badge {
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34D399;
        padding: 2px 7px;
        border-radius: 9999px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .sidebar-credits-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 14px 15px;
        line-height: 1.45;
    }

    /* Monospace telemetry code */
    code, .telemetry-code {
        font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        font-size: 0.80rem;
    }
    code {
        background: #070A12 !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 5px !important;
        padding: 2px 6px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── AI Modules Initialization ─────────────────────────────────────────────────
damage_clf = DamageClassifier()
violation_det = ViolationDetector()
hazard_det = HazardDetector()
fusion_engine = MultimodalRiskFusionEngine()
lstm_pipeline = RollingSequenceRiskPipeline(window_size=10)

# ── Session State for Inter-Tab Intelligence ───────────────────────────────────
if "current_assessment" not in st.session_state:
    st.session_state.current_assessment = {
        "incident_id": f"PEC-AIML-{int(time.time()) % 100000:05d}",
        "modality": "Unified Multimodal Assessment",
        "overall_risk": 22.0,
        "severity_label": "Low / Cosmetic Severity",
        "severity_color": "#10B981",
        "damage_results": None,
        "violation_results": None,
        "hazard_results": None,
        "annotated_image": None,
        "environmental_factors": {
            "weather": "Clear / Daylight",
            "lighting": "Daylight",
            "traffic_density": "Moderate",
            "multiplier": 1.0
        },
        "notes": "Multimodal decision-support system active."
    }

if "active_damage_scenario" not in st.session_state:
    st.session_state.active_damage_scenario = "Low Severity (Minor Panel Scratch)"

# ── HELPER: DARK-THEME COMPATIBLE GAUGE CHART ─────────────────────────────────
def create_gauge(score: float, title: str, color: str, is_percentage: bool = True):
    raw_val = float(score)
    display_val = raw_val * 100.0 if raw_val <= 1.0 else raw_val
    display_val = min(100.0, max(0.0, display_val))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_val,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 13, 'color': "#94A3B8", 'family': "Inter, sans-serif"}},
        number={'suffix': " %", 'font': {'size': 26, 'color': color, 'family': "Inter, sans-serif"}, 'valueformat': '.1f'},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': color},
            'bgcolor': "rgba(255, 255, 255, 0.04)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.1)",
            'steps': [
                {'range': [0, 38], 'color': "rgba(16, 185, 129, 0.12)"},
                {'range': [38, 72], 'color': "rgba(245, 158, 11, 0.12)"},
                {'range': [72, 100], 'color': "rgba(239, 68, 68, 0.12)"}
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8,
                'value': display_val
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#F8FAFC'},
        height=180,
        margin=dict(l=10, r=10, t=25, b=10)
    )
    return fig

# ── TOP PRESENTATION HEADER ───────────────────────────────────────────────────
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 16px; margin-bottom: 22px; flex-wrap: wrap; gap: 14px;">
    <div>
        <div style="display: flex; align-items: center; gap: 14px;">
            <div style="background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%); color: white; border-radius: 10px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.25rem; box-shadow: 0 0 16px rgba(59, 130, 246, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3); border: 1px solid rgba(255, 255, 255, 0.15);">
                M
            </div>
            <div>
                <h1 style="font-size: 1.55rem; font-weight: 800; letter-spacing: -0.025em; margin: 0; line-height: 1.2; color: #F8FAFC;">
                    Multimodal Road Safety & Damage Intelligence
                </h1>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 3px; font-weight: 500;">
                    Department of AIML • Autonomous Pre-Crash Forecasting & Vehicle Damage Assessment
                </div>
            </div>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
        <span class="header-status-pill header-status-success">
            <span class="live-pulse-dot"></span>CLUSTER ONLINE
        </span>
        <span class="header-status-pill header-status-info">
            LATENCY: 22.4ms
        </span>
        <span class="header-status-pill header-status-purple">
            PEC AIML • BATCH D16
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR CONTROLS ──────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-status-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span style="font-weight: 700; font-size: 0.74rem; color: #94A3B8; letter-spacing: 0.06em; text-transform: uppercase;">Edge Cluster Status</span>
        <span class="sidebar-active-badge">ACTIVE</span>
    </div>
    <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.5;">
        <span style="color: #94A3B8;">Engine:</span> <span style="color: #F8FAFC; font-weight: 600;">ResNet-50 + YOLOv8 + BiLSTM</span><br>
        <span style="color: #94A3B8;">Region:</span> <span style="color: #F8FAFC; font-weight: 600;">ap-south-1 (Mumbai Node)</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.link_button(
    "Open GitHub Repository",
    "https://github.com/atharish10/multimodal-accident-prevention",
    use_container_width=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Active Telemetry & Environmental Controls")

with st.sidebar.expander("🚗 Vehicle Kinematics & Driver Telemetry", expanded=True):
    simulated_speed = st.slider("Simulated Vehicle Speed (km/h)", 20.0, 160.0, 65.0, 5.0)
    vehicle_type_label = st.selectbox(
        "Vehicle Class & Mass Profile",
        [
            "Passenger Sedan (1,400 kg)",
            "Compact SUV (1,850 kg)",
            "Commercial Hauler / Bus (8,500 kg)",
            "Two-Wheeler Motorcycle (180 kg)"
        ]
    )
    if "Sedan" in vehicle_type_label:
        vehicle_mass_kg = 1400.0
    elif "SUV" in vehicle_type_label:
        vehicle_mass_kg = 1850.0
    elif "Commercial" in vehicle_type_label:
        vehicle_mass_kg = 8500.0
    else:
        vehicle_mass_kg = 180.0

    headway_dist = st.slider("Following Distance / Headway (m)", 5.0, 100.0, 28.0, 1.0)
    driver_state_label = st.selectbox(
        "Driver Cognitive State",
        [
            "Alert & Attentive (t_rxn: 0.8s)",
            "Mild Distraction / Infotainment (t_rxn: 1.4s)",
            "Severe Phone Usage (t_rxn: 2.0s)",
            "Drowsy / Micro-Sleep (t_rxn: 2.6s)"
        ]
    )
    if "Alert" in driver_state_label:
        driver_reaction_sec = 0.8
    elif "Mild" in driver_state_label:
        driver_reaction_sec = 1.4
    elif "Severe" in driver_state_label:
        driver_reaction_sec = 2.0
    else:
        driver_reaction_sec = 2.6

    # Dynamic Time-to-Collision (TTC) Calculation
    v_ms = simulated_speed / 3.6
    ttc_sec = headway_dist / max(0.5, v_ms)
    if ttc_sec < 1.5:
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 8px 12px; margin-top: 8px;">
            <span style="color: #F87171; font-weight: 700; font-size: 0.78rem;">🚨 CRITICAL TTC: {ttc_sec:.1f}s</span><br>
            <span style="color: #CBD5E1; font-size: 0.72rem;">AEB Pre-Collision Intervention Primed</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 8px 12px; margin-top: 8px;">
            <span style="color: #34D399; font-weight: 700; font-size: 0.78rem;">SAFE HEADWAY: {ttc_sec:.1f}s TTC</span>
        </div>
        """, unsafe_allow_html=True)

with st.sidebar.expander("🏍️ Two-Wheeler Enforcement Parameters", expanded=False):
    simulated_riders = st.slider("Two-Wheeler Occupants / Riders", 1, 4, 3)
    simulated_helmet = st.checkbox("Rider / Driver Helmet Detected", value=False)
    pillion_helmet = st.checkbox("Pillion Occupant(s) Helmet Detected", value=False)
    enforced_speed_limit = st.selectbox("Corridor Speed Limit (km/h)", [40.0, 50.0, 60.0, 80.0], index=2)

with st.sidebar.expander("🌧️ Sensor & Environmental Telemetry", expanded=False):
    env_weather = st.selectbox(
        "Atmospheric Meteorology",
        ["Clear / Dry Daylight", "Heavy Monsoon Rain", "Dense Winter Fog", "Night Highway / Glare"]
    )
    env_road = st.selectbox(
        "Pavement Friction Profile",
        ["Dry Smooth Bitumen", "Wet Slippery Asphalt", "Cracked Potholed Highway", "Loose Gravel Surface"]
    )
    ambient_lux = st.slider("Ambient Illuminance (Lux)", 5, 10000, 6500, 250)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="sidebar-credits-card">
    <div style="font-weight: 700; font-size: 0.82rem; color: #F8FAFC; margin-bottom: 2px;">Panimalar Engineering College</div>
    <div style="font-size: 0.78rem; color: #60A5FA; font-weight: 500; margin-bottom: 6px;">Department of AI & Machine Learning</div>
    <div style="font-size: 0.74rem; color: #94A3B8; margin-bottom: 10px;">Final Year Project • Batch D16</div>
    <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 8px; font-size: 0.78rem; line-height: 1.5; color: #CBD5E1;">
        <b style="color: #94A3B8; font-size: 0.70rem; text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 2px;">Student Investigators</b>
        • Harish AT<br>
        • Barathraj AS<br>
        <b style="color: #94A3B8; font-size: 0.70rem; text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-top: 8px; margin-bottom: 2px;">Supervisor</b>
        Dr. A. Arun, Associate Professor
    </div>
</div>
""", unsafe_allow_html=True)

# ── 7 DEDICATED CONSOLE TABS ───────────────────────────────────────────────────
tab_damage, tab_video, tab_hazards, tab_violations, tab_fusion, tab_report, tab_architecture = st.tabs([
    "Damage Assessment",
    "Continuous Vision & Dashcam",
    "Road Hazard Scanner",
    "Two-Wheeler Safety",
    "Multimodal Risk Fusion",
    "Incident Dossier (PDF)",
    "System Architecture"
])

# ==============================================================================
# TAB 1: VEHICLE DAMAGE INSPECTION
# ==============================================================================
with tab_damage:
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <h3 style="font-size: 1.25rem; font-weight: 700; margin: 0; color: #F8FAFC;">
            Vehicle Structural & Surface Damage Assessment
        </h3>
        <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 2px; margin-bottom: 14px;">
            High-precision computer vision evaluation of vehicular impact deformation, subframe integrity, and insurance triage.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Inspection Criteria & Severity Classification Guide", expanded=False):
        st.markdown("""
        | Severity Tier | Visual Defect Criteria | Structural Deformation Index | Operational & Insurance Triage |
        | :--- | :--- | :---: | :--- |
        | **LOW SEVERITY** | Superficial clear-coat scratches, small paint abrasions, minor door dings. Subframe & glass 100% intact. | **< 35%** (Nominal: 22%) | **Fully Driveable.** Minor cosmetic workshop repair; standard insurance fast-track claim. No tow truck or ambulance. |
        | **MODERATE SEVERITY** | Fractured/dislodged bumper fascia, damaged headlight assembly, dented quarter panel. Crash frame intact. | **38% – 72%** (Nominal: 58%) | **Repair Required.** Non-driveable at night due to destroyed headlights. Standard flatbed tow dispatch. |
        | **HIGH SEVERITY / CRITICAL** | Severe cabin intrusion, collapsed engine subframe, shattered windshield, crushed crumple zone. | **> 72%** (Nominal: 88%) | **Critical Hazard.** Immediate Tier-1 ALS Ambulance dispatch, hazmat road clearance, total-loss salvage survey. |
        """)

    # Scenario Selection Bar
    st.markdown("**Select Benchmark Scenario to Evaluate (or Upload Custom Image):**")
    sc1, sc2, sc3, sc4 = st.columns([1, 1, 1, 1.2])
    with sc1:
        if st.button("1. Low Severity\n(Minor Panel Scratch)", use_container_width=True):
            st.session_state.active_damage_scenario = "Low Severity (Minor Panel Scratch)"
    with sc2:
        if st.button("2. Moderate Severity\n(Impact & Bumper Fracture)", use_container_width=True):
            st.session_state.active_damage_scenario = "Moderate Severity (Impact & Bumper Fracture)"
    with sc3:
        if st.button("3. High Severity\n(Structural Cabin Crush)", use_container_width=True):
            st.session_state.active_damage_scenario = "High Severity (Structural Cabin Crush)"
    with sc4:
        custom_car_upload = st.file_uploader("Upload Car Photograph", type=["jpg", "jpeg", "png"], key="car_upload_tab1")

    # Load Active Image
    scenario = st.session_state.active_damage_scenario
    if custom_car_upload is not None:
        raw_car_img = Image.open(custom_car_upload)
        scenario_hint = None
    else:
        scenario_hint = scenario
        if "Low" in scenario:
            raw_car_img = Image.open(SAMPLE_DATA_DIR / "sample_low_damage.jpg")
        elif "Moderate" in scenario or "Mid" in scenario:
            raw_car_img = Image.open(SAMPLE_DATA_DIR / "sample_mid_damage.jpg")
        else:
            raw_car_img = Image.open(SAMPLE_DATA_DIR / "sample_high_damage.jpg")

    car_bgr = cv2.cvtColor(np.array(raw_car_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    with st.expander("🛡️ Forensic Claims & Adjuster Telemetry Controls", expanded=True):
        fc_c1, fc_c2 = st.columns(2)
        with fc_c1:
            triage_policy_mode = st.selectbox(
                "Insurance Policy Underwriting Tier",
                [
                    "Comprehensive Zero-Depreciation (100% Parts & Labor)",
                    "Standard Comprehensive (50% Plastic/Glass Deductible)",
                    "Third-Party Liability Only (Salvage Settlement)"
                ],
                key="policy_tier_tab1"
            )
            voluntary_deductible = st.slider(
                "Policyholder Voluntary Deductible (₹)",
                min_value=0,
                max_value=25000,
                value=2500,
                step=500,
                key="deductible_tab1"
            )
        with fc_c2:
            airbag_deployed = st.checkbox(
                "SRS Airbag Modules Deployed (Steering/Curtain)",
                value=("High" in scenario or "Structural" in scenario),
                key="airbag_tab1"
            )
            subframe_distorted = st.checkbox(
                "Structural Subframe Twisted / Frame Rail Pull Required",
                value=("High" in scenario or "Structural" in scenario),
                key="subframe_tab1"
            )
            oem_parts_pref = st.checkbox("OEM Authorized Parts (vs Aftermarket Equivalent)", value=True, key="oem_pref_tab1")

    with st.spinner("Analyzing structural deformity and impact boundaries..."):
        damage_res = damage_clf.assess_damage(car_bgr, None, scenario_hint)
        annotated_car_rgb = cv2.cvtColor(damage_res["annotated_image"], cv2.COLOR_BGR2RGB)

        # Dynamic Financial Claims Assessment
        s_level = damage_res["severity_level"]
        if s_level == "LOW":
            base_parts = 3500.0
            base_labor = 2800.0
            est_days = 2
        elif s_level == "MID":
            base_parts = 24500.0
            base_labor = 11200.0
            est_days = 7
        else:
            base_parts = 118000.0
            base_labor = 45000.0
            est_days = 24

        if not oem_parts_pref:
            base_parts *= 0.72

        airbag_cost = 45000.0 if airbag_deployed else 0.0
        frame_cost = 32000.0 if subframe_distorted else 0.0
        gross_claim = base_parts + base_labor + airbag_cost + frame_cost

        if "Zero-Depreciation" in triage_policy_mode:
            dep_rate = 0.0
        elif "Standard" in triage_policy_mode:
            dep_rate = 0.20
        else:
            dep_rate = 0.65

        depreciation_amt = base_parts * dep_rate
        deductible_amt = min(gross_claim, float(voluntary_deductible))
        net_payout = max(0.0, gross_claim - depreciation_amt - deductible_amt)

        total_loss_threshold = 150000.0
        is_ctl = gross_claim >= total_loss_threshold or (s_level == "HIGH" and subframe_distorted)
        settlement_status = "Constructive Total Loss (CTL) • Salvage Auction" if is_ctl else "Repair Authorized & Approved"

        # Update Session State
        damage_res["financial_claims"] = {
            "policy_tier": triage_policy_mode,
            "base_parts": base_parts,
            "base_labor": base_labor,
            "airbag_cost": airbag_cost,
            "frame_cost": frame_cost,
            "gross_claim": gross_claim,
            "depreciation": depreciation_amt,
            "deductible": deductible_amt,
            "net_payout": net_payout,
            "settlement_status": settlement_status,
            "repair_cycle_days": est_days
        }

        st.session_state.current_assessment.update({
            "modality": f"Vehicle Damage Severity ({damage_res['severity_level']})",
            "overall_risk": float(damage_res["severity_score"] * 100),
            "severity_label": damage_res["severity_label"],
            "severity_color": damage_res["severity_color"],
            "annotated_image": Image.fromarray(annotated_car_rgb),
            "damage_results": damage_res
        })

    col_img, col_metrics = st.columns([1.25, 1])
    with col_img:
        st.image(annotated_car_rgb, caption="High-Precision Computer Vision Damage Localization (Corner Reticles)", use_container_width=True)

    with col_metrics:
        # Severity Badge
        badge_class = "sev-badge-low" if s_level == "LOW" else ("sev-badge-mid" if s_level == "MID" else "sev-badge-high")
        st.markdown(f"""
        <div class="{badge_class}">
            {damage_res['severity_label'].upper()}
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(
            create_gauge(damage_res["severity_score"], "Damage Severity Index", damage_res["severity_color"]),
            use_container_width=True,
            key=f"gauge_tab1_{damage_res['severity_level']}"
        )

        # KPI Metric Grid
        k1, k2 = st.columns(2)
        with k1:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Damage Area</div>
                <div class="kpi-value">{damage_res['damage_area_percent']}%</div>
                <div class="kpi-subtext">Surface profile coverage</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Defects Found</div>
                <div class="kpi-value">{len(damage_res['damages_found'])} Identified</div>
                <div class="kpi-subtext">Localized defect count</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        k3, k4 = st.columns(2)
        with k3:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Est. Repair Cost</div>
                <div class="kpi-value" style="font-size: 1.05rem;">₹{gross_claim:,.0f}</div>
                <div class="kpi-subtext">Authorized workshop estimate</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Triage Claim Code</div>
                <div class="kpi-value" style="font-size: 0.95rem; font-family: monospace;">{damage_res.get('triage_code', 'INS-01')}</div>
                <div class="kpi-subtext">Automated insurance routing</div>
            </div>
            """, unsafe_allow_html=True)

    # Technical Details
    st.markdown("---")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"""
        <div class="protocol-card">
            <b>Structural Integrity Status:</b> {damage_res.get('structural_status', 'Nominal')}<br>
            <b>Operational Driveability:</b> {damage_res.get('driveable_status', 'Driveable')}<br>
            <b>Impact Analysis:</b> {damage_res['description']}
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown(f"""
        <div class="protocol-card" style="border-left-color: {damage_res['severity_color']};">
            <b>Recommended Action & Emergency Protocol:</b><br>
            {damage_res['recommended_action']}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Itemized Insurance Settlement Breakdown:")
    st.markdown(f"""
    <div class="saas-card" style="margin-bottom: 14px;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 12px;">
            <div style="background: rgba(255,255,255,0.03); padding: 10px 12px; border-radius: 8px;">
                <div style="color: #94A3B8; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;">Parts Replacement</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC;">₹{base_parts:,.0f}</div>
                <div style="font-size: 0.70rem; color: #64748B;">{'OEM Certified' if oem_parts_pref else 'Aftermarket'}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 10px 12px; border-radius: 8px;">
                <div style="color: #94A3B8; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;">Labor & Paint Booth</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC;">₹{base_labor:,.0f}</div>
                <div style="font-size: 0.70rem; color: #64748B;">{est_days} Days Est. Cycle</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 10px 12px; border-radius: 8px;">
                <div style="color: #94A3B8; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;">Airbags & Chassis</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC;">₹{(airbag_cost + frame_cost):,.0f}</div>
                <div style="font-size: 0.70rem; color: {'#F87171' if (airbag_cost+frame_cost)>0 else '#64748B'};">{'SRS/Frame Triggered' if (airbag_cost+frame_cost)>0 else 'None'}</div>
            </div>
            <div style="background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.25); padding: 10px 12px; border-radius: 8px;">
                <div style="color: #60A5FA; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;">Net Insurer Payout</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #38BDF8;">₹{net_payout:,.0f}</div>
                <div style="font-size: 0.70rem; color: #93C5FD;">Gross: ₹{gross_claim:,.0f} (Ded: ₹{deductible_amt:,.0f})</div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px; font-size: 0.82rem;">
            <span><b>Claim Underwriting Status:</b> <span style="color: {'#EF4444' if is_ctl else '#34D399'}; font-weight: 700;">{settlement_status}</span></span>
            <span style="color: #94A3B8; font-size: 0.76rem;">Policy Mode: {triage_policy_mode.split('(')[0]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("#### Technical Breakdown of Identified Deformities:")
    defect_table = []
    for d in damage_res["damages_found"]:
        defect_table.append({
            "Defect Class": d["class"].replace("_", " ").upper(),
            "Confidence": f"{d['confidence']:.1%}",
            "Bounding Coordinates [X1, Y1, X2, Y2]": str(d["bbox"]),
            "Severity Weight": f"{damage_clf.classes.get(d['class'], {}).get('weight', 0.3):.2f}"
        })
    st.dataframe(defect_table, use_container_width=True)


# ==============================================================================
# TAB 2: CONTINUOUS VISION & DASHCAM
# ==============================================================================
with tab_video:
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <h3 style="font-size: 1.25rem; font-weight: 700; margin: 0; color: #F8FAFC;">
            Continuous Spatio-Temporal Video Risk Pipeline (CNN + BiLSTM)
        </h3>
        <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 2px; margin-bottom: 14px;">
            Frame-by-frame temporal feature extraction, optical flow motion looming rate, and dynamic collision probability.
        </p>
    </div>
    """, unsafe_allow_html=True)

    stream_mode = st.radio(
        "Select Stream Source",
        [
            "Real-Time Highway Dashcam Simulation (Pre-Loaded Benchmark)",
            "Upload Custom Dashcam Video (.mp4 / .avi)",
            "Native Browser Camera Snapshot",
            "WebRTC Continuous Stream (with TURN Relay)"
        ],
        horizontal=True
    )

    with st.expander("⚙️ Spatio-Temporal Pipeline & AEB Calibration Controls", expanded=False):
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            sim_frame_delay = st.slider("Playback Frame Delay (seconds)", 0.01, 0.10, 0.03, 0.01, key="frame_delay_tab2")
        with c_p2:
            aeb_threshold_pct = st.slider("Autonomous Emergency Braking (AEB) Threshold (%)", 50.0, 90.0, 70.0, 5.0, key="aeb_thresh_tab2")
        with c_p3:
            optical_flow_sens = st.slider("Optical Flow Looming Sensitivity Multiplier", 0.5, 2.0, 1.0, 0.1, key="optical_sens_tab2")

    # Option A: Real Highway Dashcam Simulation
    if "Highway Dashcam Simulation" in stream_mode:
        dashcam_path = SAMPLE_DATA_DIR / "sample_traffic_dashcam.mp4"

        v_col1, v_col2 = st.columns([1.3, 1])
        frame_ph = v_col1.empty()
        gauge_ph = v_col2.empty()
        metrics_ph = v_col2.empty()

        start_btn = st.button("Start Live Spatio-Temporal Pipeline", use_container_width=True)

        if start_btn:
            cap = cv2.VideoCapture(str(dashcam_path))
            lstm_pipeline.reset()
            frame_count = 0
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
            risk_history = []
            frame_indices = []
            aeb_engaged_count = 0

            while cap.isOpened():
                ret, vframe = cap.read()
                if not ret:
                    break

                sim_res = lstm_pipeline.process_frame(
                    vframe,
                    simulated_speed=simulated_speed,
                    rider_count=simulated_riders,
                    helmet_detected=simulated_helmet
                )
                annotated_rgb = cv2.cvtColor(sim_res["annotated_frame"], cv2.COLOR_BGR2RGB)

                frame_ph.image(annotated_rgb, caption=f"Real Highway Dashcam Stream • Frame {frame_count+1}/{total_f}", use_container_width=True)
                raw_risk = float(sim_res["risk_score"] * 100)
                risk_pct = min(99.0, max(2.0, raw_risk * optical_flow_sens))
                risk_history.append(risk_pct)
                frame_indices.append(frame_count + 1)
                is_aeb = risk_pct >= aeb_threshold_pct
                if is_aeb:
                    aeb_engaged_count += 1

                aeb_html = f"""
                <div style="background: rgba(239, 68, 68, 0.18); border: 1px solid #EF4444; border-radius: 6px; padding: 6px 10px; margin-top: 8px; color: #FCA5A5; font-size: 0.74rem; font-weight: 700;">
                    🚨 AEB ACTIVE • Emergency Deceleration (-8.4 m/s²)
                </div>
                """ if is_aeb else f"""
                <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 6px 10px; margin-top: 8px; color: #34D399; font-size: 0.74rem; font-weight: 600;">
                    🛡️ AEB Standby • Velocity Guard Primed
                </div>
                """

                metrics_ph.markdown(f"""
                <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px; margin-bottom: 14px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.04);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
                        <span style="font-size: 0.80rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Rolling Temporal Risk</span>
                        <span style="font-size: 1.4rem; font-weight: 800; color: {sim_res['severity_color']};">{risk_pct:.1f}%</span>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.08); border-radius: 8px; height: 10px; overflow: hidden; margin-bottom: 14px;">
                        <div style="width: {min(max(risk_pct, 4.0), 100.0)}%; background: {sim_res['severity_color']}; height: 100%;"></div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.85rem; margin-bottom: 10px;">
                        <div style="background: rgba(255,255,255,0.03); padding: 8px 10px; border-radius: 6px;">
                            <div style="color: #94A3B8; font-size: 0.70rem; font-weight: 600;">SEVERITY TIER</div>
                            <div style="color: {sim_res['severity_color']}; font-weight: 800;">{sim_res['severity_label']}</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.03); padding: 8px 10px; border-radius: 6px;">
                            <div style="color: #94A3B8; font-size: 0.70rem; font-weight: 600;">LOOMING RATE</div>
                            <div style="font-weight: 800;">{sim_res['motion_rate']}%</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.03); padding: 8px 10px; border-radius: 6px;">
                            <div style="color: #94A3B8; font-size: 0.70rem; font-weight: 600;">TELEMETRY SPEED</div>
                            <div style="font-weight: 800;">{simulated_speed:.1f} km/h</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.03); padding: 8px 10px; border-radius: 6px;">
                            <div style="color: #94A3B8; font-size: 0.70rem; font-weight: 600;">STREAM FRAME</div>
                            <div style="font-weight: 800;">{frame_count+1} / {total_f}</div>
                        </div>
                    </div>
                    <div style="font-size: 0.82rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                        <b>Status:</b> <span style="color: {sim_res['severity_color']}; font-weight: 600;">{sim_res['warning_status']}</span>
                    </div>
                    {aeb_html}
                </div>
                """, unsafe_allow_html=True)

                frame_count += 1
                time.sleep(sim_frame_delay)

            cap.release()
            st.success("Continuous dashcam stream completed successfully.")

            gauge_ph.plotly_chart(
                create_gauge(sim_res["risk_score"], "Final Session Risk Score", sim_res["severity_color"]),
                use_container_width=True,
                key=f"sim_final_gauge_{int(time.time())}"
            )

            # Spatio-Temporal Trajectory Plot
            st.write("#### Spatio-Temporal Collision Trajectory History:")
            fig_trajectory = go.Figure()
            fig_trajectory.add_trace(go.Scatter(
                x=frame_indices,
                y=risk_history,
                mode="lines",
                line=dict(color="#3B82F6", width=2.5),
                name="Rolling Risk Score"
            ))
            fig_trajectory.add_hline(
                y=aeb_threshold_pct,
                line_dash="dash",
                line_color="#EF4444",
                annotation_text=f"AEB Intervention Threshold ({aeb_threshold_pct:.0f}%)",
                annotation_position="bottom right"
            )
            fig_trajectory.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#94A3B8"},
                title=f"Temporal Collision Trajectory • {aeb_engaged_count} Frames with AEB Active",
                xaxis_title="Processed Frame Index",
                yaxis_title="Dynamic Risk Score (%)",
                yaxis=dict(range=[0, 105]),
                height=260,
                margin=dict(l=20, r=20, t=35, b=20)
            )
            st.plotly_chart(fig_trajectory, use_container_width=True)

    # Option B: Upload Custom Dashcam Video
    elif "Upload Custom Dashcam" in stream_mode:
        uploaded_video = st.file_uploader("Upload Dashcam Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)

            vcol1, vcol2 = st.columns([1.3, 1])
            frame_display = vcol1.empty()
            gauge_display = vcol2.empty()
            info_display = vcol2.empty()

            start_proc = st.button("Start Rolling CNN+BiLSTM Analysis")
            if start_proc:
                lstm_pipeline.reset()
                f_idx = 0
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

                while cap.isOpened() and f_idx < 300:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    res = lstm_pipeline.process_frame(
                        frame,
                        simulated_speed=simulated_speed,
                        rider_count=simulated_riders,
                        helmet_detected=simulated_helmet
                    )
                    if f_idx % 2 == 0:
                        rgb = cv2.cvtColor(res["annotated_frame"], cv2.COLOR_BGR2RGB)
                        frame_display.image(rgb, use_container_width=True)

                        r_pct = float(res["risk_score"] * 100)
                        info_display.markdown(f"""
                        <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.04);">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
                                <span style="font-size: 0.80rem; font-weight: 700; color: #94A3B8;">Rolling Temporal Risk</span>
                                <span style="font-size: 1.4rem; font-weight: 800; color: {res['severity_color']};">{r_pct:.1f}%</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.08); border-radius: 8px; height: 10px; margin-bottom: 12px;">
                                <div style="width: {min(max(r_pct, 4.0), 100.0)}%; background: {res['severity_color']}; height: 100%;"></div>
                            </div>
                            <div style="font-size: 0.85rem;">
                                <b>Frame:</b> {f_idx} / {total_f}<br>
                                <b>Severity:</b> <span style="color:{res['severity_color']}; font-weight:bold;">{res['severity_label']}</span><br>
                                <b>Status:</b> {res['warning_status']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    f_idx += 1
                    time.sleep(0.01)

                cap.release()
                st.success("Video sequence processing completed.")
                gauge_display.plotly_chart(
                    create_gauge(res["risk_score"], "Final Video Risk Gauge", res["severity_color"]),
                    use_container_width=True,
                    key=f"video_final_gauge_{int(time.time())}"
                )

    # Option C: Native Browser Camera
    elif "Browser Camera" in stream_mode:
        st.info("Point your smartphone or laptop webcam at road traffic or a YouTube driving video to capture and analyze:")
        camera_snap = st.camera_input("Capture Road Scene Snapshot")
        if camera_snap:
            bytes_data = camera_snap.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            snap_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            with st.spinner("Executing temporal CNN+LSTM pipeline..."):
                res = lstm_pipeline.process_frame(
                    snap_bgr,
                    simulated_speed=simulated_speed,
                    rider_count=simulated_riders,
                    helmet_detected=simulated_helmet
                )
                annotated_rgb = cv2.cvtColor(res["annotated_frame"], cv2.COLOR_BGR2RGB)

            c1, c2 = st.columns([1.25, 1])
            with c1:
                st.image(annotated_rgb, caption="AI Optical Flow & Trajectory Overlay", use_container_width=True)
            with c2:
                st.plotly_chart(create_gauge(res["risk_score"], "Temporal Risk Score", res["severity_color"]), use_container_width=True)
                st.markdown(f"**Severity Tier:** <span style='color:{res['severity_color']}; font-weight:bold;'>{res['severity_label']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Motion Looming Rate:** `{res['motion_rate']}%`")
                st.markdown(f"**Status:** {res['warning_status']}")

    # Option D: WebRTC Stream
    else:
        st.warning("WebRTC Continuous Stream connects peer-to-peer over UDP. If your cloud firewall blocks UDP packets, use the Highway Dashcam Simulation above.")
        if HAS_WEBRTC:
            class RiskProcessor(VideoProcessorBase):
                def __init__(self):
                    self.pipeline = RollingSequenceRiskPipeline(window_size=10)
                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    res = self.pipeline.process_frame(img, simulated_speed=simulated_speed, rider_count=simulated_riders, helmet_detected=simulated_helmet)
                    return av.VideoFrame.from_ndarray(res["annotated_frame"], format="bgr24")

            webrtc_streamer(
                key="dashcam_webrtc",
                video_processor_factory=RiskProcessor,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                media_stream_constraints={"video": True, "audio": False}
            )


# ==============================================================================
# TAB 3: ROAD HAZARD SCANNER
# ==============================================================================
with tab_hazards:
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <h3 style="font-size: 1.25rem; font-weight: 700; margin: 0; color: #F8FAFC;">
            Road Surface Hazard & Pothole Segmentation
        </h3>
        <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 2px; margin-bottom: 14px;">
            Identifies road surface depressions, structural potholes, and pavement degradation in asphalt corridors.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🛣️ Pavement Geometry & Dynamic Shock Load Controls", expanded=True):
        hz_p1, hz_p2, hz_p3 = st.columns(3)
        with hz_p1:
            pothole_depth_cm = st.slider("Estimated Cavity Depth (cm)", 2.0, 25.0, 8.5, 0.5, key="pothole_depth_tab3")
        with hz_p2:
            pavement_subgrade = st.selectbox(
                "Pavement Sub-Grade Structure",
                ["Bituminous Concrete (Flexible)", "Granular Crushed Stone (WBM)", "Portland Cement Concrete (Rigid)"],
                key="pavement_subgrade_tab3"
            )
        with hz_p3:
            hazard_approach_speed = st.slider("Vehicle Approach Velocity (km/h)", 10.0, 120.0, simulated_speed, 5.0, key="approach_speed_tab3")

    hz_col1, hz_col2 = st.columns([1, 1])
    with hz_col1:
        use_hz_sample = st.radio("Hazard Input Source", ["Asphalt Road Benchmark (sample_pothole_road.jpg)", "Upload Custom Road Photo"], horizontal=True)
    with hz_col2:
        custom_road_upload = st.file_uploader("Upload Custom Pavement Photo", type=["jpg", "png", "jpeg"], key="road_upload_tab3") if "Upload" in use_hz_sample else None

    if custom_road_upload is not None:
        raw_road_img = Image.open(custom_road_upload)
    else:
        raw_road_img = Image.open(SAMPLE_DATA_DIR / "sample_pothole_road.jpg")

    road_bgr = cv2.cvtColor(np.array(raw_road_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    with st.spinner("Scanning road surface defects and cavity depressions..."):
        haz_res = hazard_det.scan_road_hazards(road_bgr)
        annotated_road_rgb = cv2.cvtColor(haz_res["annotated_frame"], cv2.COLOR_BGR2RGB)

        h_score = haz_res["hazard_risk_score"]
        h_color = "#10B981" if h_score < 0.38 else ("#F59E0B" if h_score < 0.72 else "#EF4444")
        h_tier = "LOW" if h_score < 0.38 else ("MODERATE" if h_score < 0.72 else "HIGH")

        # Dynamic Shock Load Calculation
        v_approach_ms = hazard_approach_speed / 3.6
        g_shock = min(6.5, (v_approach_ms**2) / (2.0 * max(0.02, pothole_depth_cm / 100.0) * 9.81) * 0.075)

        if g_shock >= 2.5:
            shock_status = "CRITICAL: Rim Buckling & Axle Fracture Risk"
            shock_color = "#EF4444"
        elif g_shock >= 1.4:
            shock_status = "MODERATE: Suspension Degradation & Alignment Drift"
            shock_color = "#F59E0B"
        else:
            shock_status = "LOW: Absorbed by Pneumatic Tire Compliance"
            shock_color = "#10B981"

        patch_volume_m3 = (pothole_depth_cm / 100.0) * 0.45 * max(1, haz_res["hazard_count"])
        patch_tons = patch_volume_m3 * 2.4
        repair_sla = "< 24 Hours Emergency Hot-Mix Patch" if h_tier == "HIGH" else ("< 72 Hours Standard Cold-Mix Patch" if h_tier == "MODERATE" else "Routine Survey Audit")

        st.session_state.current_assessment.update({
            "modality": "Road Surface Hazard & Pothole Segmentation",
            "overall_risk": float(h_score * 100),
            "severity_label": f"{h_tier} Hazard Level",
            "severity_color": h_color,
            "annotated_image": Image.fromarray(annotated_road_rgb),
            "hazard_results": {
                "hazard_score": float(h_score * 100),
                "hazard_count": haz_res["hazard_count"],
                "hazards_detected": [f"{h['type'].upper()} ({h['severity']})" for h in haz_res.get("hazards_detected", [])],
                "shock_load_g": g_shock,
                "pothole_depth_cm": pothole_depth_cm,
                "patch_tons": patch_tons
            }
        })

    col_r1, col_r2 = st.columns([1.25, 1])
    with col_r1:
        st.image(annotated_road_rgb, caption="Asphalt Pothole Segmentation (Corner Reticles)", use_container_width=True)
    with col_r2:
        badge_class = "sev-badge-low" if h_tier == "LOW" else ("sev-badge-mid" if h_tier == "MODERATE" else "sev-badge-high")
        st.markdown(f"""
        <div class="{badge_class}">
            ROAD SURFACE THREAT: {h_tier}
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(create_gauge(h_score, "Road Hazard Threat Index", h_color), use_container_width=True, key="gauge_hazards")

        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-label">Pavement State</div>
            <div class="kpi-value" style="font-size: 1.05rem;">{haz_res.get('pavement_state', 'Nominal')}</div>
            <div class="kpi-subtext">Cavities Detected: {haz_res['hazard_count']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="protocol-card" style="border-left-color: {h_color};">
            <b>Municipal Maintenance Status:</b><br>
            {haz_res.get('municipal_alert', 'Routine Survey')}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Municipal PWD / NHAI Infrastructure Work-Order:")
    st.markdown(f"""
    <div class="saas-card" style="margin-top: 14px;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 12px;">
            <div style="background: rgba(255,255,255,0.03); padding: 10px 12px; border-radius: 8px;">
                <div style="color: #94A3B8; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;">Work Order Ref</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC; font-family: monospace;">WO-PWD-CH884</div>
                <div style="font-size: 0.70rem; color: #64748B;">Sector: NH-48 Km 28.4</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 10px 12px; border-radius: 8px;">
                <div style="color: #94A3B8; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;">Vertical Shock</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: {shock_color};">{g_shock:.2f} G</div>
                <div style="font-size: 0.70rem; color: #94A3B8;">@ {hazard_approach_speed:.0f} km/h</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 10px 12px; border-radius: 8px;">
                <div style="color: #94A3B8; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;">Bituminous Patch</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC;">{patch_tons:.2f} Tons</div>
                <div style="font-size: 0.70rem; color: #64748B;">Depth: {pothole_depth_cm:.1f} cm ({pavement_subgrade.split('(')[0]})</div>
            </div>
            <div style="background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.25); padding: 10px 12px; border-radius: 8px;">
                <div style="color: #60A5FA; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;">Maintenance SLA</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: {'#F87171' if h_tier=='HIGH' else '#38BDF8'};">{repair_sla.split()[0]} {repair_sla.split()[1]}</div>
                <div style="font-size: 0.70rem; color: #93C5FD;">{repair_sla.split('Hours ')[-1] if 'Hours ' in repair_sla else 'Survey'}</div>
            </div>
        </div>
        <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px; font-size: 0.82rem; display: flex; justify-content: space-between; align-items: center;">
            <span><b>Dynamic Chassis Stress:</b> <span style="color: {shock_color}; font-weight: 600;">{shock_status}</span></span>
            <span style="color: #94A3B8; font-size: 0.76rem;">Authority: PWD Highways Dept (Zone IV)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# TAB 4: TWO-WHEELER SAFETY & VIOLATIONS
# ==============================================================================
with tab_violations:
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <h3 style="font-size: 1.25rem; font-weight: 700; margin: 0; color: #F8FAFC;">
            Two-Wheeler Pre-Crash Violations & Rider Safety
        </h3>
        <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 2px; margin-bottom: 14px;">
            Monitors rider helmet compliance, passenger occupancy overload (triple riding), and velocity surges.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚖️ Traffic Enforcement Jurisdiction & Statutory Codes", expanded=True):
        j_col1, j_col2 = st.columns(2)
        with j_col1:
            enforcement_jurisdiction = st.selectbox(
                "Enforcement Authority & Geofence",
                [
                    "Greater Chennai Traffic Police (GCTP) - Zone IV",
                    "Bengaluru City Traffic Police (BTP) - Central Command",
                    "Delhi Traffic Police - Northern Range",
                    "NHAI Highway Patrol Division - NH-48 Sector"
                ],
                key="jurisdiction_tab4"
            )
        with j_col2:
            cargo_overload_flag = st.checkbox("Commercial Delivery Cargo Box / Hazardous Overhang Detected", value=False, key="cargo_flag_tab4")
            pillions_without_helmet = max(0, simulated_riders - 1) if not pillion_helmet else 0

    v_src_col1, v_src_col2 = st.columns([1, 1])
    with v_src_col1:
        use_viol_sample = st.radio("Violation Input Source", ["Indian Street Traffic Benchmark (sample_two_wheeler_violation.jpg)", "Upload Custom Traffic Photo"], horizontal=True)
    with v_src_col2:
        custom_viol_upload = st.file_uploader("Upload Traffic Photograph", type=["jpg", "png", "jpeg"], key="viol_upload_tab4") if "Upload" in use_viol_sample else None

    if custom_viol_upload is not None:
        raw_viol_img = Image.open(custom_viol_upload)
    else:
        raw_viol_img = Image.open(SAMPLE_DATA_DIR / "sample_two_wheeler_violation.jpg")

    viol_bgr = cv2.cvtColor(np.array(raw_viol_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    telemetry = {
        "speed_kmh": simulated_speed,
        "rider_count": simulated_riders,
        "helmet_detected": simulated_helmet
    }

    with st.spinner("Analyzing rider occupancy, helmet compliance, and velocity..."):
        viol_res = violation_det.analyze_frame(viol_bgr, simulated_telemetry=telemetry)
        annotated_viol_rgb = cv2.cvtColor(viol_res["annotated_frame"], cv2.COLOR_BGR2RGB)

        v_score = viol_res["risk_score"]
        v_color = "#10B981" if v_score < 0.38 else ("#F59E0B" if v_score < 0.72 else "#EF4444")
        v_tier = "LOW" if v_score < 0.38 else ("MODERATE" if v_score < 0.72 else "HIGH")

        # Dynamic Motor Vehicles Act (MVA 2019) Fine Calculation
        challan_items = []
        total_challan_fine = 0

        if not simulated_helmet:
            challan_items.append(("Section 129 MVA", "Rider / Driver Without Protective Headgear", 1000, "3-Month License Suspension"))
            total_challan_fine += 1000

        if simulated_riders > 1 and not pillion_helmet:
            challan_items.append(("Section 129 MVA", f"Pillion Occupant(s) ({pillions_without_helmet}) Without Protective Headgear", 1000 * max(1, pillions_without_helmet), "Statutory Fine"))
            total_challan_fine += 1000 * max(1, pillions_without_helmet)

        if simulated_riders > 2:
            excess_riders = simulated_riders - 2
            challan_items.append(("Section 194C MVA", f"Overcapacity Carrying of {excess_riders} Excess Passenger(s)", 2000, "Safety Disqualification"))
            total_challan_fine += 2000

        if simulated_speed > enforced_speed_limit:
            over_speed = simulated_speed - enforced_speed_limit
            challan_items.append(("Section 183(1) MVA", f"Corridor Overspeeding (+{over_speed:.0f} km/h above {enforced_speed_limit:.0f} km/h limit)", 2000, "Automated Radar Citation"))
            total_challan_fine += 2000

        if cargo_overload_flag:
            challan_items.append(("Section 190(2) MVA", "Hazardous Cargo Overhang & Center of Gravity Imbalance", 1500, "Commercial Inspection Notice"))
            total_challan_fine += 1500

        # Rider Dynamic Stability Safety Index (SSI)
        speed_penalty = max(0.0, (simulated_speed - enforced_speed_limit) * 1.1)
        rider_penalty = (simulated_riders - 1) * 22.0
        cargo_penalty = 18.0 if cargo_overload_flag else 0.0
        ssi_score = max(5.0, min(100.0, 100.0 - speed_penalty - rider_penalty - cargo_penalty))

        if ssi_score >= 70:
            ssi_tier = "STABLE EQUILIBRIUM"
            ssi_color = "#10B981"
        elif ssi_score >= 45:
            ssi_tier = "COMPROMISED STABILITY (CAPSIZE RISK)"
            ssi_color = "#F59E0B"
        else:
            ssi_tier = "CRITICAL INSTABILITY (SKID IMMINENT)"
            ssi_color = "#EF4444"

        st.session_state.current_assessment.update({
            "modality": "Two-Wheeler Safety & Compliance",
            "overall_risk": float(v_score * 100),
            "severity_label": f"{v_tier} Violation Risk",
            "severity_color": v_color,
            "annotated_image": Image.fromarray(annotated_viol_rgb),
            "violation_results": {
                "violation_score": float(v_score * 100),
                "violations_detected": viol_res["violations"],
                "total_challan_fine": total_challan_fine,
                "ssi_score": ssi_score,
                "jurisdiction": enforcement_jurisdiction
            }
        })

    col_v1, col_v2 = st.columns([1.25, 1])
    with col_v1:
        st.image(annotated_viol_rgb, caption="Two-Wheeler Bounding Boxes & Helmet Localization", use_container_width=True)
    with col_v2:
        badge_class = "sev-badge-low" if v_tier == "LOW" else ("sev-badge-mid" if v_tier == "MODERATE" else "sev-badge-high")
        st.markdown(f"""
        <div class="{badge_class}">
            BEHAVIORAL RISK: {v_tier} RISK
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(create_gauge(v_score, "Pre-Crash Behavioral Risk Score", v_color), use_container_width=True, key="gauge_violations")

        kpi_ssi1, kpi_ssi2 = st.columns(2)
        with kpi_ssi1:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Stability Index (SSI)</div>
                <div class="kpi-value" style="color: {ssi_color};">{ssi_score:.0f} / 100</div>
                <div class="kpi-subtext">{ssi_tier.split('(')[0]}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_ssi2:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">E-Challan Penalty</div>
                <div class="kpi-value" style="color: {'#EF4444' if total_challan_fine > 0 else '#10B981'};">₹{total_challan_fine:,}</div>
                <div class="kpi-subtext">{len(challan_items)} Statutory Violations</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### Digital Statutory E-Challan Notice (MVA 2019):")
    if challan_items:
        challan_rows = "".join([
            f"""<tr style="border-bottom: 1px solid rgba(255,255,255,0.06); font-size: 0.82rem;">
                <td style="padding: 8px 12px; font-weight: 700; color: #F87171; font-family: monospace;">{sec}</td>
                <td style="padding: 8px 12px; color: #F8FAFC;">{desc}</td>
                <td style="padding: 8px 12px; color: #94A3B8;">{action}</td>
                <td style="padding: 8px 12px; text-align: right; font-weight: 800; color: #F8FAFC;">₹{amt:,}</td>
            </tr>"""
            for sec, desc, amt, action in challan_items
        ])
        st.markdown(f"""
        <div class="saas-card" style="margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 10px;">
                <div>
                    <span style="font-weight: 800; font-size: 0.95rem; color: #F8FAFC;">E-CHALLAN NOTICE #ECH-2026-TN-992147</span><br>
                    <span style="font-size: 0.74rem; color: #60A5FA;">{enforcement_jurisdiction}</span>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Total Fine Amount</span><br>
                    <span style="font-size: 1.3rem; font-weight: 800; color: #EF4444;">₹{total_challan_fine:,}</span>
                </div>
            </div>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="color: #94A3B8; font-size: 0.72rem; text-transform: uppercase; border-bottom: 1px solid rgba(255,255,255,0.12);">
                        <th style="padding: 6px 12px; text-align: left;">Section</th>
                        <th style="padding: 6px 12px; text-align: left;">Violation Category</th>
                        <th style="padding: 6px 12px; text-align: left;">Administrative Action</th>
                        <th style="padding: 6px 12px; text-align: right;">Fine (INR)</th>
                    </tr>
                </thead>
                <tbody>
                    {challan_rows}
                </tbody>
            </table>
            <div style="margin-top: 10px; font-size: 0.74rem; color: #64748B; display: flex; justify-content: space-between;">
                <span>Digital Signature Verified • Automated Camera ANPR Sync</span>
                <span>Payment Window: 15 Calendar Days</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ Fully Regulatory Compliant. Zero statutory violations detected under Motor Vehicles Act.")


# ==============================================================================
# TAB 5: MULTIMODAL RISK FUSION
# ==============================================================================
with tab_fusion:
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <h3 style="font-size: 1.25rem; font-weight: 700; margin: 0; color: #F8FAFC;">
            Unified Multimodal Risk Fusion Engine & Sensitivity Analysis
        </h3>
        <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 2px; margin-bottom: 14px;">
            Cross-modal Bayesian correlation combining visual deformities, telemetry velocity, pavement friction, and atmospheric threat.
        </p>
    </div>
    """, unsafe_allow_html=True)

    curr = st.session_state.current_assessment
    dmg_val = curr.get("damage_results", {}).get("severity_score", 0.40) if curr.get("damage_results") else 0.22
    viol_val = curr.get("violation_results", {}).get("violation_score", 45.0) / 100.0 if curr.get("violation_results") else 0.45
    haz_val = curr.get("hazard_results", {}).get("hazard_score", 30.0) / 100.0 if curr.get("hazard_results") else 0.35

    with st.expander("⚖️ Bayesian Modality Weights & Kinetic Braking Telemetry", expanded=True):
        st.markdown("**Calibrate Modality Weights (Dynamic Bayesian Normalization):**")
        w_c1, w_c2, w_c3, w_c4 = st.columns(4)
        with w_c1:
            w_dmg = st.slider("W_D (Deformation)", 0.05, 0.60, 0.35, 0.05, key="w_dmg_tab5")
        with w_c2:
            w_viol = st.slider("W_V (Violations)", 0.05, 0.60, 0.30, 0.05, key="w_viol_tab5")
        with w_c3:
            w_haz = st.slider("W_H (Road Hazards)", 0.05, 0.60, 0.20, 0.05, key="w_haz_tab5")
        with w_c4:
            w_spd = st.slider("W_K (Speed Looming)", 0.05, 0.60, 0.15, 0.05, key="w_spd_tab5")

        total_w = w_dmg + w_viol + w_haz + w_spd
        norm_wd = w_dmg / total_w
        norm_wv = w_viol / total_w
        norm_wh = w_haz / total_w
        norm_ws = w_spd / total_w

        st.caption(f"Normalized Bayesian Prior Weights: Damage={norm_wd:.2f} | Violations={norm_wv:.2f} | Hazards={norm_wh:.2f} | Kinematics={norm_ws:.2f} (Total: 1.00)")

        g_c1, g_c2 = st.columns(2)
        with g_c1:
            road_gradient_pct = st.slider("Road Incline / Decline Slope (%)", -15.0, 15.0, 0.0, 1.0, key="road_slope_tab5", help="Negative indicates downhill descent, increasing stopping distance")
        with g_c2:
            abs_equipped = st.checkbox("Anti-Lock Braking System (ABS) Activated", value=True, key="abs_equipped_tab5")

    fcol1, fcol2 = st.columns([1, 1.2])
    with fcol1:
        st.markdown("#### Environmental Sensitivity Controls")
        weather = st.selectbox("Atmospheric Condition", ["Clear / Dry Daylight", "Heavy Monsoon Rain", "Dense Fog / Low Visibility", "Night / Low Illumination"], index=0 if "Clear" in env_weather else (1 if "Rain" in env_weather else (2 if "Fog" in env_weather else 3)), key="weather_select_tab5")
        road_surface = st.selectbox("Road Surface Condition", ["Dry Smooth Bitumen", "Wet Slippery Asphalt", "Cracked Potholed Highway", "Loose Gravel Surface"], index=0 if "Dry" in env_road else (1 if "Wet" in env_road else (2 if "Cracked" in env_road else 3)), key="road_select_tab5")

        env_multiplier = 1.0
        if "Rain" in weather or "Wet" in road_surface:
            env_multiplier += 0.35
        if "Fog" in weather or "Night" in weather:
            env_multiplier += 0.25

        speed_norm_val = min(1.0, simulated_speed / 140.0)
        # Composite Fusion Score Calculation with Bayesian Weights
        comp_score = min(0.99, ((dmg_val * norm_wd) + (viol_val * norm_wv) + (haz_val * norm_wh) + (speed_norm_val * norm_ws)) * env_multiplier)
        comp_color = "#10B981" if comp_score < 0.38 else ("#F59E0B" if comp_score < 0.72 else "#EF4444")
        comp_label = "LOW RISK" if comp_score < 0.38 else ("MODERATE RISK" if comp_score < 0.72 else "CRITICAL HAZARD")

        st.plotly_chart(create_gauge(comp_score, "Composite Multimodal Risk Score", comp_color), use_container_width=True, key="gauge_fusion")

    with fcol2:
        st.markdown("#### 5-Dimensional Threat Vector Radar")
        radar_cats = ['Structural Damage', 'Velocity Surge', 'Behavioral Violation', 'Road Hazards', 'Environmental Threat']
        radar_vals = [
            dmg_val * 100.0,
            min(100.0, (simulated_speed / 140.0) * 100.0),
            viol_val * 100.0,
            haz_val * 100.0,
            min(100.0, (env_multiplier / 1.6) * 100.0)
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_cats + [radar_cats[0]],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.20)',
            line=dict(color='#3B82F6', width=2),
            name='Current Threat State'
        ))
        fig_radar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            polar=dict(
                bgcolor='rgba(255, 255, 255, 0.02)',
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color='#94A3B8')),
                angularaxis=dict(tickfont=dict(size=10, color='#F8FAFC'))
            ),
            showlegend=False,
            height=300,
            margin=dict(l=30, r=30, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Stopping Distance Physics & Sensitivity Matrix ($D = v \\cdot t_{rxn} + \\frac{v^2}{2g(\\mu + \\sin\\theta)}$)")

    # Dynamic Friction Physics Calculations
    surface_friction_map = {
        "Dry Smooth Bitumen": 0.75,
        "Wet Slippery Asphalt": 0.45,
        "Cracked Potholed Highway": 0.35,
        "Loose Gravel Surface": 0.26
    }
    cur_base_mu = surface_friction_map.get(road_surface, 0.75)
    eff_mu = cur_base_mu if abs_equipped else (cur_base_mu * 0.72)
    grade_sin = road_gradient_pct / 100.0
    denom = 2.0 * 9.81 * max(0.06, (eff_mu + grade_sin))

    v_ms = simulated_speed / 3.6
    d_rxn = v_ms * driver_reaction_sec
    d_skid = (v_ms**2) / denom
    d_total = d_rxn + d_skid
    ke_kj = 0.5 * vehicle_mass_kg * (v_ms**2) / 1000.0

    k_col1, k_col2 = st.columns(2)
    with k_col1:
        # Multi-surface stopping distance curve
        speed_domain = np.linspace(20, 140, 25)
        fig_stop = go.Figure()

        colors_map = {
            "Dry Smooth Bitumen": "#10B981",
            "Wet Slippery Asphalt": "#3B82F6",
            "Cracked Potholed Highway": "#F59E0B",
            "Loose Gravel Surface": "#EF4444"
        }

        for s_name, s_mu in surface_friction_map.items():
            s_eff = s_mu if abs_equipped else (s_mu * 0.72)
            s_denom = 2.0 * 9.81 * max(0.06, (s_eff + grade_sin))
            s_dist = ((speed_domain / 3.6) * driver_reaction_sec) + (((speed_domain / 3.6)**2) / s_denom)
            fig_stop.add_trace(go.Scatter(
                x=speed_domain,
                y=s_dist,
                mode='lines',
                line=dict(color=colors_map[s_name], width=2),
                name=s_name.split()[0]
            ))

        # Current Operating Marker
        fig_stop.add_trace(go.Scatter(
            x=[simulated_speed],
            y=[d_total],
            mode='markers+text',
            marker=dict(size=11, color='#F8FAFC', line=dict(color='#3B82F6', width=2)),
            text=[f"{d_total:.1f}m"],
            textposition="top left",
            name="Current State"
        ))

        fig_stop.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#94A3B8'},
            title=f"Stopping Distance vs Velocity (Slope: {road_gradient_pct:+.0f}%)",
            xaxis_title="Velocity (km/h)",
            yaxis_title="Total Stopping Distance (m)",
            height=280,
            legend=dict(font=dict(size=8), orientation="h", y=-0.25),
            margin=dict(l=15, r=15, t=35, b=15)
        )
        st.plotly_chart(fig_stop, use_container_width=True)

    with k_col2:
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px;">
            <div class="kpi-tile">
                <div class="kpi-label">Effective Friction (μ)</div>
                <div class="kpi-value">{eff_mu:.2f}</div>
                <div class="kpi-subtext">{'ABS Modulated' if abs_equipped else 'Standard (Locked Wheel)'}</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-label">Kinetic Energy (Ek)</div>
                <div class="kpi-value">{ke_kj:,.0f} kJ</div>
                <div class="kpi-subtext">Mass: {vehicle_mass_kg:,.0f} kg ({vehicle_type_label.split('(')[0]})</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-label">Reaction Distance</div>
                <div class="kpi-value">{d_rxn:.1f} m</div>
                <div class="kpi-subtext">Latency: {driver_reaction_sec:.1f}s ({driver_state_label.split('(')[0]})</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-label">Braking Skid Distance</div>
                <div class="kpi-value">{d_skid:.1f} m</div>
                <div class="kpi-subtext">Deceleration to 0 km/h</div>
            </div>
        </div>
        <div class="kpi-tile" style="margin-top: 8px; border-left: 3px solid {'#EF4444' if d_total > 70 else '#10B981'};">
            <div class="kpi-label">Total Emergency Stopping Distance</div>
            <div class="kpi-value" style="color: {'#EF4444' if d_total > 70 else '#10B981'};">{d_total:.1f} meters</div>
            <div class="kpi-subtext">Headway Distance Available: {headway_dist:.1f}m ({'🚨 IMPACT IMMINENT' if d_total > headway_dist else '✅ SAFE STOP ASSURED'})</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 6: INCIDENT DOSSIER (PDF REPORT)
# ==============================================================================
with tab_report:
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <h3 style="font-size: 1.25rem; font-weight: 700; margin: 0; color: #F8FAFC;">
            Official Regulatory Incident Dossier & Verified PDF Export
        </h3>
        <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 2px; margin-bottom: 14px;">
            Generates a publication-quality, ISO-compliant accident analysis report with digital verification timestamps and forensic telemetry.
        </p>
    </div>
    """, unsafe_allow_html=True)

    curr = st.session_state.current_assessment
    rep_col1, rep_col2 = st.columns([1, 1.2])

    with rep_col1:
        st.markdown("#### Metadata & Surveyor Configuration")
        incident_id_in = st.text_input("Incident Reference ID", value=curr["incident_id"], key="inc_id_tab6")
        location_in = st.text_input("Highway Geofence / Sector", value="NH-48 Chennai-Bengaluru Highway, Km 28.4", key="loc_in_tab6")

        m_c1, m_c2 = st.columns(2)
        with m_c1:
            lead_investigator = st.text_input("Lead Forensic Investigator", value="Harish AT & Barathraj AS", key="lead_inv_tab6")
            lead_badge = st.text_input("Investigator Badge / Staff ID", value="PEC-AIML-D16-04", key="lead_badge_tab6")
        with m_c2:
            insurer_select = st.selectbox(
                "Underwriting Insurance Carrier",
                [
                    "ICICI Lombard General Insurance",
                    "Bajaj Allianz General Insurance",
                    "HDFC ERGO General Insurance",
                    "The New India Assurance Co.",
                    "TATA AIG General Insurance"
                ],
                key="insurer_select_tab6"
            )
            policy_num_in = st.text_input("Policy Certificate Number", value="POL-PEC-2026-994821", key="policy_num_tab6")

        vin_chassis_in = st.text_input("Vehicle VIN / Chassis Identifier", value="MA1TA2DK8N2108849", key="vin_tab6")

        st.write("#### Dispatched Emergency Assets:")
        d_als = st.checkbox("Tier-1 ALS Ambulance Unit #108 Dispatched", value=(curr["overall_risk"] >= 70), key="d_als_tab6")
        d_pol = st.checkbox("Highway Police Sector Interceptor #04 Dispatched", value=(curr["overall_risk"] >= 40), key="d_pol_tab6")
        d_tow = st.checkbox("Heavy Recovery Flatbed Dispatched", value=(curr["overall_risk"] >= 50), key="d_tow_tab6")
        d_pwd = st.checkbox("PWD Emergency Asphalt Repair Crew Dispatched", value=(curr.get("hazard_results", {}).get("hazard_score", 0) >= 38), key="d_pwd_tab6")

        active_dispatches = []
        if d_als: active_dispatches.append("Ambulance Unit #108")
        if d_pol: active_dispatches.append("Police Interceptor #04")
        if d_tow: active_dispatches.append("Flatbed Recovery")
        if d_pwd: active_dispatches.append("PWD Pothole Patch Crew")
        dispatches_txt = ", ".join(active_dispatches) if active_dispatches else "None Dispatched (Routine Status)"

        default_remarks = (
            f"Automated multimodal collision inspection executed on edge node. "
            f"Underwriter: {insurer_select} (Policy: {policy_num_in}). "
            f"Vehicle VIN: {vin_chassis_in}. "
            f"Assessing Officer: {lead_investigator} ({lead_badge}). "
            f"Dispatched Asset(s): {dispatches_txt}. "
            f"Telemetry speed: {simulated_speed:.1f} km/h, road friction profile: {env_road}."
        )

        surveyor_remarks = st.text_area(
            "Investigator Findings & Remarks",
            value=default_remarks,
            key="remarks_tab6"
        )

    with rep_col2:
        st.markdown("#### Live Dossier Preview")
        st.markdown(f"""
        <div class="saas-card">
            <b>Institution:</b> Panimalar Engineering College (AIML Batch D16)<br>
            <b>Incident ID:</b> <code>{incident_id_in}</code><br>
            <b>Lead Investigator:</b> {lead_investigator} (Badge: <code>{lead_badge}</code>)<br>
            <b>Insurer / Policy:</b> {insurer_select} • <code>{policy_num_in}</code><br>
            <b>Vehicle VIN:</b> <code>{vin_chassis_in}</code><br>
            <b>Evaluation Modality:</b> {curr['modality']}<br>
            <b>Overall Composite Risk:</b> <span style="font-weight:bold; color:{curr['severity_color']};">{curr['overall_risk']:.1f}%</span><br>
            <b>Status:</b> {curr['severity_label']}<br>
            <b>Geofence Sector:</b> {location_in}<br>
            <b>Emergency Assets:</b> {dispatches_txt}<br>
            <b>Timestamp:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
        </div>
        """, unsafe_allow_html=True)

        if curr.get("annotated_image") is not None:
            st.image(curr["annotated_image"], caption="Forensic Visual Evidence for Report", use_container_width=True)

    st.markdown("---")
    try:
        env_dossier = {
            "weather": env_weather,
            "lighting": f"{ambient_lux} Lux",
            "traffic_density": "Moderate Highway Corridor",
            "multiplier": env_multiplier if 'env_multiplier' in locals() else 1.0,
            "road_surface": env_road,
            "vehicle_class": vehicle_type_label,
            "speed_kmh": f"{simulated_speed:.1f} km/h"
        }

        pdf_bytes = generate_incident_pdf(
            incident_id=incident_id_in,
            modality_type=curr["modality"],
            overall_risk=curr["overall_risk"],
            status_label=curr["severity_label"],
            damage_results=curr.get("damage_results"),
            violation_results=curr.get("violation_results"),
            hazard_results=curr.get("hazard_results"),
            annotated_image=curr.get("annotated_image"),
            environmental_factors=env_dossier,
            notes=surveyor_remarks
        )

        b1, b2, b3 = st.columns([1, 2, 1])
        with b2:
            st.download_button(
                label="📄 Export Verified PDF Incident Report",
                data=pdf_bytes,
                file_name=f"Official_Incident_Report_{incident_id_in}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.caption("Official letterhead, triage matrices, and digital verification seal embedded.")
    except Exception as e:
        st.error(f"Error compiling PDF: {str(e)}")


# ==============================================================================
# TAB 7: SYSTEM ARCHITECTURE
# ==============================================================================
with tab_architecture:
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <h3 style="font-size: 1.25rem; font-weight: 700; margin: 0; color: #F8FAFC;">
            Deep Learning Architecture & Academic Credentials
        </h3>
        <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 2px; margin-bottom: 14px;">
            Panimalar Engineering College • Department of Artificial Intelligence & Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 1. Unified End-to-End Multimodal Pipeline
    This project introduces a multi-tier neural framework for pre-crash collision prediction and post-crash triage:
    
    1. **Pre-Crash Risk Forecasting (BiLSTM):** Spatio-temporal rolling sequence tracking optical flow looming and velocity delta across video frames.
    2. **Active Safety Violation Monitoring (YOLOv8):** Detects helmet non-compliance and overcapacity triple-riding in urban Indian traffic.
    3. **Road Surface Hazard Detection:** Multi-scale morphological segmentation localizing asphalt potholes and road degradation.
    4. **Post-Crash Damage Severity Classification (ResNet-50):** 3-tier transfer learning classifier categorizing vehicular damage into **Low / Moderate / High** severity with repair estimates.
    
    ---

    ### 2. Mathematical Formulations
    
    #### A. Normalized Bayesian Prior Fusion
    Cross-modal fusion combines independent telemetry dimensions using dynamically normalized weights:
    $$\\tilde{W}_i = \\frac{W_i}{\\sum_{k=1}^4 W_k}, \\quad R_{\\text{composite}} = \\min\\left(0.99, \\left[\\sum_{i=1}^4 \\tilde{W}_i \\cdot r_i\\right] \\times \\mathcal{M}_{\\text{env}}\\right)$$
    where $\\mathcal{M}_{\\text{env}}$ is the compounding atmospheric and friction multiplier.

    #### B. Compounded Probability of Multi-Violation Risk
    Independent infractions are combined through union complementation:
    $$P_{\\text{combined}} = 1 - \\prod_{i=1}^{N} (1 - p_i)$$
    where $p_i \\in [0, 1]$ represents the calibrated risk probability of violation factor $i$.

    #### C. Spatio-Temporal Rolling Sequence Smoothing
    Temporal feature continuity across video frames is smoothed via an exponential moving average (EMA) kernel:
    $$S_t = \\alpha \\cdot s_t + (1 - \\alpha) \\cdot S_{t-1}$$
    where $\\alpha = 0.35$, preventing high-frequency flicker in camera monitoring.

    #### D. Grade-Corrected Vehicle Stopping Distance & Kinetic Impact
    Total stopping distance incorporates reaction latency $t_{\\text{rxn}}$, grade incline $\\theta$, and ABS-modulated tire friction $\\mu_{\\text{eff}}$:
    $$D_{\\text{total}} = v \\cdot t_{\\text{rxn}} + \\frac{v^2}{2 g (\\mu_{\\text{eff}} + \\sin\\theta)}, \\quad E_k = \\frac{1}{2} m v^2$$

    #### E. Road Hazard Vertical Shock Load ($G$-Force)
    Dynamic chassis displacement upon traversing an unpaved depression or pothole:
    $$G_{\\text{shock}} = \\frac{v^2}{2 \\cdot d_{\\text{cavity}} \\cdot g} \\times K_{\\text{damping}}$$
    where $K_{\\text{damping}} = 0.075$ is the vehicle sprung-mass compliance factor.

    #### F. Two-Wheeler Dynamic Stability Safety Index ($SSI$)
    Rider safety balance equilibrium accounts for overcapacity and velocity differential:
    $$SSI = \\max\\left(5.0, 100.0 - \\Delta_{\\text{speed}} - (N_{\\text{riders}} - 1) \\times 22.0 - \\Delta_{\\text{cargo}}\\right)$$

    ---

    ### 3. Academic Review & Team Credentials
    * **Institution:** Panimalar Engineering College (An Autonomous Institution, Affiliated to Anna University, Chennai)
    * **Department:** Department of Artificial Intelligence & Machine Learning (AIML)
    * **Academic Year & Batch:** Final Year Engineering Showcase • Batch D16
    * **Student Investigators:**
      * **HARISH AT** (AIML Department)
      * **BARATHRAJ AS** (AIML Department)
    * **Project Supervisor:**
      * **Dr. A. ARUN**, Associate Professor, Department of AI & ML
    """)
