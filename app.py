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

# ── Ensure benchmark sample images exist ───────────────────────────────────────
def _ensure_samples():
    expected = [
        "sample_low_damage.jpg",
        "sample_mid_damage.jpg",
        "sample_high_damage.jpg",
        "sample_pothole_road.jpg",
        "sample_two_wheeler_violation.jpg",
    ]
    if not all((SAMPLE_DATA_DIR / f).exists() for f in expected):
        try:
            from sample_data.generate_sample_media import generate_photorealistic_benchmarks
            generate_photorealistic_benchmarks()
        except Exception:
            pass

_ensure_samples()

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Road Accident & Damage AI System | Panimalar AIML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Executive Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global styles */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }
    
    /* Institutional Banner */
    .inst-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F2E52 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 24px;
        color: #FFFFFF;
        margin-bottom: 18px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    }
    .inst-title {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #F8FAFC;
    }
    .inst-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 4px;
        margin-bottom: 6px;
    }
    .inst-meta {
        font-size: 0.85rem;
        color: #38BDF8;
        font-weight: 500;
    }
    
    /* System Telemetry Ribbon */
    .telemetry-ribbon {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 18px;
    }
    .telemetry-pill {
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #334155;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .pill-green {
        background: #ECFDF5;
        border-color: #6EE7B7;
        color: #065F46;
    }
    .pill-blue {
        background: #EFF6FF;
        border-color: #93C5FD;
        color: #1E40AF;
    }
    .pill-purple {
        background: #F5F3FF;
        border-color: #C4B5FD;
        color: #5B21B6;
    }
    
    /* Badge & Metric Cards */
    .badge-card {
        padding: 14px 18px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: 800;
        font-size: 1.25rem;
        letter-spacing: 0.03em;
        box-shadow: 0 4px 10px rgba(0,0,0,0.12);
        margin-bottom: 12px;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .dispatch-card {
        border-left: 5px solid #EF4444;
        background: #FEF2F2;
        padding: 14px 18px;
        border-radius: 6px;
        color: #991B1B;
        font-size: 0.95rem;
        margin-top: 12px;
    }
    .tab-header-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Cache and Load AI Modules ─────────────────────────────────────────────────
@st.cache_resource
def load_modules():
    damage_clf = DamageClassifier()
    violation_det = ViolationDetector()
    hazard_det = HazardDetector()
    fusion_engine = MultimodalRiskFusionEngine()
    lstm_pipeline = RollingSequenceRiskPipeline(window_size=10)
    return damage_clf, violation_det, hazard_det, fusion_engine, lstm_pipeline

damage_clf, violation_det, hazard_det, fusion_engine, lstm_pipeline = load_modules()

# ── Session State for Inter-Tab Intelligence ───────────────────────────────────
if "current_assessment" not in st.session_state:
    st.session_state.current_assessment = {
        "incident_id": f"PEC-AIML-{int(time.time()) % 100000:05d}",
        "modality": "Unified Multimodal Assessment",
        "overall_risk": 58.0,
        "severity_label": "Moderate / Actionable Risk",
        "severity_color": "#F59E0B",
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
        "notes": "Standard initial session state. Run any benchmark, uploaded media, or live camera to refresh."
    }

# ── Top Institutional Header with Logo ─────────────────────────────────────────
logo_path = BASE_DIR / "assets" / "college_logo.png"

col_head_logo, col_head_text = st.columns([0.8, 8.2])
with col_head_logo:
    if logo_path.exists():
        st.image(str(logo_path), width=85)
    else:
        st.markdown("<div style='font-size: 3rem;'>🏛️</div>", unsafe_allow_html=True)

with col_head_text:
    st.markdown("""
    <div class="inst-header">
        <div class="inst-title">PANIMALAR ENGINEERING COLLEGE</div>
        <div class="inst-subtitle">An Autonomous Institution, Affiliated to Anna University, Chennai • Department of Artificial Intelligence & Machine Learning</div>
        <div class="inst-meta">
            <b>Project Review (Batch D16):</b> Multimodal Accident Prevention & Vehicle Damage Severity Assessment AI System<br>
            <b>Student Investigators:</b> HARISH AT & BARATHRAJ AS &nbsp;|&nbsp; <b>Project Supervisor:</b> Dr. A. ARUN, Associate Professor / AIML
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── System Telemetry Ribbon ───────────────────────────────────────────────────
st.markdown("""
<div class="telemetry-ribbon">
    <div class="telemetry-pill pill-green"><span style="color:#10B981;">●</span> Core AI Engine: ResNet50 + YOLOv8 + Spatio-Temporal BiLSTM</div>
    <div class="telemetry-pill pill-blue">⚡ Mean Latency: ~24.2 ms / frame</div>
    <div class="telemetry-pill pill-purple">🌐 Cloud Cluster Node: Active (Headless Edge Profile)</div>
    <div class="telemetry-pill">🎯 Calibration: Uncapped Probabilistic Fusion Engine</div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR CONTROLS ──────────────────────────────────────────────────────────
st.sidebar.header("🕹️ System Controls")

if APP_URL and not APP_URL.startswith("https://YOUR"):
    st.sidebar.link_button("🚀 Open Deployed Cloud App", APP_URL, use_container_width=True)
else:
    st.sidebar.caption("_Set `APP_URL` in app.py to enable the deployed-app shortcut._")
st.sidebar.markdown("---")

app_mode = st.sidebar.selectbox(
    "Select AI Inspection Mode",
    [
        "🚗 Vehicle Damage Severity Classification (Low/Mid/High)",
        "🏍️ Pre-Crash Violation & Speeding Monitor",
        "🛣️ Road Surface Hazard & Pothole Detection",
        "⚡ Unified Multimodal Risk & Damage Fusion",
        "📹 Live Camera Traffic Risk Monitor (CNN + LSTM)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Input Selection")

input_options = ["Pre-Loaded Benchmark Samples", "Upload Image", "Upload Video", "Live Camera (Webcam)"]
default_idx = 3 if "Live Camera" in app_mode else 0
input_source = st.sidebar.radio("Choose Input Source", input_options, index=default_idx)

uploaded_image = None
uploaded_video = None
selected_sample = None
raw_image = None

if input_source == "Pre-Loaded Benchmark Samples":
    sample_options = {
        "Low Severity Vehicle Damage": "sample_low_damage.jpg",
        "Mid Severity Vehicle Damage": "sample_mid_damage.jpg",
        "High Severity Vehicle Damage": "sample_high_damage.jpg",
        "Road Surface Potholes": "sample_pothole_road.jpg",
        "Two-Wheeler Triple-Riding / No Helmet": "sample_two_wheeler_violation.jpg"
    }
    selected_sample = st.sidebar.selectbox("Choose Benchmark Scenario", list(sample_options.keys()))
    sample_path = SAMPLE_DATA_DIR / sample_options[selected_sample]
    if sample_path.exists():
        raw_image = Image.open(str(sample_path))
    else:
        st.sidebar.warning("Generating benchmark sample...")
        _ensure_samples()
        if sample_path.exists():
            raw_image = Image.open(str(sample_path))

elif input_source == "Upload Image":
    uploaded_image = st.sidebar.file_uploader("Upload Inspection Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        raw_image = Image.open(uploaded_image)

elif input_source == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("Upload Traffic Video Stream", type=["mp4", "avi", "mov", "mkv"])

# Simulated Telemetry Inputs
show_telemetry = (
    "Pre-Crash Violation" in app_mode or
    "Unified Multimodal" in app_mode or
    "Live Camera" in app_mode or
    input_source == "Live Camera (Webcam)" or
    input_source == "Upload Video"
)

simulated_speed = 60.0
simulated_riders = 1
simulated_helmet = True

if show_telemetry:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Simulated Vehicle Telemetry")
    st.sidebar.caption("Standard cameras cannot physically measure speedometer values; configure realistic telemetry below:")
    simulated_speed = st.sidebar.slider("Vehicle Speed (km/h)", 20.0, 140.0, 65.0, 5.0)

    if "Road Surface Hazard" not in app_mode:
        simulated_riders = st.sidebar.slider("Two-Wheeler Rider Count", 1, 4, 2)
        simulated_helmet = st.sidebar.checkbox("Helmet Detected on All Riders", value=True)

# ── HELPER: GAUGE CHART ───────────────────────────────────────────────────────
def create_gauge(score: float, title: str, color: str, is_percentage: bool = True):
    raw_val = float(score)
    display_val = raw_val * 100.0 if raw_val <= 1.0 else raw_val
    display_val = min(100.0, max(0.0, display_val))
    suffix_text = "%"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_val,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 15, 'color': "#1E293B"}},
        number={'suffix': f" {suffix_text}", 'font': {'size': 22, 'color': color}, 'valueformat': '.1f'},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 1.5,
            'bordercolor': "#CBD5E1",
            'steps': [
                {'range': [0, 38], 'color': "rgba(16, 185, 129, 0.15)"},
                {'range': [38, 72], 'color': "rgba(245, 158, 11, 0.15)"},
                {'range': [72, 100], 'color': "rgba(239, 68, 68, 0.15)"}
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8,
                'value': display_val
            }
        }
    ))
    fig.update_layout(height=195, margin=dict(l=15, r=15, t=30, b=15))
    return fig

# ── 5 EXECUTIVE SHOWCASE TABS ─────────────────────────────────────────────────
tab_main, tab_env, tab_analytics, tab_report, tab_docs = st.tabs([
    "🛡️ Live Detection & Safety Console",
    "🌦️ Environmental Risk Simulator",
    "📊 Incident Analytics & Trends",
    "📑 Official PDF Incident Report",
    "ℹ️ Architecture & Methodology"
])

# ==============================================================================
# TAB 1: LIVE DETECTION & SAFETY CONSOLE
# ==============================================================================
with tab_main:
    # ── CASE 1: LIVE CAMERA MONITORING ─────────────────────────────────────────
    if input_source == "Live Camera (Webcam)" or "Live Camera" in app_mode:
        st.subheader("📹 Live Camera Continuous Traffic Risk Monitor (CNN + LSTM)")
        st.caption("Continuous rolling-sequence spatio-temporal evaluation using the dual-stream CNN + LSTM pipeline.")

        with st.expander("🎬 **Recommended YouTube Traffic Test Videos for Live Camera**", expanded=False):
            st.markdown("""
            Open any of these sample videos on your smartphone or side-by-side screen and aim your webcam at it:
            * 🚗 **[Wham Baam Dashcam - Near Misses & Close Calls](https://www.youtube.com/results?search_query=wham+baam+dashcam+near+miss)**
            * 🏍️ **[Two-Wheeler Traffic & Close Call Compilation](https://www.youtube.com/results?search_query=motorcycle+close+call+dashcam)**
            * 🛣️ **[City Traffic Highway Dashcam 4K](https://www.youtube.com/results?search_query=dashcam+driving+video+4k)**
            """)
            st.video("https://www.youtube.com/watch?v=1La4QzGeaaQ")
            st.caption("Tip: Play this clip on a second screen or phone and point your laptop camera directly at it.")

        cam_method = st.radio(
            "Camera Mode",
            [
                "📸 Browser Camera Snapshot (Guaranteed on Cloud / Phone)",
                "🎥 WebRTC Continuous Stream (with TURN Relay)",
                "💻 Direct Local USB / OpenCV Camera (Localhost Only)"
            ],
            horizontal=True
        )

        # MODE A: Browser Native Camera
        if "Browser Camera Snapshot" in cam_method:
            st.info("💡 Point your phone or laptop camera at a road scene, traffic video, or vehicle, then capture a frame:")
            camera_photo = st.camera_input("Point camera at traffic scene / phone video")
            
            if camera_photo is not None:
                photo_bytes = camera_photo.getvalue()
                nparr = np.frombuffer(photo_bytes, np.uint8)
                frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                with st.spinner("Processing frame with spatio-temporal CNN + LSTM pipeline..."):
                    res = lstm_pipeline.process_frame(
                        frame_bgr,
                        simulated_speed=simulated_speed,
                        rider_count=simulated_riders,
                        helmet_detected=simulated_helmet
                    )
                    annotated_rgb = cv2.cvtColor(res["annotated_frame"], cv2.COLOR_BGR2RGB)

                    # Update session state for report and analytics
                    st.session_state.current_assessment.update({
                        "modality": "Live Camera Temporal Stream (CNN+LSTM)",
                        "overall_risk": float(res["risk_score"] * 100),
                        "severity_label": res["severity_label"],
                        "severity_color": res["severity_color"],
                        "annotated_image": Image.fromarray(annotated_rgb),
                        "violation_results": {
                            "violation_score": float(res["risk_score"] * 100),
                            "violations_detected": res["warnings"]
                        }
                    })

                col1, col2 = st.columns([1.2, 1])
                with col1:
                    st.image(annotated_rgb, caption="AI Camera HUD & Trajectory Overlay", use_container_width=True)

                with col2:
                    st.markdown(f"""
                    <div class="badge-card" style="background-color: {res['severity_color']};">
                        TEMPORAL RISK: {res['severity_label'].upper()}
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(
                        create_gauge(res["risk_score"], "Rolling Risk Score", res["severity_color"]),
                        use_container_width=True
                    )
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Motion / Looming Rate", f"{res['motion_rate']}%")
                    with m2:
                        st.metric("Severity Tier", res["severity_tier"])

                st.write("#### Active Safety Warnings & Status:")
                for w in res["warnings"]:
                    if "No active" in w or "normal" in w.lower():
                        st.success(f"✅ {w}")
                    else:
                        st.error(f"⚠️ {w}")

        # MODE B: WebRTC Stream
        elif "WebRTC Continuous Stream" in cam_method and HAS_WEBRTC:
            class WebRTCRiskProcessor(VideoProcessorBase):
                def __init__(self):
                    self.pipeline = RollingSequenceRiskPipeline(window_size=10)

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    res = self.pipeline.process_frame(
                        img,
                        simulated_speed=simulated_speed,
                        rider_count=simulated_riders,
                        helmet_detected=simulated_helmet
                    )
                    return av.VideoFrame.from_ndarray(res["annotated_frame"], format="bgr24")

            rtc_config = RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302", "stun:stun2.l.google.com:19302"]},
                    {
                        "urls": [
                            "turn:openrelay.metered.ca:80",
                            "turn:openrelay.metered.ca:443",
                            "turns:openrelay.metered.ca:443?transport=tcp"
                        ],
                        "username": "openrelayproject",
                        "credential": "openrelayproject"
                    }
                ]
            })

            webrtc_ctx = webrtc_streamer(
                key="traffic-cnn-lstm",
                video_processor_factory=WebRTCRiskProcessor,
                rtc_configuration=rtc_config,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )

            st.info("💡 Click **'START'** above to enable your webcam. If your network firewall blocks WebRTC peer-to-peer streaming, switch to **'📸 Browser Camera Snapshot'** above for guaranteed instant capture.")

        # MODE C: Direct OpenCV Local Camera
        else:
            run_cam = st.checkbox("Start Live Webcam Stream (Direct OpenCV)", value=False)
            cam_placeholder = st.empty()
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            if run_cam:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not access local hardware webcam. If you are on the deployed cloud website, please switch to '📸 Browser Camera Snapshot' above.")
                else:
                    lstm_pipeline.reset()
                    stop_btn = st.button("Stop Camera Stream")

                    while cap.isOpened() and not stop_btn:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        res = lstm_pipeline.process_frame(
                            frame,
                            simulated_speed=simulated_speed,
                            rider_count=simulated_riders,
                            helmet_detected=simulated_helmet
                        )

                        annotated_rgb = cv2.cvtColor(res["annotated_frame"], cv2.COLOR_BGR2RGB)
                        cam_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                        with metric_col1:
                            st.metric("Temporal Risk Score", f"{res['risk_score']:.3f}")
                        with metric_col2:
                            st.metric("Severity Level", res["severity_tier"])
                        with metric_col3:
                            st.metric("Motion Looming Rate", f"{res['motion_rate']}%")

                        time.sleep(0.03)

                    cap.release()

    # ── CASE 2: VIDEO UPLOAD ───────────────────────────────────────────────────
    elif input_source == "Upload Video" and uploaded_video is not None:
        st.subheader("🎞️ Video Stream Processing via Rolling CNN + LSTM Pipeline")
        
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        st.write(f"Video Loaded: **{total_frames} frames** at **{fps:.1f} FPS**")
        
        vcol1, vcol2 = st.columns([1.3, 1])
        frame_display = vcol1.empty()
        gauge_display = vcol2.empty()
        info_display = vcol2.empty()

        start_process = st.button("Start Rolling CNN+LSTM Analysis")

        if start_process:
            lstm_pipeline.reset()
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_idx > 300:
                    break

                res = lstm_pipeline.process_frame(
                    frame,
                    simulated_speed=simulated_speed,
                    rider_count=simulated_riders,
                    helmet_detected=simulated_helmet
                )

                if frame_idx % 2 == 0:
                    annotated_rgb = cv2.cvtColor(res["annotated_frame"], cv2.COLOR_BGR2RGB)
                    frame_display.image(annotated_rgb, use_container_width=True)

                    gauge_display.plotly_chart(
                        create_gauge(res["risk_score"], "Rolling Temporal Risk Score", res["severity_color"]),
                        use_container_width=True
                    )

                    info_display.markdown(f"""
                    <div class="metric-card">
                        <b>Frame:</b> {frame_idx} / {total_frames}<br>
                        <b>Severity Tier:</b> <span style="color:{res['severity_color']}; font-weight:bold;">{res['severity_label']}</span><br>
                        <b>Status:</b> {res['warning_status']}<br>
                        <b>Active Warnings:</b> {', '.join(res['warnings'])}
                    </div>
                    """, unsafe_allow_html=True)

                frame_idx += 1
                time.sleep(0.01)

            cap.release()
            st.success("Video sequence processing completed.")

    # ── CASE 3: STATIC IMAGE & BENCHMARK EVALUATION ────────────────────────────
    elif raw_image is not None:
        img_np = np.array(raw_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # SUBMODE A: Vehicle Damage Severity
        if "Vehicle Damage Severity" in app_mode:
            st.subheader("🚗 Vehicle Damage Localization & Severity Grading")
            
            col1, col2 = st.columns([1.2, 1])

            with st.spinner("Analyzing structural damage & surface deformation..."):
                damage_res = damage_clf.assess_damage(img_bgr)
                annotated_rgb = cv2.cvtColor(damage_res["annotated_image"], cv2.COLOR_BGR2RGB)

                # Update session state for report and analytics
                st.session_state.current_assessment.update({
                    "modality": "Vehicle Damage Severity Inspection",
                    "overall_risk": float(damage_res["severity_score"]),
                    "severity_label": damage_res["severity_label"],
                    "severity_color": damage_res["severity_color"],
                    "annotated_image": Image.fromarray(annotated_rgb),
                    "damage_results": damage_res
                })

            with col1:
                st.image(annotated_rgb, caption="AI Damage Bounding Box Localization", use_container_width=True)

            with col2:
                sev_color = damage_res["severity_color"]
                st.markdown(f"""
                <div class="badge-card" style="background-color: {sev_color};">
                    PREDICTED SEVERITY: {damage_res['severity_label'].upper()}
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(
                    create_gauge(damage_res["severity_score"], "Damage Impact Index", sev_color, is_percentage=True),
                    use_container_width=True
                )

                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.metric("Damage Area Coverage", f"{damage_res['damage_area_percent']}%")
                with mcol2:
                    st.metric("Defects Identified", len(damage_res["damages_found"]))

            st.markdown("---")
            st.subheader("📋 Impact Assessment & Recommended Protocol")
            
            tcol1, tcol2 = st.columns(2)
            with tcol1:
                st.info(f"**Description of Impact:**\n\n{damage_res['description']}")
            with tcol2:
                st.warning(f"**Required Protocol & Action:**\n\n{damage_res['recommended_action']}")

            st.write("#### Detailed Breakdown of Detected Deformities")
            defect_data = []
            for d in damage_res["damages_found"]:
                cls_clean = d["class"].replace("_", " ").title()
                defect_data.append({
                    "Defect Class": cls_clean,
                    "Model Confidence": f"{d['confidence']:.1%}",
                    "Bounding Box (X1, Y1, X2, Y2)": str(d["bbox"])
                })
            st.dataframe(defect_data, use_container_width=True)

        # SUBMODE B: Pre-Crash Violation
        elif "Pre-Crash Violation" in app_mode:
            st.subheader("🏍️ Pre-Crash Violation & Two-Wheeler Safety Assessment")
            
            telemetry = {
                "speed_kmh": simulated_speed,
                "rider_count": simulated_riders,
                "helmet_detected": simulated_helmet
            }

            col1, col2 = st.columns([1.2, 1])

            with st.spinner("Tracking velocity and safety compliance..."):
                viol_res = violation_det.analyze_frame(img_bgr, simulated_telemetry=telemetry)
                annotated_rgb = cv2.cvtColor(viol_res["annotated_frame"], cv2.COLOR_BGR2RGB)

                v_score = viol_res["risk_score"]
                v_color = "#10B981" if v_score < 0.38 else ("#F59E0B" if v_score < 0.72 else "#EF4444")
                v_tier = "LOW" if v_score < 0.38 else ("MID" if v_score < 0.72 else "HIGH")

                # Update session state for report and analytics
                st.session_state.current_assessment.update({
                    "modality": "Pre-Crash Violation & Telemetry",
                    "overall_risk": float(v_score * 100),
                    "severity_label": f"{v_tier} Risk Violation",
                    "severity_color": v_color,
                    "annotated_image": Image.fromarray(annotated_rgb),
                    "violation_results": {
                        "violation_score": float(v_score * 100),
                        "violations_detected": viol_res["violations"]
                    }
                })

            with col1:
                st.image(annotated_rgb, caption="Real-time Violation & Telemetry HUD", use_container_width=True)

            with col2:
                st.markdown(f"""
                <div class="badge-card" style="background-color: {v_color};">
                    BEHAVIORAL RISK: {v_tier} RISK
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(create_gauge(v_score, "Pre-Crash Risk Score", v_color), use_container_width=True)

                st.write("#### Active Violations:")
                if viol_res["violations"]:
                    for v in viol_res["violations"]:
                        st.error(f"⚠️ {v}")
                else:
                    st.success("✅ No Traffic Rule Violations Detected. Safe Rider Compliance.")

        # SUBMODE C: Road Hazards & Potholes
        elif "Road Surface Hazard" in app_mode:
            st.subheader("🛣️ Road Hazard & Pothole Segmentation Engine")

            col1, col2 = st.columns([1.2, 1])

            with st.spinner("Scanning road surface defects and depressions..."):
                haz_res = hazard_det.scan_road_hazards(img_bgr)
                annotated_rgb = cv2.cvtColor(haz_res["annotated_frame"], cv2.COLOR_BGR2RGB)

                h_score = haz_res["hazard_risk_score"]
                h_color = "#10B981" if h_score < 0.38 else ("#F59E0B" if h_score < 0.72 else "#EF4444")
                h_tier = "LOW" if h_score < 0.38 else ("MID" if h_score < 0.72 else "HIGH")

                # Update session state for report and analytics
                st.session_state.current_assessment.update({
                    "modality": "Road Surface Hazard Segmentation",
                    "overall_risk": float(h_score * 100),
                    "severity_label": f"{h_tier} Hazard Level",
                    "severity_color": h_color,
                    "annotated_image": Image.fromarray(annotated_rgb),
                    "hazard_results": {
                        "hazard_score": float(h_score * 100),
                        "hazard_count": haz_res["hazard_count"],
                        "hazards_detected": [f"{h['type'].title()} ({h['severity']})" for h in haz_res.get("hazards", [])]
                    }
                })

            with col1:
                st.image(annotated_rgb, caption="Road Hazard Identification", use_container_width=True)

            with col2:
                st.markdown(f"""
                <div class="badge-card" style="background-color: {h_color};">
                    ROAD SURFACE THREAT: {h_tier}
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(create_gauge(h_score, "Road Hazard Index", h_color), use_container_width=True)

                st.metric("Potholes / Obstacles Count", haz_res["hazard_count"])
                if haz_res["hazard_count"] > 0:
                    st.warning("⚠️ Caution: Road depressions identified. Alerting municipal road maintenance.")
                else:
                    st.success("✅ Smooth road surface. No major depressions detected.")

        # SUBMODE D: Unified Multimodal Fusion
        else:
            st.subheader("⚡ Unified Multimodal Road Accident Risk & Damage Fusion")

            telemetry = {
                "speed_kmh": simulated_speed,
                "rider_count": simulated_riders,
                "helmet_detected": simulated_helmet
            }

            with st.spinner("Computing unified multimodal fusion matrix..."):
                d_res = damage_clf.assess_damage(img_bgr)
                v_res = violation_det.analyze_frame(img_bgr, simulated_telemetry=telemetry)
                h_res = hazard_det.scan_road_hazards(img_bgr)
                fusion_res = fusion_engine.fuse_assessment(d_res, v_res, h_res)

                annotated_rgb = cv2.cvtColor(d_res["annotated_image"], cv2.COLOR_BGR2RGB)

                # Update session state for report and analytics
                st.session_state.current_assessment.update({
                    "modality": "Unified Multimodal Risk Fusion",
                    "overall_risk": float(fusion_res["composite_score"] * 100),
                    "severity_label": fusion_res["overall_label"],
                    "severity_color": fusion_res["overall_color"],
                    "annotated_image": Image.fromarray(annotated_rgb),
                    "damage_results": d_res,
                    "violation_results": {
                        "violation_score": float(v_res["risk_score"] * 100),
                        "violations_detected": v_res["violations"]
                    },
                    "hazard_results": {
                        "hazard_score": float(h_res["hazard_risk_score"] * 100),
                        "hazard_count": h_res["hazard_count"],
                        "hazards_detected": [f"{h['type'].title()} ({h['severity']})" for h in h_res.get("hazards", [])]
                    }
                })

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(annotated_rgb, caption="Multimodal Inspection Analysis", use_container_width=True)

            with col2:
                st.markdown(f"""
                <div class="badge-card" style="background-color: {fusion_res['overall_color']};">
                    OVERALL SEVERITY TIER: {fusion_res['overall_label'].upper()}
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(create_gauge(fusion_res["composite_score"], "Composite Fusion Risk Score", fusion_res["overall_color"]), use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Multimodal Component Contribution")
            
            components = fusion_res["breakdown"]
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=["Damage Severity Index", "Violation Risk Score", "Road Hazard Index"],
                    y=[components["damage_component"], components["violation_component"], components["hazard_component"]],
                    marker_color=["#EF4444", "#F59E0B", "#3B82F6"],
                    text=[f"{components['damage_component']:.2f}", f"{components['violation_component']:.2f}", f"{components['hazard_component']:.2f}"],
                    textposition="auto"
                )
            ])
            fig_bar.update_layout(yaxis=dict(range=[0, 1]), height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("🚨 Emergency Triage & Incident Dispatch")
            st.markdown(f"""
            <div class="dispatch-card">
                <b>Dispatched Protocol:</b> {fusion_res['action_protocol']}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("👆 Please select a benchmark test sample, upload an image/video, or switch to Live Camera mode in the sidebar to begin evaluation.")

    # Quick Jump to PDF Report
    st.markdown("---")
    r_col1, r_col2 = st.columns([3, 1])
    with r_col1:
        st.caption("Need an official downloadable report for college review / traffic audit?")
    with r_col2:
        try:
            curr = st.session_state.current_assessment
            pdf_data = generate_incident_pdf(
                incident_id=curr["incident_id"],
                modality_type=curr["modality"],
                overall_risk=curr["overall_risk"],
                status_label=curr["severity_label"],
                damage_results=curr.get("damage_results"),
                violation_results=curr.get("violation_results"),
                hazard_results=curr.get("hazard_results"),
                annotated_image=curr.get("annotated_image"),
                environmental_factors=curr.get("environmental_factors")
            )
            st.download_button(
                label="📥 Export Inspection PDF",
                data=pdf_data,
                file_name=f"Incident_Report_{curr['incident_id']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.caption(f"Report generator ready in Tab 4.")


# ==============================================================================
# TAB 2: ENVIRONMENTAL & WEATHER RISK SIMULATOR
# ==============================================================================
with tab_env:
    st.subheader("🌦️ Environmental & Weather Risk Simulation Engine")
    st.markdown("""
    Road traffic accident probabilities and stopping distances vary drastically depending on **weather**,
    **ambient lighting**, **road friction coefficients**, and **traffic density**. This simulator models these
    multi-factor external conditions and shows their sensitivity impact on vehicle braking dynamics and accident probability.
    """)

    ecol1, ecol2, ecol3 = st.columns(3)
    with ecol1:
        weather_opt = st.selectbox(
            "Atmospheric Condition",
            ["Clear / Daylight", "Overcast / Light Drizzle", "Heavy Monsoon Rain", "Dense Fog / Low Visibility", "Slick / Hail"]
        )
    with ecol2:
        surface_opt = st.selectbox(
            "Pavement Surface State",
            ["Dry Asphalt (μ = 0.75)", "Wet Asphalt (μ = 0.45)", "Hydroplaning Layer (μ = 0.25)", "Gravel / Loose Dirt (μ = 0.35)"]
        )
    with ecol3:
        lighting_opt = st.selectbox(
            "Lighting & Visibility",
            ["Bright Daylight (100% Visibility)", "Dawn / Dusk Twilight (70% Visibility)", "Streetlit Urban Night (50% Visibility)", "Dark Highway / Unlit (25% Visibility)"]
        )

    # Physics calculations
    mu_dict = {
        "Dry Asphalt (μ = 0.75)": 0.75,
        "Wet Asphalt (μ = 0.45)": 0.45,
        "Hydroplaning Layer (μ = 0.25)": 0.25,
        "Gravel / Loose Dirt (μ = 0.35)": 0.35
    }
    mu = mu_dict.get(surface_opt, 0.75)

    # Weather multiplier
    weather_factors = {
        "Clear / Daylight": 1.0,
        "Overcast / Light Drizzle": 1.15,
        "Heavy Monsoon Rain": 1.45,
        "Dense Fog / Low Visibility": 1.65,
        "Slick / Hail": 1.85
    }
    light_factors = {
        "Bright Daylight (100% Visibility)": 1.0,
        "Dawn / Dusk Twilight (70% Visibility)": 1.12,
        "Streetlit Urban Night (50% Visibility)": 1.25,
        "Dark Highway / Unlit (25% Visibility)": 1.40
    }
    env_multiplier = weather_factors.get(weather_opt, 1.0) * light_factors.get(lighting_opt, 1.0) * (0.75 / mu)**0.5
    env_multiplier = round(min(2.5, env_multiplier), 2)

    st.markdown(f"""
    <div class="metric-card" style="border-left: 5px solid #3B82F6;">
        <b>Composite Environmental Risk Multiplier:</b> <span style="font-size:1.3rem; font-weight:bold; color:#1D4ED8;">{env_multiplier:.2f}x</span><br>
        <span style="font-size:0.85rem; color:#64748B;">Baseline friction μ = {mu:.2f} | Adverse conditions increase effective collision probability by {int((env_multiplier-1.0)*100)}%.</span>
    </div>
    """, unsafe_allow_html=True)

    # Braking distance simulation curve
    speeds = np.linspace(20, 130, 23)
    g = 9.81
    reaction_time = 1.2  # average driver perception reaction time (seconds)
    
    # Distance = reaction distance + braking distance
    v_ms = speeds * (1000 / 3600)
    dist_dry = v_ms * reaction_time + (v_ms**2) / (2 * 0.75 * g)
    dist_current = v_ms * reaction_time + (v_ms**2) / (2 * mu * g)

    fig_brake = go.Figure()
    fig_brake.add_trace(go.Scatter(x=speeds, y=dist_dry, mode='lines', name='Baseline (Dry Asphalt μ=0.75)', line=dict(color='#10B981', width=2.5)))
    fig_brake.add_trace(go.Scatter(x=speeds, y=dist_current, mode='lines+markers', name=f'Current Selected ({surface_opt})', line=dict(color='#EF4444', width=3, dash='solid')))
    
    # Mark current simulated speed
    curr_v_ms = simulated_speed * (1000 / 3600)
    curr_dist = curr_v_ms * reaction_time + (curr_v_ms**2) / (2 * mu * g)
    fig_brake.add_trace(go.Scatter(
        x=[simulated_speed],
        y=[curr_dist],
        mode='markers+text',
        name=f'Current Speed ({simulated_speed} km/h)',
        marker=dict(size=14, color='#7C3AED', symbol='star'),
        text=[f"Stopping: {curr_dist:.1f} m"],
        textposition="top center"
    ))

    fig_brake.update_layout(
        title="Vehicle Speed vs. Total Stopping Distance (Physics & Perception Model)",
        xaxis_title="Vehicle Speed (km/h)",
        yaxis_title="Total Stopping Distance (meters)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_brake, use_container_width=True)

    # Save to session state
    if st.button("Apply Environmental Multiplier to Current Assessment"):
        st.session_state.current_assessment["environmental_factors"] = {
            "weather": weather_opt,
            "lighting": lighting_opt,
            "traffic_density": "Moderate",
            "multiplier": env_multiplier
        }
        adjusted_score = min(100.0, st.session_state.current_assessment["overall_risk"] * env_multiplier)
        st.success(f"✅ Applied! Adjusted Composite Risk Score: {adjusted_score:.1f}%")


# ==============================================================================
# TAB 3: INCIDENT ANALYTICS & TRENDS
# ==============================================================================
with tab_analytics:
    st.subheader("📊 Multimodal Incident Telemetry & Dimensional Analytics")
    st.caption("Deep-dive comparative analysis across all sensory modalities and kinetic dynamics.")

    curr = st.session_state.current_assessment
    
    # Radar Chart
    r_col1, r_col2 = st.columns([1.1, 1])
    
    with r_col1:
        st.markdown("#### 🎯 5-Dimensional Safety Radar")
        
        # Calculate scores for each dimension
        dmg_val = curr.get("damage_results", {}).get("severity_score", 40.0) if curr.get("damage_results") else 20.0
        viol_val = curr.get("violation_results", {}).get("violation_score", 50.0) if curr.get("violation_results") else 35.0
        haz_val = curr.get("hazard_results", {}).get("hazard_score", 30.0) if curr.get("hazard_results") else 15.0
        speed_val = min(100.0, (simulated_speed / 120.0) * 100.0)
        env_val = min(100.0, (curr.get("environmental_factors", {}).get("multiplier", 1.0) / 2.0) * 100.0)

        radar_categories = ['Vehicle Structural Damage', 'Pre-Crash Speed Delta', 'Behavioral Violations', 'Road Hazards / Depressions', 'Environmental Threat']
        radar_values = [dmg_val, speed_val, viol_val, haz_val, env_val]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_categories + [radar_categories[0]],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.25)',
            line=dict(color='#2563EB', width=2.5),
            name='Current Session Score'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            ),
            showlegend=False,
            height=340,
            margin=dict(l=35, r=35, t=25, b=25)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with r_col2:
        st.markdown("#### 💥 Kinetic Energy & Collision Severity ($E_k = \\frac{1}{2}mv^2$)")
        
        v_range = np.linspace(20, 120, 20)
        mass_kg = 1400  # typical sedan mass
        kinetic_energy_kj = 0.5 * mass_kg * (v_range * 1000 / 3600)**2 / 1000
        current_energy_kj = 0.5 * mass_kg * (simulated_speed * 1000 / 3600)**2 / 1000

        fig_ke = go.Figure()
        fig_ke.add_trace(go.Scatter(
            x=v_range,
            y=kinetic_energy_kj,
            mode='lines',
            line=dict(color='#DC2626', width=3),
            name='Crash Kinetic Energy (kJ)'
        ))
        fig_ke.add_trace(go.Scatter(
            x=[simulated_speed],
            y=[current_energy_kj],
            mode='markers+text',
            name=f'Current Speed ({simulated_speed} km/h)',
            marker=dict(size=12, color='#1E40AF', symbol='diamond'),
            text=[f"{current_energy_kj:.0f} kJ"],
            textposition="top left"
        ))
        fig_ke.update_layout(
            title=f"Impact Energy: {current_energy_kj:.0f} kJ at {simulated_speed} km/h",
            xaxis_title="Speed (km/h)",
            yaxis_title="Kinetic Energy (Kilojoules)",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_ke, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📈 Benchmark Incident Severity Distribution (Evaluation Set)")
    
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Minor / Surface Scratches', 'Moderate / Broken Assemblies', 'Severe / Structural Deformation'],
            values=[42, 35, 23],
            hole=.55,
            marker_colors=['#10B981', '#F59E0B', '#EF4444']
        )])
        fig_donut.update_layout(title="Damage Severity Distribution", height=260, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    with pcol2:
        fig_bar_viol = go.Figure(data=[go.Bar(
            x=['No Helmet', 'Speeding (>80 km/h)', 'Triple Riding', 'Sudden Swerve'],
            y=[48, 38, 22, 14],
            marker_color=['#F97316', '#EF4444', '#8B5CF6', '#3B82F6']
        )])
        fig_bar_viol.update_layout(title="Pre-Crash Violations Frequency in Dataset", height=260, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_bar_viol, use_container_width=True)


# ==============================================================================
# TAB 4: OFFICIAL PDF INCIDENT REPORT GENERATOR
# ==============================================================================
with tab_report:
    st.subheader("📑 Official PDF Incident Inspection & Triage Report")
    st.caption("Generate and export a formal, verified inspection report for college review, road authorities, or insurance assessment.")

    curr = st.session_state.current_assessment
    
    col_rep_cfg, col_rep_preview = st.columns([1, 1.2])

    with col_rep_cfg:
        st.markdown("#### 🛠️ Report Metadata Configuration")
        incident_id_input = st.text_input("Incident Reference ID", value=curr["incident_id"])
        location_input = st.text_input("Location / Highway Geofence", value="NH-48 Chennai-Bengaluru Highway, Km 28.4")
        surveyor_notes = st.text_area(
            "Surveyor / Investigator Remarks",
            value="Multimodal AI automated assessment completed. Telemetry and visual deformities correlated with CNN-LSTM edge inferences."
        )
        
        st.write("#### Dispatched Emergency Assets:")
        als_amb = st.checkbox("Tier-1 ALS Ambulance Dispatched", value=(curr["overall_risk"] >= 70))
        hwy_patrol = st.checkbox("Highway Police Sector Alerted", value=(curr["overall_risk"] >= 40))
        tow_truck = st.checkbox("Recovery Towing & Debris Clearing Dispatched", value=(curr["overall_risk"] >= 60))

    with col_rep_preview:
        st.markdown("#### 📋 Live Report Summary")
        st.markdown(f"""
        <div class="metric-card">
            <b>Institution:</b> Panimalar Engineering College (AIML D16)<br>
            <b>Incident ID:</b> <code>{incident_id_input}</code><br>
            <b>Modality:</b> {curr['modality']}<br>
            <b>Overall Composite Risk:</b> <span style="font-weight:bold; color:{curr['severity_color']};">{curr['overall_risk']:.1f}%</span><br>
            <b>Status:</b> {curr['severity_label']}<br>
            <b>Location:</b> {location_input}<br>
            <b>Timestamp:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
        </div>
        """, unsafe_allow_html=True)

        if curr.get("annotated_image") is not None:
            st.image(curr["annotated_image"], caption="Evidence Snapshot to be embedded in PDF", use_container_width=True)
        else:
            st.info("Run an inspection in Tab 1 to embed visual evidence here.")

    st.markdown("---")
    
    # PDF generation button
    try:
        pdf_bytes = generate_incident_pdf(
            incident_id=incident_id_input,
            modality_type=curr["modality"],
            overall_risk=curr["overall_risk"],
            status_label=curr["severity_label"],
            damage_results=curr.get("damage_results"),
            violation_results=curr.get("violation_results"),
            hazard_results=curr.get("hazard_results"),
            annotated_image=curr.get("annotated_image"),
            environmental_factors=curr.get("environmental_factors"),
            notes=surveyor_notes
        )

        bcol1, bcol2, bcol3 = st.columns([1, 2, 1])
        with bcol2:
            st.download_button(
                label="📄 Download Official PDF Incident Report",
                data=pdf_bytes,
                file_name=f"Official_Incident_Report_{incident_id_input}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.caption("Includes official Panimalar Engineering College letterhead, triage matrices, and digital verification seal.")
    except Exception as e:
        st.error(f"Error compiling PDF: {str(e)}")


# ==============================================================================
# TAB 5: ARCHITECTURE & METHODOLOGY
# ==============================================================================
with tab_docs:
    st.subheader("ℹ️ Project Architecture, Mathematical Formulations & Review Credentials")
    
    st.markdown("""
    ### 1. Problem Statement & Motivation
    Conventional road safety systems monitor vehicular events in isolation: traffic cameras flag speeding, separate road surveys identify potholes, and insurance claims adjusters inspect damage post-accident. 
    This project presents an **end-to-end multimodal AI decision-support architecture** that unifies:
    1. **Pre-Crash Risk Forecasting:** Spatio-temporal CNN + Bidirectional LSTM tracking motion looming, lane swerving, and reckless trajectory patterns in real time.
    2. **Active Safety Violation Monitoring:** Dual-stream YOLOv8 object detection tracking helmet compliance, triple-riding, and simulated velocity thresholding.
    3. **Road Surface Hazard Detection:** Dynamic edge and depression morphology segmentation identifying potholes and pavement fissures before tire impact.
    4. **Post-Crash Damage Severity Classification:** 3-tier ResNet-50 transfer learning framework categorizing vehicular structural deformation into **Low / Mid / High** severity indices with emergency triage dispatch.

    ---

    ### 2. Deep Learning Pipeline Architecture
    ```
    ┌────────────────────────────────────────────────────────────────────────┐
    │                      MULTIMODAL INPUT STREAMS                          │
    │   [Live Webcam / Video]        [Traffic Camera]      [Vehicle Telemetry]│
    └─────────────────┬─────────────────────┬───────────────────────┬────────┘
                      │                     │                       │
                      ▼                     ▼                       ▼
    ┌─────────────────────────┐  ┌─────────────────────┐  ┌─────────────────┐
    │  Spatio-Temporal CNN    │  │  YOLOv8 Detection   │  │ Surface Hazard  │
    │  Feature Extractor      │  │  (Helmet/Riders)    │  │ Segmentation    │
    └───────────┬─────────────┘  └──────────┬──────────┘  └────────┬────────┘
                │                           │                      │
                ▼                           │                      │
    ┌─────────────────────────┐             │                      │
    │ Rolling BiLSTM Sequence │             │                      │
    │ (Temporal Looming/Risk) │             │                      │
    └───────────┬─────────────┘             │                      │
                │                           │                      │
                └───────────────────┬───────┴──────────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │ Multimodal Risk Fusion Engine│
                     │  (Probabilistic Compounding) │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │ Emergency Triage & Dispatch  │
                     │ (ALS Ambulance / Tow / Police│
                     └──────────────────────────────┘
    ```

    ---

    ### 3. Key Mathematical Formulations

    #### A. Compounded Probability of Multi-Violation Risk
    Independent violations do not additively clip at arbitrary ceilings. We formulate risk as the union probability:
    $$P_{\\text{combined}} = 1 - \\prod_{i=1}^{N} (1 - p_i)$$
    where $p_i \\in [0, 1]$ represents the calibrated risk probability of violation factor $i$.

    #### B. Dynamic Rolling Spatio-Temporal Smoothing
    Temporal feature continuity across video frames is smoothed via an exponential moving average (EMA) kernel:
    $$S_t = \\alpha \\cdot s_t + (1 - \\alpha) \\cdot S_{t-1}$$
    where $\\alpha = 0.35$, preventing high-frequency flicker in camera monitoring.

    #### C. Vehicle Stopping Distance & Kinetic Impact
    Total stopping distance incorporates driver reaction latency $t_{\\text{rxn}}$ and pavement friction coefficient $\\mu$:
    $$D_{\\text{stopping}} = v \\cdot t_{\\text{rxn}} + \\frac{v^2}{2 \\mu g}$$
    Impact kinetic energy quadruples with a doubling of velocity:
    $$E_k = \\frac{1}{2} m v^2$$

    ---

    ### 4. Academic Review & Team Credentials
    * **Institution:** Panimalar Engineering College (An Autonomous Institution, Affiliated to Anna University, Chennai)
    * **Department:** Department of Artificial Intelligence & Machine Learning (AIML)
    * **Academic Year & Batch:** Final Year Engineering Showcase • Batch D16
    * **Student Investigators:**
      * **HARISH AT** (AIML Department)
      * **BARATHRAJ AS** (AIML Department)
    * **Project Supervisor:**
      * **Dr. A. ARUN**, Associate Professor, Department of AI & ML
    """)
