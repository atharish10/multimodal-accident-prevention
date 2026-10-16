"""
Interactive Web Application for Multimodal Road Accident Prevention
and Vehicle Damage Severity Classification (Low / Mid / High).
Features continuous Live Camera (Webcam) monitoring using CNN + LSTM rolling sequence pipeline.
"""

# ── DEPLOYMENT URL ─────────────────────────────────────────────────────────────
# Change this single constant to point to wherever the app is hosted.
# The "Open Deployed App" button in the sidebar will open this URL.
APP_URL = "https://atharish10-multimodal-accident-prevention-app-7u8wx1.streamlit.app/"
# ───────────────────────────────────────────────────────────────────────────────

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import time
import tempfile
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

# ── Ensure benchmark sample images exist (auto-generates if missing) ───────────
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
            from PIL import Image as _Img, ImageDraw as _Draw
            _make_samples(SAMPLE_DATA_DIR, _Img, _Draw)
        except Exception:
            pass  # samples will be missing; app warns inline

def _make_samples(out_dir, _Img, _Draw):
    """Minimal inline sample generator so the app is self-contained on any server."""
    import numpy as np, cv2

    # Low – minor scratch
    img = np.full((500, 750, 3), (240, 240, 240), dtype=np.uint8)
    cv2.rectangle(img, (0, 350), (750, 500), (90, 95, 100), -1)
    pts = np.array([[120,380],[180,260],[380,240],[550,310],[640,380]], np.int32)
    cv2.fillPoly(img, [pts], (180, 110, 50))
    cv2.rectangle(img, (100,360),(660,420),(190,120,60),-1)
    cv2.circle(img,(210,420),45,(30,30,30),-1); cv2.circle(img,(560,420),45,(30,30,30),-1)
    cv2.line(img,(320,380),(390,388),(245,245,245),2)
    cv2.putText(img,"LOW DAMAGE BENCHMARK",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(40,40,40),2)
    cv2.imwrite(str(out_dir/"sample_low_damage.jpg"), img)

    # Mid – broken headlight & bumper
    img = np.full((500, 750, 3), (230, 230, 230), dtype=np.uint8)
    cv2.rectangle(img,(0,350),(750,500),(90,95,100),-1)
    cv2.rectangle(img,(180,230),(570,400),(160,160,160),-1)
    cv2.rectangle(img,(220,160),(530,250),(130,150,160),-1)
    cv2.rectangle(img,(200,280),(280,330),(255,255,220),-1)
    cv2.rectangle(img,(470,280),(550,330),(70,70,70),-1)
    for dx in [10,25,40,60]: cv2.line(img,(470+dx,280),(480+dx,330),(220,220,220),2)
    pts_b = np.array([[170,390],[420,390],[580,450],[560,465],[170,415]], np.int32)
    cv2.fillPoly(img,[pts_b],(90,90,90))
    cv2.putText(img,"MID DAMAGE BENCHMARK",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(20,20,20),2)
    cv2.imwrite(str(out_dir/"sample_mid_damage.jpg"), img)

    # High – structural crush
    img = np.full((500, 750, 3), (220, 220, 220), dtype=np.uint8)
    cv2.rectangle(img,(0,330),(750,500),(80,80,80),-1)
    pts_w = np.array([[120,420],[180,310],[320,260],[430,210],[480,350],[650,430],[300,460]],np.int32)
    cv2.fillPoly(img,[pts_w],(40,40,180))
    cv2.fillPoly(img,[np.array([[220,300],[350,270],[320,350],[190,350]],np.int32)],(200,210,220))
    for i in range(12): cv2.line(img,(260,310),(210+i*15,290+(i%4)*20),(50,50,50),2)
    for k in range(5): cv2.line(img,(350+k*20,260+(k%2)*30),(370+k*20,380),(20,20,100),4)
    cv2.putText(img,"HIGH DAMAGE BENCHMARK",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,180),2)
    cv2.imwrite(str(out_dir/"sample_high_damage.jpg"), img)

    # Pothole
    img = np.full((500, 750, 3), (110, 115, 120), dtype=np.uint8)
    for y_d in [0,200,400]: cv2.rectangle(img,(360,y_d),(390,y_d+120),(240,240,240),-1)
    cv2.ellipse(img,(280,360),(130,65),-15,0,360,(35,35,40),-1)
    cv2.ellipse(img,(280,360),(105,45),-15,0,360,(15,15,20),-1)
    cv2.putText(img,"POTHOLE BENCHMARK",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.imwrite(str(out_dir/"sample_pothole_road.jpg"), img)

    # Two-wheeler violation
    img = np.full((500, 750, 3), (215, 225, 235), dtype=np.uint8)
    cv2.rectangle(img,(0,320),(750,500),(85,90,95),-1)
    cv2.circle(img,(260,420),40,(30,30,30),10); cv2.circle(img,(490,420),40,(30,30,30),10)
    cv2.line(img,(260,420),(370,410),(180,30,30),12); cv2.line(img,(370,410),(490,420),(180,30,30),12)
    cv2.line(img,(370,410),(350,290),(180,30,30),10)
    for hx in [330,380,430]:
        cv2.circle(img,(hx,230),22,(70,120,210),-1); cv2.circle(img,(hx,230),22,(20,20,180),2)
        cv2.rectangle(img,(hx-18,252),(hx+18,330),(60,60,60),-1)
    cv2.putText(img,"TWO-WHEELER VIOLATION BENCHMARK",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.65,(20,20,150),2)
    cv2.imwrite(str(out_dir/"sample_two_wheeler_violation.jpg"), img)

_ensure_samples()

st.set_page_config(
    page_title="Multimodal Road Accident & Damage AI System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 15px;
    }
    .badge-card {
        padding: 14px;
        border-radius: 8px;
        color: white;
        text-align: center;
        font-weight: bold;
        font-size: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Cache and instantiate core models
@st.cache_resource
def load_modules():
    damage_clf = DamageClassifier()
    violation_det = ViolationDetector()
    hazard_det = HazardDetector()
    fusion_engine = MultimodalRiskFusionEngine()
    lstm_pipeline = RollingSequenceRiskPipeline(window_size=10)
    return damage_clf, violation_det, hazard_det, fusion_engine, lstm_pipeline

damage_clf, violation_det, hazard_det, fusion_engine, lstm_pipeline = load_modules()

# Top Header
st.markdown('<div class="main-title">🛡️ Multimodal AI System for Early Detection, Prevention & Damage Severity</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title"><b>Team:</b> HARISH AT & BARATHRAJ AS &nbsp;|&nbsp; <b>Guide:</b> Dr. A. ARUN &nbsp;|&nbsp; Panimalar Engineering College (AIML D16)</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.header("🕹️ System Controls")

# ── Open Deployed App button ───────────────────────────────────────────────────
# Update APP_URL at the top of this file to change where this button points.
if APP_URL and not APP_URL.startswith("https://YOUR"):
    st.sidebar.link_button("🚀 Open Deployed App", APP_URL, use_container_width=True)
else:
    st.sidebar.caption("_Set `APP_URL` in app.py to enable the deployed-app shortcut._")
st.sidebar.markdown("---")
# ───────────────────────────────────────────────────────────────────────────────

app_mode = st.sidebar.selectbox(
    "Select System Mode",
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

# If Live Camera mode is selected, default to Live Camera input source
input_options = ["Pre-Loaded Benchmark Samples", "Upload Image", "Upload Video", "Live Camera (Webcam)"]
default_idx = 3 if "Live Camera" in app_mode else 0

input_source = st.sidebar.radio("Choose Input Source", input_options, index=default_idx)

uploaded_image = None
uploaded_video = None
selected_sample = None
raw_image = None

# Context-sensitive input widgets
if input_source == "Pre-Loaded Benchmark Samples":
    sample_options = {
        "Low Severity Vehicle Damage": "sample_low_damage.jpg",
        "Mid Severity Vehicle Damage": "sample_mid_damage.jpg",
        "High Severity Vehicle Damage": "sample_high_damage.jpg",
        "Road Surface Potholes": "sample_pothole_road.jpg",
        "Two-Wheeler Triple-Riding / No Helmet": "sample_two_wheeler_violation.jpg"
    }
    selected_sample = st.sidebar.selectbox("Choose Benchmark Test Scenario", list(sample_options.keys()))
    sample_path = SAMPLE_DATA_DIR / sample_options[selected_sample]
    if sample_path.exists():
        raw_image = Image.open(str(sample_path))
    else:
        st.warning("Benchmark sample not found. Generating...")

elif input_source == "Upload Image":
    uploaded_image = st.sidebar.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        raw_image = Image.open(uploaded_image)

elif input_source == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("Upload Traffic Video Clip", type=["mp4", "avi", "mov", "mkv"])

# Simplified Sidebar: Only show simulated telemetry when actually relevant
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
    st.sidebar.subheader("📡 Simulated Telemetry")
    st.sidebar.caption("Normal webcams cannot measure physical vehicle speed; speed is configured as a simulated telemetry parameter.")
    simulated_speed = st.sidebar.slider("Simulated Speed (km/h)", 20.0, 140.0, 65.0, 5.0)

    # Rider & helmet only relevant when not strictly pothole inspection
    if "Road Surface Hazard" not in app_mode:
        simulated_riders = st.sidebar.slider("Two-Wheeler Rider Count", 1, 4, 2)
        simulated_helmet = st.sidebar.checkbox("Helmet Detected on All Riders", value=True)

# ----------------- HELPER FUNCTIONS -----------------
def create_gauge(score: float, title: str, color: str, is_percentage: bool = False, max_val: float = 1.0):
    """
    Renders Plotly Gauge accurately representing either a 0.0 - 1.0 Risk Score / Index
    or a true percentage metric without artificial clamping or confusing units.
    """
    display_val = score * 100 if (score <= 1.0 and not is_percentage) else score
    suffix_text = "%" if is_percentage else " / 100"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_val,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': "#1E293B"}},
        number={'suffix': suffix_text, 'font': {'size': 24, 'color': color}, 'valueformat': '.1f'},
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
    fig.update_layout(height=210, margin=dict(l=15, r=15, t=35, b=15))
    return fig


# ----------------- LIVE CAMERA (WEBCAM) MODE -----------------
if input_source == "Live Camera (Webcam)" or "Live Camera" in app_mode:
    st.subheader("📹 Live Camera Continuous Traffic Risk Monitor (CNN + LSTM)")
    st.markdown("""
    Continuous rolling-sequence spatio-temporal evaluation using the **CNN + LSTM pipeline**.
    You can point the webcam at a road scene, moving vehicles, or a video playing on another screen.
    """)

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

    # MODE A: Browser Native Camera (100% reliable on Cloud and mobile, zero firewall/STUN/TURN issues)
    if "Browser Camera Snapshot" in cam_method:
        st.info("💡 Point your phone or laptop camera at a road scene, traffic video, or toy cars, then capture a frame:")
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

            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.image(annotated_rgb, caption="AI Camera HUD & Trajectory Overlay", use_container_width=True)

            with col2:
                st.markdown(f"""
                <div class="badge-card" style="background-color: {res['severity_color']};">
                    TEMPORAL RISK: {res['severity_label'].upper()}
                </div>
                """, unsafe_allow_html=True)
                st.write("")
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

    # MODE B: WebRTC Stream with Multi-Region STUN & TURN Relay
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

        # Configured with both Google STUN and OpenRelay public TURN servers to traverse cloud firewalls
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


# ----------------- VIDEO UPLOAD PROCESSING -----------------
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
            if not ret or frame_idx > 300:  # process up to 300 frames for preview
                break

            res = lstm_pipeline.process_frame(
                frame,
                simulated_speed=simulated_speed,
                rider_count=simulated_riders,
                helmet_detected=simulated_helmet
            )

            if frame_idx % 2 == 0:  # refresh display every 2 frames
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


# ----------------- IMAGE & BENCHMARK STATIC EVALUATION -----------------
elif raw_image is not None:
    img_np = np.array(raw_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # MODE 1: Vehicle Damage Severity
    if "Vehicle Damage Severity" in app_mode:
        st.subheader("🚗 Vehicle Damage Localization & Severity Grading")
        
        col1, col2 = st.columns([1.2, 1])

        with st.spinner("Analyzing structural damage & surface deformation..."):
            damage_res = damage_clf.assess_damage(img_bgr)
            annotated_rgb = cv2.cvtColor(damage_res["annotated_image"], cv2.COLOR_BGR2RGB)

        with col1:
            st.image(annotated_rgb, caption="AI Damage Bounding Box Localization", use_container_width=True)

        with col2:
            sev_lvl = damage_res["severity_level"]
            sev_color = damage_res["severity_color"]
            
            st.markdown(f"""
            <div class="badge-card" style="background-color: {sev_color};">
                PREDICTED SEVERITY: {damage_res['severity_label'].upper()}
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.plotly_chart(
                create_gauge(damage_res["severity_score"], "Damage Impact Index", sev_color),
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

    # MODE 2: Pre-Crash Violation
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

        with col1:
            st.image(annotated_rgb, caption="Real-time Violation & Telemetry HUD", use_container_width=True)

        with col2:
            v_score = viol_res["risk_score"]
            v_color = "#10B981" if v_score < 0.38 else ("#F59E0B" if v_score < 0.72 else "#EF4444")
            v_tier = "LOW" if v_score < 0.38 else ("MID" if v_score < 0.72 else "HIGH")

            st.markdown(f"""
            <div class="badge-card" style="background-color: {v_color};">
                BEHAVIORAL RISK: {v_tier} RISK
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.plotly_chart(create_gauge(v_score, "Pre-Crash Risk Score", v_color), use_container_width=True)

            st.write("#### Active Violations:")
            if viol_res["violations"]:
                for v in viol_res["violations"]:
                    st.error(f"⚠️ {v}")
            else:
                st.success("✅ No Traffic Rule Violations Detected. Safe Rider Compliance.")

    # MODE 3: Road Hazards & Potholes
    elif "Road Surface Hazard" in app_mode:
        st.subheader("🛣️ Road Hazard & Pothole Segmentation Engine")

        col1, col2 = st.columns([1.2, 1])

        with st.spinner("Scanning road surface defects and depressions..."):
            haz_res = hazard_det.scan_road_hazards(img_bgr)
            annotated_rgb = cv2.cvtColor(haz_res["annotated_frame"], cv2.COLOR_BGR2RGB)

        with col1:
            st.image(annotated_rgb, caption="Road Hazard Identification", use_container_width=True)

        with col2:
            h_score = haz_res["hazard_risk_score"]
            h_color = "#10B981" if h_score < 0.38 else ("#F59E0B" if h_score < 0.72 else "#EF4444")
            h_tier = "LOW" if h_score < 0.38 else ("MID" if h_score < 0.72 else "HIGH")

            st.markdown(f"""
            <div class="badge-card" style="background-color: {h_color};">
                ROAD SURFACE THREAT: {h_tier}
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.plotly_chart(create_gauge(h_score, "Road Hazard Index", h_color), use_container_width=True)

            st.metric("Potholes / Obstacles Count", haz_res["hazard_count"])
            if haz_res["hazard_count"] > 0:
                st.warning("⚠️ Caution: Road depressions identified. Alerting municipal road maintenance.")
            else:
                st.success("✅ Smooth road surface. No major depressions detected.")

    # MODE 4: Unified Multimodal Fusion
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

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(cv2.cvtColor(d_res["annotated_image"], cv2.COLOR_BGR2RGB), caption="Vehicle Inspection Analysis", use_container_width=True)

        with col2:
            st.markdown(f"""
            <div class="badge-card" style="background-color: {fusion_res['overall_color']};">
                OVERALL SEVERITY TIER: {fusion_res['overall_label'].upper()}
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.plotly_chart(create_gauge(fusion_res["composite_score"], "Composite Fusion Risk Score", fusion_res["overall_color"]), use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Multimodal Component Contribution")
        
        components = fusion_res["breakdown"]
        fig_bar = go.Figure(data=[
            go.Bar(
                x=["Damage Severity Index", "Violation Risk Score", "Road Hazard Index"],
                y=[components["damage_component"], components["violation_component"], components["hazard_component"]],
                marker_color=["#EF4444", "#F59E0B", "#3B82F6"]
            )
        ])
        fig_bar.update_layout(yaxis=dict(range=[0, 1]), height=280, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("🚨 Emergency Triage & Incident Dispatch")
        st.error(f"**Dispatched Action:** {fusion_res['action_protocol']}")

else:
    st.info("👆 Please select a benchmark test sample, upload an image/video, or switch to Live Camera mode to begin evaluation.")
