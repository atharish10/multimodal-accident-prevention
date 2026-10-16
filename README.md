# Multimodal AI System for Early Detection, Accident Prevention & Vehicle Damage Severity Classification

**Department:** B.Tech Artificial Intelligence & Data Science  
**Project Team:** HARISH AT & BARATHRAJ AS  
**Supervisor / Guide:** Dr. A. ARUN  
**Institution:** Panimalar Engineering College  

---

## 📌 Project Overview
This project presents an integrated computer vision and multimodal intelligence framework designed to address both **pre-crash accident prevention** and **post-crash vehicle damage severity classification**.

Unlike legacy systems that either exclusively classify damage after collisions or rely on expensive dedicated hardware (radar/lidar), this software-driven architecture operates on standard camera feeds, continuous video streams, and vehicle telemetry to:
1. **Continuous Live Camera Monitoring (CNN + LSTM):** Evaluates rolling temporal sequences (10-frame buffer) using a lightweight CNN feature encoder coupled with a 2-layer LSTM sequence network for trajectory dynamics and collision proximity assessment.
2. **Detect pre-crash risk behaviors:** Overspeeding, missing helmets, and hazardous triple riding.
3. **Segment roadway hazards:** Identify potholes, road cavities, and surface obstacles.
4. **Classify Vehicle Damage Severity:** Grade collision impacts into **Low**, **Mid (Medium)**, or **High** severity tiers based on defect category, deformation area coverage, and structural damage.
5. **Coordinate Emergency Response:** Trigger real-time incident protocols and provide an automated triage breakdown.

---

## 🚦 Severity Classification Criteria (Low / Mid / High)

| Severity Level | Color | Damage Indicators | Action / Protocol |
| :--- | :---: | :--- | :--- |
| **LOW** | 🟢 Green | Minor superficial scratches, paint abrasions, small dents (<4% surface area). No structural compromise; vehicle driveable. | Standard insurance photo documentation; no emergency dispatch required. |
| **MID (Moderate)** | 🟡 Amber | Broken headlights, shattered windshield, bumper displacement/sagging, localized panel deformation (4%–30% area). | Roadside towing assistance dispatch; traffic police alert to clear lane obstructions. |
| **HIGH (Critical)** | 🔴 Red | Severe structural crush, front-end crumple, roof collapse / rollover risk, axle breakage (>30% area). | **Immediate 108/911 ambulance dispatch**, emergency rescue team, hospital trauma center notification. |

---

## ⚙️ Architecture & Modules

```text
multimodal-accident-prevention/
├── app.py                         # Interactive Streamlit Web UI Dashboard
├── config.py                      # System thresholds, severity classes, and weights
├── run.sh                         # One-click startup script
├── requirements.txt               # Python dependencies
├── modules/
│   ├── damage_classifier.py       # Damage localization & Low/Mid/High severity engine
│   ├── violation_detector.py      # Helmet, triple riding & speed violation analyzer
│   ├── hazard_detector.py         # Pothole & road defect segmentation
│   ├── risk_fusion.py             # Multimodal pre/post-crash risk fusion
│   ├── temporal_risk_lstm.py      # Spatio-Temporal CNN + LSTM rolling sequence engine
│   └── __init__.py
└── sample_data/                   # Synthetic & benchmark evaluation datasets
    └── generate_sample_media.py
```

---

## 🚀 Quick Start Guide

### 1. Launch the Application
Simply execute the provided startup script:
```bash
./run.sh
```
Or manually run:
```bash
source venv/bin/activate
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 2. Available Modes in the Dashboard:
* **🚗 Vehicle Damage Severity Classification:** Test images to localize damage areas and inspect the Low / Mid / High impact index.
* **🏍️ Pre-Crash Violation & Speeding Monitor:** Live simulation of two-wheeler speed and rider compliance (helmets, passenger count).
* **🛣️ Road Surface Hazard Detection:** Identify road depressions and potholes in real-time.
* **⚡ Unified Multimodal Fusion:** Fuses active data sources to provide a unified threat score and emergency action plan.
* **📹 Live Camera Traffic Risk Monitor (CNN + LSTM):** Uses laptop webcam via WebRTC or direct OpenCV capture to process continuous rolling frame sequences.
