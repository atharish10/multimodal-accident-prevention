"""
Configuration and constants for Multimodal AI System for Early Detection,
Accident Prevention & Damage Severity Classification.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Severity Levels
SEVERITY_LEVELS = {
    "LOW": {
        "label": "Low Severity",
        "color": "#10B981",  # Green
        "description": "Minor superficial damage (scratches, small dents, light abrasion). Vehicle driveable. Low safety hazard.",
        "action": "Immediate assistance not required. Standard insurance documentation / road maintenance ticket."
    },
    "MID": {
        "label": "Mid / Moderate Severity",
        "color": "#F59E0B",  # Amber/Orange
        "description": "Moderate structural damage (bumper detachment, shattered lights, cracked glass, localized deformation).",
        "action": "Dispatch roadside assistance/towing. Cordon off lane hazard if debris present."
    },
    "HIGH": {
        "label": "High / Critical Severity",
        "color": "#EF4444",  # Red
        "description": "Severe structural deformation (cabin intrusion, front-end crush, axle failure, rollover risk).",
        "action": "URGENT: Alert emergency medical services (108/911), traffic police, and deploy emergency rescue unit."
    }
}

# Damage Classes and Weights for Scoring
DAMAGE_CLASSES = {
    "scratch": {"weight": 0.15, "typical_severity": "LOW"},
    "dent": {"weight": 0.35, "typical_severity": "MID"},
    "bumper_damage": {"weight": 0.50, "typical_severity": "MID"},
    "broken_glass": {"weight": 0.60, "typical_severity": "MID"},
    "headlight_damage": {"weight": 0.40, "typical_severity": "MID"},
    "structural_crush": {"weight": 0.95, "typical_severity": "HIGH"},
    "rollover_damage": {"weight": 1.00, "typical_severity": "HIGH"},
}

# Pre-Crash Violation & Hazard Weights
RISK_WEIGHTS = {
    "overspeeding": 0.30,
    "helmet_violation": 0.25,
    "triple_riding": 0.25,
    "pothole_hazard": 0.20,
    "near_miss_proximity": 0.35
}

# Thresholds for overall Severity assignment
SEVERITY_THRESHOLDS = {
    "LOW_MAX": 0.38,
    "MID_MAX": 0.72
    # > 0.72 is HIGH
}
