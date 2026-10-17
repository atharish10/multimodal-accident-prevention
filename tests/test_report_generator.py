import pytest
import numpy as np
from PIL import Image

from modules.report_generator import generate_incident_pdf

class TestReportGenerator:
    def test_minimal_pdf_generation(self):
        pdf_bytes = generate_incident_pdf(
            incident_id="INC-TEST-001",
            modality_type="Car Damage Assessment",
            overall_risk=25.0,
            status_label="Low Severity",
            notes="Routine test assessment."
        )
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF-")

    def test_full_dossier_pdf_generation(self):
        # Create a dummy image
        pil_img = Image.new("RGB", (400, 300), color=(73, 109, 137))

        damage_results = {
            "severity_level": "HIGH",
            "severity_score": 0.88,
            "damage_area_percent": 42.5,
            "repair_estimate": "₹2,85,000 – ₹4,20,000+ (Total Loss Salvage)",
            "triage_code": "INS-TOTAL-LOSS-SALVAGE-03",
            "damages_found": [
                {"class": "structural_crush", "confidence": 0.96, "bbox": [50, 50, 350, 250]},
                {"class": "broken_glass", "confidence": 0.92, "bbox": [100, 80, 250, 180]}
            ]
        }

        violation_results = {
            "risk_score": 0.78,
            "speed_kmh": 85.0,
            "rider_count": 3,
            "has_helmet": False,
            "violations": [
                "Overspeeding: 85.0 km/h (Limit: 60 km/h)",
                "Violation: Overcapacity / Triple Riding",
                "Violation: Rider Without Helmet"
            ]
        }

        hazard_results = {
            "hazard_risk_score": 0.65,
            "hazard_count": 2,
            "pavement_state": "Severe Structural Pothole",
            "municipal_alert": "Urgent Maintenance Work-Order Dispatched",
            "hazards_detected": [
                {"type": "pothole", "severity": "HIGH", "confidence": 0.94, "depth_est": "6.8 cm"}
            ]
        }

        env_factors = {
            "weather": "Heavy Monsoon Rain",
            "lighting": "350 Lux",
            "traffic_density": "High Density Urban Corridor",
            "multiplier": 1.35,
            "road_surface": "Wet Slippery Asphalt",
            "vehicle_class": "Compact SUV (1,850 kg)",
            "speed_kmh": "75.0 km/h"
        }

        notes = "Multi-vehicle critical accident occurred at junction. Immediate tier-1 trauma triage required."

        pdf_bytes = generate_incident_pdf(
            incident_id="INC-PEC-2026-9912",
            modality_type="Multimodal Cross-Sensor Fusion",
            overall_risk=82.5,
            status_label="Critical Incident",
            damage_results=damage_results,
            violation_results=violation_results,
            hazard_results=hazard_results,
            annotated_image=pil_img,
            environmental_factors=env_factors,
            notes=notes
        )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 5000
        assert pdf_bytes.startswith(b"%PDF-")

    def test_pdf_with_numpy_image(self):
        np_img = np.zeros((300, 400, 3), dtype=np.uint8)
        pdf_bytes = generate_incident_pdf(
            incident_id="INC-NUMPY-TEST",
            modality_type="Road Hazard Scanner",
            overall_risk=55.0,
            status_label="Moderate Threat",
            annotated_image=np_img
        )
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF-")

    def test_extreme_risk_boundary_colors(self):
        # Test low risk (< 40)
        low_pdf = generate_incident_pdf("INC-LOW", "Modality", 20.0, "Low")
        assert low_pdf.startswith(b"%PDF-")

        # Test mid risk (40 - 70)
        mid_pdf = generate_incident_pdf("INC-MID", "Modality", 55.0, "Moderate")
        assert mid_pdf.startswith(b"%PDF-")

        # Test high risk (>= 70)
        high_pdf = generate_incident_pdf("INC-HIGH", "Modality", 95.0, "Critical")
        assert high_pdf.startswith(b"%PDF-")
