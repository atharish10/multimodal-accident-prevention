import pytest
import numpy as np
import cv2

from modules.violation_detector import ViolationDetector

class TestViolationDetector:
    @pytest.fixture
    def detector(self):
        return ViolationDetector(speed_limit_kmh=60.0)

    def test_init(self, detector):
        assert detector.speed_limit == 60.0

    def test_safe_driving_baseline(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        telemetry = {
            "speed_kmh": 45.0,
            "rider_count": 1,
            "helmet_detected": True
        }
        res = detector.analyze_frame(frame, telemetry)
        assert res["risk_score"] == 0.05
        assert len(res["violations"]) == 0
        assert res["speed_kmh"] == 45.0
        assert res["rider_count"] == 1
        assert res["has_helmet"] is True
        assert res["annotated_frame"] is not None

    def test_overspeeding_violation(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        telemetry = {
            "speed_kmh": 85.0,  # 25 km/h above 60 limit
            "rider_count": 1,
            "helmet_detected": True
        }
        res = detector.analyze_frame(frame, telemetry)
        assert len(res["violations"]) == 1
        assert "Overspeeding" in res["violations"][0]
        assert res["risk_score"] > 0.20

    def test_helmet_violation(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        telemetry = {
            "speed_kmh": 40.0,
            "rider_count": 1,
            "helmet_detected": False
        }
        res = detector.analyze_frame(frame, telemetry)
        assert len(res["violations"]) == 1
        assert "Without Helmet" in res["violations"][0]
        assert res["risk_score"] >= 0.40

    def test_triple_riding_violation(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        telemetry = {
            "speed_kmh": 40.0,
            "rider_count": 3,
            "helmet_detected": True
        }
        res = detector.analyze_frame(frame, telemetry)
        assert len(res["violations"]) == 1
        assert "Triple Riding" in res["violations"][0]
        assert res["risk_score"] >= 0.40

    def test_compounded_severe_violations(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        telemetry = {
            "speed_kmh": 95.0,   # Extreme overspeeding
            "rider_count": 3,    # Triple riding
            "helmet_detected": False  # No helmet
        }
        res = detector.analyze_frame(frame, telemetry)
        assert len(res["violations"]) == 3
        # Compounded score should be high (> 0.75) and <= 1.0
        assert res["risk_score"] > 0.75
        assert res["risk_score"] <= 1.0

    def test_extreme_and_boundary_telemetry(self, detector):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)

        # 0 km/h, 1 rider, helmet True
        res_zero = detector.analyze_frame(frame, {"speed_kmh": 0.0, "rider_count": 1, "helmet_detected": True})
        assert res_zero["risk_score"] == 0.05

        # 200 km/h, 4 riders, helmet False
        res_ext = detector.analyze_frame(frame, {"speed_kmh": 200.0, "rider_count": 4, "helmet_detected": False})
        assert res_ext["risk_score"] > 0.85
        assert res_ext["risk_score"] <= 1.0

        # None telemetry should use safe defaults
        res_none = detector.analyze_frame(frame, None)
        assert res_none is not None
        assert "risk_score" in res_none
