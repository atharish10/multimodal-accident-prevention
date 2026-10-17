import pytest
import numpy as np
import cv2

from modules.hazard_detector import HazardDetector
from config import SAMPLE_DATA_DIR

class TestHazardDetector:
    @pytest.fixture
    def detector(self):
        return HazardDetector()

    def test_benchmark_pothole_image(self, detector):
        pothole_path = SAMPLE_DATA_DIR / "sample_pothole_road.jpg"
        if pothole_path.exists():
            img = cv2.imread(str(pothole_path))
            assert img is not None
            res = detector.scan_road_hazards(img)
            assert res["hazard_count"] >= 1
            assert res["hazard_risk_score"] > 0.50
            assert "Severe" in res["pavement_state"]
            assert "Work-Order" in res["municipal_alert"]
            assert res["annotated_frame"].shape == img.shape

    def test_nominal_smooth_road(self, detector):
        # A completely flat gray road with no dark depressions
        clean_road = np.full((500, 700, 3), 120, dtype=np.uint8)
        res = detector.scan_road_hazards(clean_road)
        assert res["hazard_count"] == 0
        assert res["hazard_risk_score"] == 0.05
        assert "Nominal" in res["pavement_state"]
        assert "Routine" in res["municipal_alert"]

    def test_synthetic_pothole_depression(self, detector):
        # Create an asphalt surface with a dark cavity in the center
        road = np.full((600, 800, 3), 130, dtype=np.uint8)
        # Draw dark pothole cavity
        cv2.circle(road, (400, 300), 50, (20, 20, 20), -1)
        res = detector.scan_road_hazards(road)
        assert res["hazard_count"] >= 1
        assert res["hazard_risk_score"] > 0.05
        assert len(res["hazards_detected"]) >= 1
        haz = res["hazards_detected"][0]
        assert "bbox" in haz
        assert len(haz["bbox"]) == 4

    def test_various_image_sizes(self, detector):
        sizes = [(100, 100), (480, 640), (1080, 1920)]
        for h, w in sizes:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            res = detector.scan_road_hazards(img)
            assert res is not None
            assert "hazard_risk_score" in res
