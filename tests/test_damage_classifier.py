import pytest
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from modules.damage_classifier import DamageClassifier
from config import SAMPLE_DATA_DIR, DAMAGE_CLASSES, SEVERITY_LEVELS

class TestDamageClassifier:
    @pytest.fixture
    def classifier(self):
        return DamageClassifier()

    def test_init(self, classifier):
        assert classifier.classes == DAMAGE_CLASSES
        assert classifier.levels == SEVERITY_LEVELS
        assert "LOW" in classifier.levels
        assert "MID" in classifier.levels
        assert "HIGH" in classifier.levels

    def test_benchmark_scenarios_hints(self, classifier):
        # Create a blank test canvas
        dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)

        # Low hint
        low_res = classifier.assess_damage(dummy_img, scenario_hint="Low Severity (Minor Panel Scratch)")
        assert low_res["severity_level"] == "LOW"
        assert low_res["severity_score"] < 0.38
        assert "scratch" in [d["class"] for d in low_res["damages_found"]]
        assert "INS-FAST-TRACK-DIRECT-01" in low_res["triage_code"]

        # Mid hint
        mid_res = classifier.assess_damage(dummy_img, scenario_hint="Moderate Severity (Impact & Bumper Fracture)")
        assert mid_res["severity_level"] == "MID"
        assert 0.38 <= mid_res["severity_score"] <= 0.72
        classes = [d["class"] for d in mid_res["damages_found"]]
        assert "bumper_damage" in classes or "headlight_damage" in classes
        assert "INS-FIELD-ADJUSTER-02" in mid_res["triage_code"]

        # High hint
        high_res = classifier.assess_damage(dummy_img, scenario_hint="High Severity (Structural Cabin Crush)")
        assert high_res["severity_level"] == "HIGH"
        assert high_res["severity_score"] > 0.72
        classes = [d["class"] for d in high_res["damages_found"]]
        assert "structural_crush" in classes or "broken_glass" in classes
        assert "INS-TOTAL-LOSS-SALVAGE-03" in high_res["triage_code"]

    def test_real_sample_images(self, classifier):
        samples = [
            ("sample_low_damage.jpg", "LOW"),
            ("sample_mid_damage.jpg", "MID"),
            ("sample_high_damage.jpg", "HIGH"),
        ]
        for filename, expected_level in samples:
            img_path = SAMPLE_DATA_DIR / filename
            if img_path.exists():
                img = cv2.imread(str(img_path))
                assert img is not None
                res = classifier.assess_damage(img)
                assert res["severity_level"] == expected_level, f"Failed for {filename}: got {res['severity_level']} instead of {expected_level}"
                assert "annotated_image" in res
                assert res["annotated_image"].shape == img.shape
                assert 0.0 <= res["damage_area_percent"] <= 100.0

    def test_custom_detected_damages(self, classifier):
        dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
        custom_damages = [
            {"class": "dent", "confidence": 0.88, "bbox": [50, 50, 200, 200]}
        ]
        res = classifier.assess_damage(dummy_img, detected_damages=custom_damages)
        assert res["severity_level"] == "LOW"  # dent only is typically low/mid
        assert len(res["damages_found"]) == 1

    def test_arbitrary_random_image(self, classifier):
        # Test random noise image doesn't crash
        np.random.seed(42)
        noise_img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        res = classifier.assess_damage(noise_img)
        assert res["severity_level"] in ["LOW", "MID", "HIGH"]
        assert isinstance(res["damage_area_percent"], float)
        assert res["annotated_image"] is not None

    def test_extreme_dimensions(self, classifier):
        # Small image
        small_img = np.zeros((50, 50, 3), dtype=np.uint8)
        res_small = classifier.assess_damage(small_img)
        assert res_small is not None

        # Large image
        large_img = np.zeros((1200, 1600, 3), dtype=np.uint8)
        res_large = classifier.assess_damage(large_img)
        assert res_large is not None
