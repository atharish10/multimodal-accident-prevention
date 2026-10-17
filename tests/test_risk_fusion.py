import pytest
from modules.risk_fusion import MultimodalRiskFusionEngine
from config import SEVERITY_LEVELS, SEVERITY_THRESHOLDS

class TestMultimodalRiskFusionEngine:
    @pytest.fixture
    def engine(self):
        return MultimodalRiskFusionEngine()

    def test_init(self, engine):
        assert engine.levels == SEVERITY_LEVELS
        assert engine.thresholds == SEVERITY_THRESHOLDS

    def test_empty_or_none_inputs(self, engine):
        res = engine.fuse_assessment(None, None, None)
        assert res["composite_score"] == 0.0
        assert res["overall_severity"] == "LOW"
        assert res["breakdown"]["damage_component"] == 0.0
        assert res["breakdown"]["violation_component"] == 0.0
        assert res["breakdown"]["hazard_component"] == 0.0

    def test_single_modality_damage_low(self, engine):
        damage_res = {"severity_score": 0.22}
        res = engine.fuse_assessment(damage_result=damage_res)
        assert res["composite_score"] == 0.22
        assert res["overall_severity"] == "LOW"

    def test_single_modality_damage_high(self, engine):
        damage_res = {"severity_score": 0.88}
        res = engine.fuse_assessment(damage_result=damage_res)
        assert res["composite_score"] == 0.88
        assert res["overall_severity"] == "HIGH"

    def test_single_modality_violation(self, engine):
        viol_res = {"risk_score": 0.65}
        res = engine.fuse_assessment(violation_result=viol_res)
        assert res["composite_score"] == 0.65
        assert res["overall_severity"] == "MID"

    def test_single_modality_hazard(self, engine):
        haz_res = {"hazard_risk_score": 0.75}
        res = engine.fuse_assessment(hazard_result=haz_res)
        assert res["composite_score"] == 0.75
        assert res["overall_severity"] == "HIGH"

    def test_multi_modality_weighted_fusion(self, engine):
        # Damage (0.70 weight): 0.50
        # Violation (0.20 weight): 0.80
        # Hazard (0.10 weight): 0.40
        # Expected: (0.50*0.70 + 0.80*0.20 + 0.40*0.10) / (0.70 + 0.20 + 0.10)
        # = (0.35 + 0.16 + 0.04) / 1.0 = 0.55
        damage_res = {"severity_score": 0.50}
        viol_res = {"risk_score": 0.80}
        haz_res = {"hazard_risk_score": 0.40}
        res = engine.fuse_assessment(damage_res, viol_res, haz_res)
        assert abs(res["composite_score"] - 0.55) < 0.01
        assert res["overall_severity"] == "MID"

    def test_threshold_boundaries(self, engine):
        # Exactly at LOW_MAX (0.38)
        res_low = engine.fuse_assessment(damage_result={"severity_score": 0.38})
        assert res_low["overall_severity"] == "LOW"

        # Just above LOW_MAX (0.39)
        res_mid1 = engine.fuse_assessment(damage_result={"severity_score": 0.39})
        assert res_mid1["overall_severity"] == "MID"

        # Exactly at MID_MAX (0.72)
        res_mid2 = engine.fuse_assessment(damage_result={"severity_score": 0.72})
        assert res_mid2["overall_severity"] == "MID"

        # Just above MID_MAX (0.73)
        res_high = engine.fuse_assessment(damage_result={"severity_score": 0.73})
        assert res_high["overall_severity"] == "HIGH"
