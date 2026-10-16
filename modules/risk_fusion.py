"""
Multimodal Risk & Severity Fusion Engine.
Merges pre-crash indicators and post-crash damage assessment into a unified
Low / Mid / High severity evaluation.
"""

from typing import Dict, Any
from config import SEVERITY_LEVELS, SEVERITY_THRESHOLDS

class MultimodalRiskFusionEngine:
    """
    Fuses vision streams, vehicle telemetry, and damage characteristics.
    Normalizes active modalities to prevent artificial score ceilings.
    """

    def __init__(self):
        self.levels = SEVERITY_LEVELS
        self.thresholds = SEVERITY_THRESHOLDS

    def fuse_assessment(self,
                        damage_result: Dict[str, Any] = None,
                        violation_result: Dict[str, Any] = None,
                        hazard_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculates composite risk score and maps to LOW, MID, or HIGH.
        Dynamically balances weights across active modalities so that single or
        dual-modality evaluations are not artificially suppressed or capped.
        """
        damage_score = damage_result.get("severity_score", 0.0) if damage_result else 0.0
        violation_score = violation_result.get("risk_score", 0.0) if violation_result else 0.0
        hazard_score = hazard_result.get("hazard_risk_score", 0.0) if hazard_result else 0.0

        active_weights = []
        weighted_sum = 0.0

        # Primary baseline weights: Damage (0.70), Violation (0.20), Hazard (0.10)
        # Only observed/applicable components are weighted, preventing score dilution.
        if damage_score > 0.0:
            active_weights.append(0.70)
            weighted_sum += damage_score * 0.70

        if violation_score > 0.0:
            active_weights.append(0.20)
            weighted_sum += violation_score * 0.20

        if hazard_score > 0.0:
            active_weights.append(0.10)
            weighted_sum += hazard_score * 0.10

        if active_weights:
            total_active_weight = sum(active_weights)
            composite_score = weighted_sum / total_active_weight
        else:
            composite_score = 0.0

        composite_score = round(min(1.0, max(0.0, composite_score)), 3)

        # Assign tier
        if composite_score <= self.thresholds["LOW_MAX"]:
            severity_tier = "LOW"
        elif composite_score <= self.thresholds["MID_MAX"]:
            severity_tier = "MID"
        else:
            severity_tier = "HIGH"

        level_info = self.levels[severity_tier]

        return {
            "overall_severity": severity_tier,
            "overall_label": level_info["label"],
            "overall_color": level_info["color"],
            "composite_score": composite_score,
            "breakdown": {
                "damage_component": round(damage_score, 3),
                "violation_component": round(violation_score, 3),
                "hazard_component": round(hazard_score, 3)
            },
            "description": level_info["description"],
            "action_protocol": level_info["action"]
        }
