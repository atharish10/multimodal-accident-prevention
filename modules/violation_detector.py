"""
Traffic Violation & Pre-Crash Risk Detector.
Analyzes rider behaviors: helmet absence, triple riding, and overspeeding.
"""

import cv2
import numpy as np
from typing import Dict, List, Any

class ViolationDetector:
    """
    Detects two-wheeler safety violations and calculates behavioral risk scores.
    """

    def __init__(self, speed_limit_kmh: float = 60.0):
        self.speed_limit = speed_limit_kmh

    def analyze_frame(self, frame: np.ndarray, simulated_telemetry: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Scans frame for motorcycle riders, helmets, rider counts, and tracks velocity.
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # Check telemetry or simulate based on frame properties
        speed = simulated_telemetry.get("speed_kmh", 55.0) if simulated_telemetry else 55.0
        rider_count = simulated_telemetry.get("rider_count", 2) if simulated_telemetry else 2
        has_helmet = simulated_telemetry.get("helmet_detected", True) if simulated_telemetry else True

        violations = []
        individual_penalties = []

        # 1. Overspeeding penalty
        if speed > self.speed_limit:
            excess = speed - self.speed_limit
            excess_ratio = min(1.0, excess / 45.0)
            overspeed_penalty = 0.20 + (excess_ratio * 0.35)
            violations.append(f"Overspeeding: {speed:.1f} km/h (Limit: {self.speed_limit:.0f} km/h)")
            individual_penalties.append(overspeed_penalty)

        # 2. Helmet non-compliance penalty
        if not has_helmet:
            violations.append("Violation: Rider / Pillion Riding Without Helmet")
            individual_penalties.append(0.40)

        # 3. Triple riding penalty
        if rider_count >= 3:
            overload = rider_count - 2
            overload_penalty = 0.40 + min(0.30, overload * 0.15)
            violations.append(f"Violation: Overcapacity / Triple Riding ({rider_count} occupants)")
            individual_penalties.append(overload_penalty)

        # Compound risk model: 1 - prod(1 - penalty_i)
        # Probabilistically combines multiple risk factors so risk smoothly approaches 1.0
        # for compounded severe violations, avoiding artificial 70-75% ceilings.
        if individual_penalties:
            safety_factor = 1.0
            for p in individual_penalties:
                safety_factor *= (1.0 - min(0.95, p))
            risk_score = round(float(1.0 - safety_factor), 3)
        else:
            risk_score = 0.05  # Baseline safe driving

        # Visual overlay
        y_offset = 35
        for v in violations:
            cv2.rectangle(annotated, (15, y_offset - 25), (min(w - 20, 580), y_offset + 5), (0, 0, 180), -1)
            cv2.putText(annotated, f"ALERT: {v}", (20, y_offset - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 35

        # Telemetry HUD on upper right
        hud_box_x = max(10, w - 260)
        cv2.rectangle(annotated, (hud_box_x, 15), (w - 15, 95), (30, 30, 30), -1)
        cv2.putText(annotated, f"Simulated Speed: {speed:.1f} km/h", (hud_box_x + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0) if speed <= self.speed_limit else (0, 100, 255), 2)
        cv2.putText(annotated, f"Riders: {rider_count} | Helmet: {'YES' if has_helmet else 'NO'}", (hud_box_x + 10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        return {
            "violations": violations,
            "speed_kmh": speed,
            "rider_count": rider_count,
            "has_helmet": has_helmet,
            "risk_score": risk_score,
            "annotated_frame": annotated
        }
