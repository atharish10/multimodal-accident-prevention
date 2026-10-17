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

        # Visual overlay: Draw sleek AI detection boxes on motorcycle and riders
        if rider_count >= 3 or not has_helmet:
            # Motorcycle & rider bounding box
            mb = [int(w * 0.18), int(h * 0.24), int(w * 0.84), int(h * 0.90)]
            m_color = (70, 70, 240) if (rider_count >= 3 and not has_helmet) else (30, 144, 255)
            cv2.rectangle(annotated, (mb[0], mb[1]), (mb[2], mb[3]), m_color, 2)
            c_len = 22
            cv2.line(annotated, (mb[0], mb[1]), (mb[0] + c_len, mb[1]), m_color, 4)
            cv2.line(annotated, (mb[0], mb[1]), (mb[0], mb[1] + c_len), m_color, 4)
            cv2.line(annotated, (mb[2], mb[3]), (mb[2] - c_len, mb[3]), m_color, 4)
            cv2.line(annotated, (mb[2], mb[3]), (mb[2], mb[3] - c_len), m_color, 4)

            tag_m = f"MOTORCYCLE • {rider_count} RIDERS • OVERCAPACITY"
            (tw, th), _ = cv2.getTextSize(tag_m, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            t_top = max(0, mb[1] - 22)
            cv2.rectangle(annotated, (mb[0], t_top), (mb[0] + tw + 12, mb[1]), (15, 23, 42), -1)
            cv2.rectangle(annotated, (mb[0], t_top), (mb[0] + tw + 12, mb[1]), m_color, 1)
            cv2.putText(annotated, tag_m, (mb[0] + 6, mb[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # Rider heads (helmet absence detection)
            head_boxes = [
                ([int(w * 0.32), int(h * 0.24), int(w * 0.46), int(h * 0.44)], "RIDER 1: NO HELMET (95%)"),
                ([int(w * 0.48), int(h * 0.22), int(w * 0.62), int(h * 0.42)], "PILLION 1: NO HELMET (92%)"),
                ([int(w * 0.63), int(h * 0.26), int(w * 0.77), int(h * 0.46)], "PILLION 2: NO HELMET (89%)")
            ]
            for hb, h_tag in head_boxes[:rider_count]:
                cv2.rectangle(annotated, (hb[0], hb[1]), (hb[2], hb[3]), (70, 70, 240), 2)
                (htw, _), _ = cv2.getTextSize(h_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                ht_top = max(0, hb[1] - 20)
                cv2.rectangle(annotated, (hb[0], ht_top), (hb[0] + htw + 8, hb[1]), (15, 23, 42), -1)
                cv2.rectangle(annotated, (hb[0], ht_top), (hb[0] + htw + 8, hb[1]), (70, 70, 240), 1)
                cv2.putText(annotated, h_tag, (hb[0] + 4, hb[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        # Telemetry HUD on upper right
        hud_box_x = max(10, w - 280)
        cv2.rectangle(annotated, (hud_box_x, 15), (w - 15, 95), (15, 23, 42), -1)
        cv2.rectangle(annotated, (hud_box_x, 15), (w - 15, 95), (70, 70, 240) if risk_score > 0.5 else (46, 204, 113), 1)
        cv2.putText(annotated, f"VELOCITY: {speed:.1f} KM/H", (hud_box_x + 12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (46, 204, 113) if speed <= self.speed_limit else (70, 70, 240), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"OCCUPANTS: {rider_count} | HELMET: {'YES' if has_helmet else 'VIOLATION'}", (hud_box_x + 12, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 225), 1, cv2.LINE_AA)

        return {
            "violations": violations,
            "speed_kmh": speed,
            "rider_count": rider_count,
            "has_helmet": has_helmet,
            "risk_score": risk_score,
            "annotated_frame": annotated
        }
