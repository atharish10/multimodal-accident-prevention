"""
Road Hazard & Surface Defect Detector (Potholes, Obstacles, Debris).
"""

import cv2
import numpy as np
from typing import Dict, List, Any

class HazardDetector:
    """
    Detects road surface defects (potholes, cracks, roadblocks) in the roadway region.
    """

    def scan_road_hazards(self, frame: np.ndarray) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # Road ROI is typically lower half of the frame
        roi_y_start = int(h * 0.55)
        roi = frame[roi_y_start:h, 0:w]

        # Detect dark elliptical concave regions (typical potholes)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_roi, (7, 7), 0)
        
        # Adaptive thresholding to pick up road texture depressions
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 4)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hazards = []
        hazard_penalties = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (w * h * 0.003) < area < (w * h * 0.12):
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / max(1, bh)
                if 0.6 < aspect_ratio < 3.5:
                    real_y = roi_y_start + y
                    is_large = area > (w * h * 0.015)
                    sev_label = "MID" if is_large else "LOW"
                    confidence = 0.88 if is_large else 0.82
                    
                    # Individual hazard penalty based on cavity surface area
                    area_fraction = min(1.0, area / (w * h * 0.03))
                    hazard_penalties.append(0.30 + (0.40 * area_fraction))

                    hazards.append({
                        "type": "pothole",
                        "confidence": confidence,
                        "bbox": [x, real_y, x + bw, real_y + bh],
                        "severity": sev_label
                    })
                    # Draw hazard on image
                    box_color = (0, 140, 255) if is_large else (0, 200, 100)
                    cv2.rectangle(annotated, (x, real_y), (x + bw, real_y + bh), box_color, 2)
                    cv2.putText(annotated, f"POTHOLE ({sev_label})", (x, max(20, real_y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Compound hazard risk without arbitrary cap
        if hazard_penalties:
            safety = 1.0
            for pen in hazard_penalties:
                safety *= (1.0 - min(0.90, pen))
            hazard_score = round(float(1.0 - safety), 3)
        else:
            hazard_score = 0.05

        return {
            "hazards_detected": hazards,
            "hazard_count": len(hazards),
            "hazard_risk_score": hazard_score,
            "annotated_frame": annotated
        }
