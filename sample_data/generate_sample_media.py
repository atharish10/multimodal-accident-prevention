"""
Generates realistic benchmark sample images representing Low, Mid, and High damage,
as well as traffic violations and road hazard scenarios for testing.
"""

import os
import cv2
import numpy as np
from pathlib import Path

def generate_samples(output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Low Damage Sample (Blue sedan with small scratch / minor rear dent)
    img_low = np.full((500, 750, 3), (240, 240, 240), dtype=np.uint8)
    # Draw roadway / ground
    cv2.rectangle(img_low, (0, 350), (750, 500), (90, 95, 100), -1)
    # Draw car body silhouette
    pts = np.array([[120, 380], [180, 260], [380, 240], [550, 310], [640, 380]], np.int32)
    cv2.fillPoly(img_low, [pts], (180, 110, 50))  # Metallic blue
    cv2.rectangle(img_low, (100, 360), (660, 420), (190, 120, 60), -1)
    # Wheels
    cv2.circle(img_low, (210, 420), 45, (30, 30, 30), -1)
    cv2.circle(img_low, (560, 420), 45, (30, 30, 30), -1)
    # Minor scratch (thin irregular line)
    cv2.line(img_low, (320, 380), (390, 388), (245, 245, 245), 2)
    cv2.putText(img_low, "BENCHMARK SAMPLE: MINOR REAR-PANEL SCRATCH", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2)
    cv2.imwrite(str(output_dir / "sample_low_damage.jpg"), img_low)

    # 2. Mid Damage Sample (Silver car with deformed bumper and broken headlight)
    img_mid = np.full((500, 750, 3), (230, 230, 230), dtype=np.uint8)
    cv2.rectangle(img_mid, (0, 350), (750, 500), (90, 95, 100), -1)
    # Front-facing car
    cv2.rectangle(img_mid, (180, 230), (570, 400), (160, 160, 160), -1) # Body
    cv2.rectangle(img_mid, (220, 160), (530, 250), (130, 150, 160), -1) # Windshield
    # Headlights
    cv2.rectangle(img_mid, (200, 280), (280, 330), (255, 255, 220), -1) # Intact
    cv2.rectangle(img_mid, (470, 280), (550, 330), (70, 70, 70), -1)    # Broken
    # Cracked lines in damaged headlight
    for dx in [10, 25, 40, 60]:
        cv2.line(img_mid, (470 + dx, 280), (480 + dx, 330), (220, 220, 220), 2)
    # Dented sagging bumper on right
    pts_bump = np.array([[170, 390], [420, 390], [580, 450], [560, 465], [170, 415]], np.int32)
    cv2.fillPoly(img_mid, [pts_bump], (90, 90, 90))
    cv2.putText(img_mid, "BENCHMARK SAMPLE: MODERATE BUMPER & HEADLIGHT DAMAGE", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.imwrite(str(output_dir / "sample_mid_damage.jpg"), img_mid)

    # 3. High Damage Sample (Severe crash collision, crushed front, shattered glass)
    img_high = np.full((500, 750, 3), (220, 220, 220), dtype=np.uint8)
    cv2.rectangle(img_high, (0, 330), (750, 500), (80, 80, 80), -1)
    # Heavy distorted chassis
    pts_wreck = np.array([[120, 420], [180, 310], [320, 260], [430, 210], [480, 350], [650, 430], [300, 460]], np.int32)
    cv2.fillPoly(img_high, [pts_wreck], (40, 40, 180)) # Red car crushed
    # Smashed shattered windshield
    cv2.fillPoly(img_high, [np.array([[220, 300], [350, 270], [320, 350], [190, 350]], np.int32)], (200, 210, 220))
    # Spiderweb crack lines
    for i in range(12):
        cv2.line(img_high, (260, 310), (210 + i * 15, 290 + (i % 4) * 20), (50, 50, 50), 2)
    # Severely crumpled front hood (accordion folds)
    for k in range(5):
        cv2.line(img_high, (350 + k * 20, 260 + (k % 2) * 30), (370 + k * 20, 380), (20, 20, 100), 4)
    cv2.putText(img_high, "BENCHMARK SAMPLE: HIGH/CRITICAL STRUCTURAL CRUSH", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 180), 2)
    cv2.imwrite(str(output_dir / "sample_high_damage.jpg"), img_high)

    # 4. Road Pothole Sample
    img_pothole = np.full((500, 750, 3), (110, 115, 120), dtype=np.uint8) # Dark asphalt
    # White lane lines
    cv2.rectangle(img_pothole, (360, 0), (390, 120), (240, 240, 240), -1)
    cv2.rectangle(img_pothole, (360, 220), (390, 340), (240, 240, 240), -1)
    cv2.rectangle(img_pothole, (360, 420), (390, 500), (240, 240, 240), -1)
    # Deep jagged pothole
    cv2.ellipse(img_pothole, (280, 360), (130, 65), -15, 0, 360, (35, 35, 40), -1)
    cv2.ellipse(img_pothole, (280, 360), (105, 45), -15, 0, 360, (15, 15, 20), -1)
    cv2.putText(img_pothole, "BENCHMARK SAMPLE: ROAD DEEP CAVITY POTHOLE", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.imwrite(str(output_dir / "sample_pothole_road.jpg"), img_pothole)

    # 5. Two-Wheeler Pre-crash scenario
    img_bike = np.full((500, 750, 3), (215, 225, 235), dtype=np.uint8)
    cv2.rectangle(img_bike, (0, 320), (750, 500), (85, 90, 95), -1)
    # Bike frame
    cv2.circle(img_bike, (260, 420), 40, (30, 30, 30), 10)
    cv2.circle(img_bike, (490, 420), 40, (30, 30, 30), 10)
    cv2.line(img_bike, (260, 420), (370, 410), (180, 30, 30), 12)
    cv2.line(img_bike, (370, 410), (490, 420), (180, 30, 30), 12)
    cv2.line(img_bike, (370, 410), (350, 290), (180, 30, 30), 10)
    # 3 Riders (heads without helmets)
    for head_x in [330, 380, 430]:
        cv2.circle(img_bike, (head_x, 230), 22, (70, 120, 210), -1) # Head
        cv2.circle(img_bike, (head_x, 230), 22, (20, 20, 180), 2)   # No helmet border
        cv2.rectangle(img_bike, (head_x - 18, 252), (head_x + 18, 330), (60, 60, 60), -1) # Body
    cv2.putText(img_bike, "BENCHMARK SAMPLE: TWO-WHEELER TRIPLE RIDING (NO HELMETS)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 150), 2)
    cv2.imwrite(str(output_dir / "sample_two_wheeler_violation.jpg"), img_bike)

    print(f"Sample media successfully generated in {output_dir}")

if __name__ == "__main__":
    generate_samples(Path(__file__).resolve().parent)
