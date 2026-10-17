"""
High-Fidelity Realistic Benchmark Media Generator for Multimodal Road Safety AI.
Generates realistic benchmark scenes with asphalt textures, vehicle lighting,
metallic gradients, realistic impact deformation, fractured glass, and road defects.
"""

import os
import cv2
import numpy as np
from pathlib import Path

def create_asphalt_road(w: int, h: int, horizon_y: int = 240) -> np.ndarray:
    """Generates asphalt road background with realistic grain, gradient, and perspective lane lines."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Sky / background gradient
    for y in range(horizon_y):
        ratio = y / float(horizon_y)
        b = int(215 - 50 * ratio)
        g = int(225 - 40 * ratio)
        r = int(235 - 30 * ratio)
        img[y, :] = (b, g, r)

    # Road surface gradient (darker near horizon, textured asphalt near camera)
    for y in range(horizon_y, h):
        ratio = (y - horizon_y) / float(h - horizon_y)
        base = int(48 + 32 * ratio)
        img[y, :] = (base, base + 2, base + 4)

    # Asphalt grain noise
    noise = np.random.normal(0, 7, (h - horizon_y, w, 3)).astype(np.int16)
    road_area = img[horizon_y:h, :].astype(np.int16) + noise
    img[horizon_y:h, :] = np.clip(road_area, 0, 255).astype(np.uint8)

    # Road shoulder / curb
    cv2.line(img, (int(w * 0.15), horizon_y), (0, h), (140, 140, 145), 3)
    cv2.line(img, (int(w * 0.85), horizon_y), (w, h), (140, 140, 145), 3)

    # Perspective dashed center lines
    for i in range(5):
        y1 = int(horizon_y + 15 + i * 55)
        y2 = int(y1 + 25 + i * 15)
        if y2 < h:
            lw = int(2 + i * 2)
            cv2.line(img, (w // 2, y1), (w // 2, y2), (240, 240, 240), lw)

    return img


def generate_samples(output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)
    w, h = 800, 520

    # =========================================================================
    # 1. LOW DAMAGE: Modern Blue Sedan with Minor Rear Quarter Scratch
    # =========================================================================
    img1 = create_asphalt_road(w, h, horizon_y=260)
    
    # Vehicle ground shadow
    cv2.ellipse(img1, (400, 445), (280, 25), 0, 0, 360, (20, 20, 22), -1)

    # Car body contour (Sedan side profile)
    body_pts = np.array([
        [150, 420], [140, 380], [170, 340], [280, 335], [360, 275],
        [540, 275], [630, 340], [700, 355], [710, 410], [670, 435],
        [590, 435], [560, 400], [470, 400], [440, 435], [260, 435],
        [230, 400], [180, 400]
    ], np.int32)
    
    # Metallic Blue Body Fill
    cv2.fillPoly(img1, [body_pts], (165, 85, 30))
    
    # Cabin glass
    glass_pts = np.array([[370, 280], [530, 280], [615, 335], [290, 335]], np.int32)
    cv2.fillPoly(img1, [glass_pts], (120, 140, 150))
    cv2.line(img1, (455, 280), (455, 335), (70, 70, 75), 4) # Pillar

    # Wheels & Rims
    for cx in [205, 515]:
        cv2.circle(img1, (cx, 418), 42, (25, 25, 28), -1)
        cv2.circle(img1, (cx, 418), 24, (180, 185, 190), -1)
        cv2.circle(img1, (cx, 418), 10, (50, 50, 55), -1)

    # Minor Scratch: Fine jagged white/silver line on door panel
    scratch_pts = np.array([[225, 360], [250, 363], [275, 357], [305, 362]], np.int32)
    cv2.polylines(img1, [scratch_pts], False, (255, 255, 255), 2)
    cv2.polylines(img1, [scratch_pts + np.array([0, 1])], False, (70, 35, 15), 1)

    cv2.putText(img1, "BENCHMARK SAMPLE: MINOR CLEAR-COAT PANEL SCRATCH (LOW SEVERITY)", (25, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.imwrite(str(output_dir / "sample_low_damage.jpg"), img1)

    # =========================================================================
    # 2. MID DAMAGE: Front Vehicle with Damaged Bumper & Fractured Headlight
    # =========================================================================
    img2 = create_asphalt_road(w, h, horizon_y=250)
    cv2.ellipse(img2, (400, 455), (260, 28), 0, 0, 360, (18, 18, 20), -1)

    # Frontal sedan silhouette
    cv2.rectangle(img2, (210, 270), (590, 420), (145, 148, 152), -1) # Silver Body
    cv2.rectangle(img2, (260, 200), (540, 275), (90, 115, 130), -1)  # Windshield
    cv2.line(img2, (400, 200), (400, 275), (50, 50, 55), 3)

    # Radiator Grille
    cv2.rectangle(img2, (310, 330), (490, 395), (35, 35, 38), -1)
    for gy in range(335, 395, 10):
        cv2.line(img2, (315, gy), (485, gy), (90, 90, 95), 1)

    # Intact Left Headlight
    cv2.rectangle(img2, (225, 315), (295, 355), (255, 255, 230), -1)
    cv2.rectangle(img2, (225, 315), (295, 355), (100, 100, 105), 2)

    # Fractured Right Headlight (shattered dark plastic with crack web)
    cv2.rectangle(img2, (505, 315), (575, 355), (60, 60, 65), -1)
    for dx in [12, 28, 45, 60]:
        cv2.line(img2, (505 + dx, 315), (515 + dx, 355), (210, 210, 220), 2)
    cv2.line(img2, (510, 335), (570, 335), (230, 230, 240), 2)

    # Deformed Sagging Bumper (dropped on right side)
    bump_pts = np.array([
        [205, 415], [390, 418], [585, 465], [595, 435],
        [490, 398], [205, 398]
    ], np.int32)
    cv2.fillPoly(img2, [bump_pts], (75, 78, 82))
    cv2.polylines(img2, [bump_pts], True, (40, 40, 42), 2)

    cv2.putText(img2, "BENCHMARK SAMPLE: MODERATE BUMPER & HEADLIGHT DAMAGE (MID SEVERITY)", (25, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.imwrite(str(output_dir / "sample_mid_damage.jpg"), img2)

    # =========================================================================
    # 3. HIGH DAMAGE: Severe Collision, Crumpled Front End & Smashed Windshield
    # =========================================================================
    img3 = create_asphalt_road(w, h, horizon_y=240)
    cv2.ellipse(img3, (400, 465), (290, 30), 0, 0, 360, (15, 15, 18), -1)

    # Crumpled body shell (angled collision deformation)
    wreck_pts = np.array([
        [160, 440], [210, 320], [330, 270], [450, 220], [530, 320],
        [690, 440], [580, 470], [380, 450], [260, 470]
    ], np.int32)
    cv2.fillPoly(img3, [wreck_pts], (35, 35, 175)) # Deep Crimson Red

    # Smashed shattered windshield
    glass_wreck = np.array([[260, 290], [390, 250], [360, 335], [230, 340]], np.int32)
    cv2.fillPoly(img3, [glass_wreck], (180, 195, 205))
    # Spiderweb shatter lines
    for i in range(16):
        ex = int(220 + i * 11)
        ey = int(260 + (i % 5) * 15)
        cv2.line(img3, (300, 300), (ex, ey), (40, 40, 45), 2)
    cv2.circle(img3, (300, 300), 22, (230, 240, 250), -1) # Point of impact

    # Accordion crumple folds on engine hood
    for k in range(7):
        kx = 370 + k * 28
        ky1 = 280 + (k % 2) * 35
        ky2 = 410 - (k % 3) * 20
        cv2.line(img3, (kx, ky1), (kx + 15, ky2), (15, 15, 80), 4)
        cv2.line(img3, (kx + 2, ky1), (kx + 17, ky2), (230, 100, 100), 2)

    # Detached hanging bumper piece
    frag_pts = np.array([[480, 420], [640, 465], [600, 485], [450, 445]], np.int32)
    cv2.fillPoly(img3, [frag_pts], (30, 30, 32))

    cv2.putText(img3, "BENCHMARK SAMPLE: HIGH/CRITICAL STRUCTURAL CRUSH (HIGH SEVERITY)", (25, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.imwrite(str(output_dir / "sample_high_damage.jpg"), img3)

    # =========================================================================
    # 4. ROAD HAZARD: Deep Cavity Pothole with Fracture Rim
    # =========================================================================
    img4 = create_asphalt_road(w, h, horizon_y=180)

    # Deep jagged pothole depression
    cv2.ellipse(img4, (390, 365), (155, 75), -12, 0, 360, (22, 22, 25), -1) # Outer depth
    cv2.ellipse(img4, (380, 370), (125, 55), -12, 0, 360, (10, 10, 12), -1) # Deep core
    
    # Asphalt rim fractures and loose aggregate
    for ang in range(0, 360, 25):
        rad = np.radians(ang)
        px1 = int(390 + 150 * np.cos(rad))
        py1 = int(365 + 72 * np.sin(rad))
        px2 = int(390 + (165 + (ang % 15)) * np.cos(rad))
        py2 = int(365 + (80 + (ang % 10)) * np.sin(rad))
        cv2.line(img4, (px1, py1), (px2, py2), (15, 15, 18), 2)
        cv2.circle(img4, (px2, py2), 3, (160, 160, 165), -1) # Stones

    cv2.putText(img4, "BENCHMARK SAMPLE: ROAD DEEP CAVITY POTHOLE HAZARD", (25, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
    cv2.imwrite(str(output_dir / "sample_pothole_road.jpg"), img4)

    # =========================================================================
    # 5. TWO-WHEELER VIOLATION: Triple Riding with Missing Helmets
    # =========================================================================
    img5 = create_asphalt_road(w, h, horizon_y=240)
    cv2.ellipse(img5, (400, 455), (190, 22), 0, 0, 360, (20, 20, 22), -1)

    # Motorcycle Wheels & Chassis
    cv2.circle(img5, (255, 415), 45, (25, 25, 28), 12) # Rear wheel
    cv2.circle(img5, (535, 415), 45, (25, 25, 28), 12) # Front wheel
    cv2.line(img5, (255, 415), (395, 405), (175, 25, 25), 14) # Frame
    cv2.line(img5, (395, 405), (535, 415), (175, 25, 25), 14)
    cv2.line(img5, (395, 405), (360, 305), (175, 25, 25), 12) # Seat
    cv2.line(img5, (535, 415), (500, 290), (70, 70, 75), 10)  # Fork
    cv2.line(img5, (485, 290), (515, 290), (30, 30, 32), 6)   # Handlebar

    # Triple Riders (3 human heads WITHOUT helmets)
    heads = [(335, 220), (390, 215), (445, 225)]
    for idx, (hx, hy) in enumerate(heads):
        # Body / Jacket
        cv2.rectangle(img5, (hx - 20, hy + 22), (hx + 20, hy + 95), (45 + idx * 25, 50, 60 + idx * 30), -1)
        # Head (Hair, skin tone, NO helmet)
        cv2.circle(img5, (hx, hy), 22, (60, 110, 185), -1) # Skin tone (BGR)
        cv2.ellipse(img5, (hx, hy - 8), (22, 14), 0, 180, 360, (20, 20, 25), -1) # Dark hair

    # =========================================================================
    # 6. VIDEO STREAM BENCHMARK: Highway Traffic Dashcam Looming Sequence
    # =========================================================================
    vid_path = str(output_dir / "sample_traffic_dashcam.mp4")
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw, vh = 640, 360
        out_vid = cv2.VideoWriter(vid_path, fourcc, 20.0, (vw, vh))
        for f in range(60):
            vframe = np.full((vh, vw, 3), (185, 195, 205), dtype=np.uint8)
            # Road
            cv2.fillPoly(vframe, [np.array([[0, vh], [240, 150], [400, 150], [vw, vh]])], (52, 54, 58))
            # Perspective lane lines
            y1 = int(160 + f * 3) % 190 + 150
            cv2.line(vframe, (vw // 2, y1), (vw // 2, min(vh, y1 + 35)), (240, 240, 240), 3)
            # Looming car ahead
            car_w = int(45 + f * 1.8)
            car_h = int(28 + f * 1.1)
            cx = vw // 2 + int(np.sin(f * 0.12) * 22)
            cy = int(175 + f * 2.1)
            cv2.rectangle(vframe, (cx - car_w // 2, cy - car_h // 2), (cx + car_w // 2, cy + car_h // 2), (180, 45, 45), -1)
            cv2.rectangle(vframe, (cx - car_w // 3, cy - car_h // 2 - 12), (cx + car_w // 3, cy - car_h // 2), (120, 145, 165), -1)
            cv2.putText(vframe, f"DASHCAM STREAM | FRAME {f:02d}/60", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            out_vid.write(vframe)
        out_vid.release()
    except Exception as e:
        print("Warning: could not write mp4:", e)

    print(f"Photorealistic benchmark media successfully generated in {output_dir}")

generate_photorealistic_benchmarks = generate_samples

if __name__ == "__main__":
    generate_samples(Path(__file__).resolve().parent)
