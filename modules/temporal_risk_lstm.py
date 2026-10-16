"""
Temporal Risk Prediction Module using CNN + LSTM with rolling frame sequences.
Analyzes continuous video streams and webcam feeds for trajectory dynamics,
looming motion (Time-to-Collision proxy), and pre-crash risk evolution over time.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Dict, List, Any, Tuple
from config import SEVERITY_LEVELS, SEVERITY_THRESHOLDS

class SpatioTemporalCNNLSTM(nn.Module):
    """
    Lightweight 2D-CNN feature encoder coupled with a 2-layer LSTM sequence model.
    Processes rolling temporal windows of road-scene frames.
    """
    def __init__(self, cnn_feat_dim: int = 48, telemetry_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        # Fast spatial feature extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(32 * 4, cnn_feat_dim),
            nn.ReLU()
        )
        # LSTM sequence model
        self.lstm = nn.LSTM(
            input_size=cnn_feat_dim + telemetry_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        # Output risk regressor
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, frames_tensor: torch.Tensor, telemetry_tensor: torch.Tensor) -> torch.Tensor:
        # frames_tensor: (1, seq_len, 3, H, W)
        # telemetry_tensor: (1, seq_len, 4)
        B, T, C, H, W = frames_tensor.shape
        flat_frames = frames_tensor.view(B * T, C, H, W)
        feats = self.encoder(flat_frames).view(B, T, -1)
        
        lstm_in = torch.cat([feats, telemetry_tensor], dim=2)
        lstm_out, _ = self.lstm(lstm_in)
        # Prediction at final temporal step
        risk = self.head(lstm_out[:, -1, :])
        return risk


class RollingSequenceRiskPipeline:
    """
    Maintains a rolling sequence of frames, tracks motion dynamics,
    and runs CNN + LSTM inference on continuous video / webcam streams.
    """
    def __init__(self, window_size: int = 10, speed_limit_kmh: float = 60.0):
        self.window_size = window_size
        self.speed_limit = speed_limit_kmh
        self.frame_buffer = deque(maxlen=window_size)
        self.telemetry_buffer = deque(maxlen=window_size)
        self.prev_gray = None

        self.model = SpatioTemporalCNNLSTM()
        self.model.eval()

        # Seed with initial state
        self._init_weights()

    def _init_weights(self):
        torch.manual_seed(42)
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def reset(self):
        self.frame_buffer.clear()
        self.telemetry_buffer.clear()
        self.prev_gray = None

    def process_frame(self, frame_bgr: np.ndarray, simulated_speed: float = 60.0,
                      rider_count: int = 1, helmet_detected: bool = True) -> Dict[str, Any]:
        """
        Ingests a new video/webcam frame into the rolling sequence,
        computes spatio-temporal features, optical flow motion, and returns risk analysis.
        """
        h, w = frame_bgr.shape[:2]
        annotated = frame_bgr.copy()

        # 1. Motion & looming rate estimation via frame differencing / optical flow
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        motion_rate = 0.0
        motion_bbox = None

        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            diff = cv2.absdiff(self.prev_gray, gray)
            _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_pixels = np.sum(diff_thresh > 0)
            motion_rate = float(motion_pixels) / float(h * w)

            # Find bounding box of primary moving entity
            cnts, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sig_cnts = [c for c in cnts if cv2.contourArea(c) > (w * h * 0.01)]
            if sig_cnts:
                largest = max(sig_cnts, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(largest)
                motion_bbox = [x, y, x + bw, y + bh]
                # Draw motion tracker box
                box_color = (0, 165, 255) if motion_rate < 0.15 else (0, 0, 255)
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), box_color, 2)
                cv2.putText(annotated, f"TRACKING TARGET [Motion {motion_rate:.1%}]",
                            (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        self.prev_gray = gray.copy()

        # 2. Preprocess frame for CNN (112x112 RGB normalized)
        resized_rgb = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (112, 112))
        norm_frame = resized_rgb.astype(np.float32) / 255.0
        norm_frame = np.transpose(norm_frame, (2, 0, 1))  # (3, 112, 112)

        # 3. Telemetry vector: [norm_speed, speed_excess, rider_count_norm, helmet_flag]
        norm_speed = min(1.0, simulated_speed / 140.0)
        speed_excess = max(0.0, (simulated_speed - self.speed_limit) / 50.0)
        rider_norm = min(1.0, (rider_count - 1) / 3.0)
        helmet_val = 0.0 if helmet_detected else 1.0

        telemetry_vec = [norm_speed, speed_excess, rider_norm, helmet_val]

        # Add to rolling sequence
        self.frame_buffer.append(norm_frame)
        self.telemetry_buffer.append(telemetry_vec)

        # Pad sequence if buffer not yet full
        while len(self.frame_buffer) < self.window_size:
            self.frame_buffer.append(norm_frame)
            self.telemetry_buffer.append(telemetry_vec)

        # 4. CNN + LSTM Temporal Sequence Inference
        frames_seq = torch.from_numpy(np.array(list(self.frame_buffer))).unsqueeze(0)  # (1, T, 3, 112, 112)
        telemetry_seq = torch.from_numpy(np.array(list(self.telemetry_buffer), dtype=np.float32)).unsqueeze(0)  # (1, T, 4)

        with torch.no_grad():
            raw_model_score = self.model(frames_seq, telemetry_seq).item()

        # 5. Domain-guided risk integration: combine LSTM sequence dynamics with physical rules
        # Motion factor: rapid looming toward camera elevates collision threat
        motion_factor = min(1.0, motion_rate * 4.0)

        # Violation penalties – collected individually for probabilistic compounding
        individual_penalties = []
        warnings = []

        if simulated_speed > self.speed_limit:
            excess = simulated_speed - self.speed_limit
            excess_ratio = min(1.0, excess / 45.0)
            individual_penalties.append(0.20 + (excess_ratio * 0.35))
            warnings.append(f"Overspeeding: {simulated_speed:.1f} km/h (Limit: {self.speed_limit:.0f} km/h)")

        if not helmet_detected:
            individual_penalties.append(0.40)
            warnings.append("Helmet Violation: Rider / Pillion without helmet")

        if rider_count >= 3:
            overload = rider_count - 2
            individual_penalties.append(0.40 + min(0.30, overload * 0.15))
            warnings.append(f"Triple Riding Violation: {rider_count} occupants")

        if motion_rate > 0.15:
            warnings.append("Rapid Forward Looming / Near-Miss Proximity Hazard")

        # Probabilistic compounding: 1 - prod(1 - p_i)
        # Each violation independently reduces the "safety budget".
        # Avoids the artificial 70-75% ceiling from plain addition.
        if individual_penalties:
            safety = 1.0
            for p in individual_penalties:
                safety *= (1.0 - min(0.95, p))
            violation_score = float(1.0 - safety)
        else:
            violation_score = 0.0

        # Composite temporal risk score
        # LSTM temporal state (0.35) + compounded violations (0.35) + motion looming (0.30)
        risk_score = 0.35 * raw_model_score + 0.35 * violation_score + 0.30 * motion_factor
        risk_score = round(min(1.0, max(0.05, risk_score)), 3)

        # Assign Severity Tier
        if risk_score <= SEVERITY_THRESHOLDS["LOW_MAX"]:
            severity_tier = "LOW"
            severity_label = "Low Risk"
            severity_color = "#10B981"
            warning_status = "SAFE: Traffic dynamics normal."
        elif risk_score <= SEVERITY_THRESHOLDS["MID_MAX"]:
            severity_tier = "MID"
            severity_label = "Moderate Risk"
            severity_color = "#F59E0B"
            warning_status = "CAUTION: Elevated risk detected. Maintain safe headway."
        else:
            severity_tier = "HIGH"
            severity_label = "Critical Risk"
            severity_color = "#EF4444"
            warning_status = "ALERT: Imminent collision / severe violation hazard!"

        if not warnings:
            warnings.append("No active violations. Trajectory stable.")

        # Draw HUD on top of annotated frame
        cv2.rectangle(annotated, (10, 10), (380, 125), (20, 20, 20), -1)
        cv2.rectangle(annotated, (10, 10), (380, 125), (100, 100, 100), 1)

        cv2.putText(annotated, "CNN+LSTM LIVE TEMPORAL MONITOR", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.putText(annotated, f"Risk Score: {risk_score:.3f} | Tier: {severity_tier}", (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 100) if severity_tier == "LOW" else ((0, 180, 255) if severity_tier == "MID" else (0, 0, 255)), 2)
        cv2.putText(annotated, f"Simulated Speed: {simulated_speed:.1f} km/h", (20, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(annotated, f"Sequence Buffer: {len(self.frame_buffer)}/{self.window_size} frames", (20, 106),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 200, 240), 1)

        return {
            "risk_score": risk_score,
            "severity_tier": severity_tier,
            "severity_label": severity_label,
            "severity_color": severity_color,
            "motion_rate": round(motion_rate * 100, 1),
            "warnings": warnings,
            "warning_status": warning_status,
            "annotated_frame": annotated,
            "motion_bbox": motion_bbox
        }
