import pytest
import numpy as np
import cv2
import torch

from modules.temporal_risk_lstm import RollingSequenceRiskPipeline, SpatioTemporalCNNLSTM

class TestTemporalRiskLSTM:
    def test_spatio_temporal_cnn_lstm_architecture(self):
        model = SpatioTemporalCNNLSTM(cnn_feat_dim=48, telemetry_dim=4, hidden_dim=64)
        model.eval()

        # Batch 1, Sequence 5, Channels 3, Height 112, Width 112
        frames = torch.randn(1, 5, 3, 112, 112)
        telemetry = torch.randn(1, 5, 4)

        with torch.no_grad():
            output = model(frames, telemetry)

        assert output.shape == (1, 1)
        assert 0.0 <= output.item() <= 1.0

    def test_pipeline_initialization(self):
        pipeline = RollingSequenceRiskPipeline(window_size=10, speed_limit_kmh=60.0)
        assert pipeline.window_size == 10
        assert pipeline.speed_limit == 60.0
        assert len(pipeline.frame_buffer) == 0
        assert len(pipeline.telemetry_buffer) == 0
        assert pipeline.prev_gray is None

    def test_pipeline_process_frame(self):
        pipeline = RollingSequenceRiskPipeline(window_size=8, speed_limit_kmh=60.0)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        res = pipeline.process_frame(frame, simulated_speed=50.0, rider_count=1, helmet_detected=True)
        assert "risk_score" in res
        assert "severity_tier" in res
        assert "annotated_frame" in res
        assert "motion_rate" in res
        assert 0.0 <= res["risk_score"] <= 1.0
        assert res["severity_tier"] in ["LOW", "MID", "HIGH"]
        assert len(pipeline.frame_buffer) == 8  # Padded to window_size

    def test_pipeline_sequence_motion(self):
        pipeline = RollingSequenceRiskPipeline(window_size=5, speed_limit_kmh=60.0)

        # Feed frame 1 (black)
        f1 = np.zeros((240, 320, 3), dtype=np.uint8)
        res1 = pipeline.process_frame(f1, simulated_speed=50.0)
        assert res1["motion_rate"] == 0.0

        # Feed frame 2 (with moving white box)
        f2 = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(f2, (100, 80), (220, 180), (255, 255, 255), -1)
        res2 = pipeline.process_frame(f2, simulated_speed=50.0)
        assert res2["motion_rate"] > 0.0
        assert res2["motion_bbox"] is not None

    def test_pipeline_high_risk_conditions(self):
        pipeline = RollingSequenceRiskPipeline(window_size=5, speed_limit_kmh=60.0)
        f1 = np.zeros((240, 320, 3), dtype=np.uint8)
        # Baseline frame
        pipeline.process_frame(f1, simulated_speed=110.0, rider_count=3, helmet_detected=False)

        # Subsequent frame with rapid optical looming motion
        f2 = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(f2, (50, 40), (270, 200), (255, 255, 255), -1)
        res = pipeline.process_frame(f2, simulated_speed=110.0, rider_count=3, helmet_detected=False)

        assert res["risk_score"] > 0.60
        assert any("Overspeeding" in w for w in res["warnings"])
        assert any("Helmet Violation" in w for w in res["warnings"])
        assert any("Triple Riding" in w for w in res["warnings"])

    def test_pipeline_reset(self):
        pipeline = RollingSequenceRiskPipeline(window_size=5, speed_limit_kmh=60.0)
        f = np.zeros((240, 320, 3), dtype=np.uint8)
        pipeline.process_frame(f)
        assert len(pipeline.frame_buffer) > 0
        assert pipeline.prev_gray is not None

        pipeline.reset()
        assert len(pipeline.frame_buffer) == 0
        assert len(pipeline.telemetry_buffer) == 0
        assert pipeline.prev_gray is None
