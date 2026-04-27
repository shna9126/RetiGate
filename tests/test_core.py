import numpy as np
import pytest
from retigate.core.retina import RetinaCore


def _random_frame(h=480, w=640):
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def test_golden_baseline_runs_on_random_frame():
    r = RetinaCore.golden_baseline()
    out = r.process_frame(_random_frame())
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (480, 640)


def test_sparsity_increases_with_static_frames():
    r = RetinaCore.golden_baseline()
    frame = _random_frame()
    first_out = r.process_frame(frame)
    for _ in range(18):
        r.process_frame(frame)
    last_out = r.process_frame(frame)
    # On static frames, output should remain zero (no motion)
    assert np.allclose(last_out, 0.0), (
        f"Expected output to stay zero on static input: "
        f"frame1={np.mean(first_out):.4f}, frame20={np.mean(last_out):.4f}"
    )


def test_no_temporal_causes_collapse():
    # alpha=1.0 makes leaky_integrator = thresh each frame,
    # but since thresh is absdiff, on static frames it should be zero
    r = RetinaCore(alpha=1.0, tau=0.1, use_vos=False)
    frame = _random_frame()
    for _ in range(5):
        out = r.process_frame(frame)
    assert np.allclose(out, 0.0), (
        f"Output should be all zeros on static frames; max was {out.max()}"
    )


def test_roi_bbox_returns_none_on_empty_mask():
    r = RetinaCore.golden_baseline()
    frame = np.zeros((480, 640), dtype=np.uint8)
    out = r.process_frame(frame)
    result = r.get_roi_bbox(out, frame_shape=frame.shape)
    assert result is None, f"Expected None for empty mask, got {result}"


def test_roi_bbox_applies_padding():
    r = RetinaCore(alpha=0.9, tau=0.01, use_vos=False)  # Lower tau to detect motion
    # Create a frame with some motion in the center
    frame1 = np.zeros((100, 100), dtype=np.uint8)
    frame2 = np.zeros((100, 100), dtype=np.uint8)
    frame2[40:60, 50:70] = 100  # Add motion
    r.process_frame(frame1)  # Initialize
    out = r.process_frame(frame2)
    
    no_pad = r.get_roi_bbox(out, frame_shape=(100, 100), padding=0)
    with_pad = r.get_roi_bbox(out, frame_shape=(100, 100), padding=10)
    
    assert no_pad is not None, "ROI should be detected"
    assert with_pad is not None, "ROI should be detected with padding"
    
    # With padding, bbox should be larger or at bounds
    assert with_pad[0] <= no_pad[0] or with_pad[0] == 0
    assert with_pad[1] <= no_pad[1] or with_pad[1] == 0
    assert with_pad[2] >= no_pad[2] or with_pad[2] == 100
    assert with_pad[3] >= no_pad[3] or with_pad[3] == 100
