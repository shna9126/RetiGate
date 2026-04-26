import numpy as np
import pytest
from retigate.core import RetinaCore


def _random_frame(h=480, w=640):
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def test_golden_baseline_runs_on_random_frame():
    r = RetinaCore.golden_baseline()
    out = r.process_frame(_random_frame())
    required_keys = {'M_Motion', 'DS_Right', 'DS_Left', 'sparsity', 'active_mask'}
    assert required_keys == set(out.keys())
    assert 0.0 <= out['sparsity'] <= 1.0


def test_sparsity_increases_with_static_frames():
    r = RetinaCore.golden_baseline()
    frame = _random_frame()
    first_sparsity = r.process_frame(frame)['sparsity']
    for _ in range(18):
        r.process_frame(frame)
    last_sparsity = r.process_frame(frame)['sparsity']
    assert last_sparsity > first_sparsity, (
        f"Expected sparsity to grow on repeated static input: "
        f"frame1={first_sparsity:.4f}, frame20={last_sparsity:.4f}"
    )


def test_no_temporal_causes_collapse():
    # amacrine_decay=1.0 makes amacrine_state = m_bipolar each frame,
    # so inhibition = m_bipolar exactly → m_ganglion = max(0, 0) = 0
    r = RetinaCore.from_config(dict(
        amacrine_decay=1.0,
        global_weight=1.5,
        tail_len=15,
        shift_factor=0.5,
        threshold=0.05,
        use_global_inh=False,
        use_sac_tail=False,
    ))
    frame = _random_frame()
    for _ in range(5):
        out = r.process_frame(frame)
    assert np.allclose(out['M_Motion'], 0.0), (
        f"M_Motion should be all zeros; max was {out['M_Motion'].max()}"
    )


def test_sac_removal_zeros_directional_asymmetry():
    r = RetinaCore.from_config(dict(
        amacrine_decay=0.1,
        global_weight=1.5,
        tail_len=15,
        shift_factor=0.5,
        threshold=0.05,
        use_global_inh=True,
        use_sac_tail=False,
    ))
    out = r.process_frame(_random_frame())
    assert np.array_equal(out['DS_Right'], out['DS_Left']), \
        "DS_Right and DS_Left must be identical when use_sac_tail=False"


def test_roi_bbox_returns_none_on_empty_mask():
    r = RetinaCore.golden_baseline()
    frame = np.zeros((480, 640), dtype=np.uint8)
    out = r.process_frame(frame)
    result = r.get_roi_bbox(out, frame_shape=frame.shape)
    assert result is None, f"Expected None for empty mask, got {result}"
