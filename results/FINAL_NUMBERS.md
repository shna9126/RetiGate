RETIGATE — FINAL PAPER NUMBERS (locked)

PRIMARY RESULTS (KITTI Tracking, 21 sequences, 2000+ frames)
├── Dense YOLO11m mAP@0.5:        41.07%
├── RetiGate+YOLO11m mAP@0.5:     38.14%
├── Accuracy Retention:            92.86%
├── Mask Sparsity:                 98.25%
└── Fidelity Recall:               95.50% ± 7.28%

EFFICIENCY
├── Sensing latency (RetiGate):    22.97 ms
├── Sensing latency (Farnebäck):   ~35 ms
├── Sensing speedup:               2.15x over dense optical flow
├── Async effective latency:       67.88 ms vs 69.93 ms dense
└── Pixel sparsity:                98.25%

GENERALIZATION (no retuning, same golden_baseline)
├── DAVIS coverage (90 sequences): ~97% mean
├── Middlebury coverage (12/12):   100%
└── DAVIS recall at τ=0.10:        99.28%

ABLATION (KITTI Tracking, 2000 frames)
├── no_temporal sparsity drop:     98.25% → 82.80% (-15.45 points)
├── no_global sparsity drop:       98.25% → 94.66% (-3.59 points)
├── no_sac DSI:                    0.000 (proves sole directionality source)
└── no_vor ≈ baseline:             honest limitation on this dataset

ARCHITECTURE
└── VOS bug fixed April 2026
    Parameters locked: λ=0.1, ω=1.5, τ=0.05, tail=15, shift=0.5