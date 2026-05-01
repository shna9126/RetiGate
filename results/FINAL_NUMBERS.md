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

## GPU BENCHMARK (NVIDIA Tesla T4, April 2026)
## Three independent runs — values are stable

Hardware:  Tesla T4 15GB VRAM
Timing:    CUDA events, n=200 runs, warmup=50
Power:     nvidia-smi 20Hz, trimmed ±10%, n=300 sustained runs

### KITTI Canonical (1242×375 → ROI 812×300, 47.9% of frame)

| Metric         | Dense          | Sparse         | Delta          |
|----------------|----------------|----------------|----------------|
| Median latency | 32.74 ms       | 20.15 ms       | 1.63× faster   |
| Latency std    | 3.94 ms        | 0.37 ms        | 10.6× stable   |
| P95 latency    | 37.06 ms       | 20.78 ms       | —              |
| Throughput     | 30.5 FPS       | 49.6 FPS       | +19.1 FPS      |
| Avg power      | 64.3 W         | 64.2 W         | ~neutral       |
| Energy/frame   | 2107 mJ        | 1294 mJ        | −38.6%         |

### Resolution Scaling

| Resolution | Dense   | Sparse  | Speedup | E-save | S-FPS |
|------------|---------|---------|---------|--------|-------|
| KITTI      | 31.3 ms | 19.7 ms | 1.59×   | 37.1%  | 50.7  |
| 1080p      | 127.0ms | 70.8 ms | 1.79×   | 44.9%  | 14.1  |
| 4K         | 564.5ms | 303.8ms | 1.86×   | 47.1%  | 3.3   |

ROI Statistics (100 KITTI frames):
  Median: 812×300 px (47.9% of frame)
  Mean:   755×291 px (48.2% ± 25.4%)
  Range:  5.5% – 100.0%



# RetiGate — FINAL PAPER NUMBERS (locked May 2026)

---

## FIDELITY RECALL — KITTI TRACKING

Metric:   GT-Containment Recall (GT box inside ROI ±5px)
Dataset:  KITTI Tracking, 21 sequences
τ:        0.10 (selected by grid search on val split)

### τ Validation (Val split: sequences 0000–0006)
| τ    | Micro-recall | Mean±std          |
|------|-------------|-------------------|
| 0.05 | 90.87%      | 90.23% ± 5.20%    |
| 0.08 | 96.10%      | 96.45% ± 1.61%    |
| 0.10 | 97.24%      | 97.17% ± 2.15%    |  ← selected
| 0.12 | 97.06%      | 97.07% ± 2.72%    |
| 0.15 | 96.19%      | 96.12% ± 4.11%    |

### Final Test Results (Test split: sequences 0007–0020)
  Micro-avg recall : 95.68%
  Mean per-seq     : 94.39% ± 8.65%

---

## SYSTEM ACCURACY — KITTI TRACKING (21 sequences)
  Dense mAP@50:  41.07%
  Sparse mAP@50: 38.14%
  Retention:     92.86%

---

## GPU EFFICIENCY — NVIDIA Tesla T4
  Dense:   32.26 ± 0.81ms  (31.0 FPS)
  Sparse:  20.13 ± 0.61ms  (49.8 FPS)
  Speedup: 1.61×
  Energy:  2107mJ → 1294mJ (−38.6%)
  Power:   64.3W → 64.2W   (~neutral)

  Resolution scaling:
    KITTI: 1.59× | 1080p: 1.79× | 4K: 1.86×

---

## GENERALIZATION — DAVIS 2017 (90 sequences)
  τ=0.10: Sparsity=99.8%, IoG-Recall=93.9%
  τ=0.05: Sparsity=98.6%, IoG-Recall=93.3%
  τ=0.20: collapses to 22.6% (noted as limitation)
  Frames: 6208 annotated | No retuning

## GENERALIZATION — MIDDLEBURY (12 sequences)
  Coverage: 100% (all 12 sequences)

---

## MOTION COMPARISON — TABLE II
  RetiGate:  97.1% sparsity, 25.1ms, ego-robust=YES
  FrameDiff: 50.2% sparsity,  0.6ms, ego-robust=NO
  MOG2:      73.3% sparsity,  2.3ms, ego-robust=NO
  DISFlow:    1.2% sparsity, 10.9ms, ego-robust=NO
  Farneback:  7.6% sparsity, 34.9ms, ego-robust=NO
  RAFT:       1.2% sparsity, 724ms,  ego-robust=NO (CPU)

---

## ABLATION — KITTI TRACKING (2000 frames)
  baseline:    98.25% sparsity, DSI=0.046
  no_temporal: 82.80% sparsity (−15.45 pts)
  no_global:   94.66% sparsity (−3.59 pts)
  no_sac:      98.25% sparsity, DSI=0.000
  no_dog:      99.20% sparsity, DSI=0.315 (see paper)
  no_vor:      98.20% sparsity (≈baseline on KITTI)

---

## PARAMETERS (locked April 2026)
  λ=0.10, ω=1.5, τ=0.10, tail=15, shift=0.5
  τ confirmed by grid search on val split 0000–0006