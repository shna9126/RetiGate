# RetiGate — FINAL_NUMBERS.md
# Last cleaned: May 2026 (post Sprint 4)
# Rule: never use a number in the paper not in this file

---

## PARAMETERS (locked April 2026, VOS bug fixed)
  λ (amacrine_decay) = 0.10
  ω (global_weight)  = 1.50
  τ (threshold)      = 0.10
  tail               = 15 px
  shift              = 0.5 (shift_amount = 7 px)
  use_vos            = True
  use_global_inh     = True
  use_sac_tail       = True

---

## PRIMARY ACCURACY — KITTI Tracking (21 sequences)

  Dense YOLO11m mAP@0.5:    41.07%
  Sparse RetiGate mAP@0.5:  38.14%
  mAP Retention:             92.86%
  Pixel sparsity:            98.25%

  NOTE — detection count retention: 52.0%
  (lower than mAP retention because sparse inference
  eliminates false positives on background regions)

---

## FIDELITY RECALL — KITTI Tracking

  Metric: GT-Containment Recall (GT box inside ROI ±5px)

  τ grid search on VAL split (seq 0000–0006):
    τ=0.05: 90.87%  (90.23% ± 5.20%)
    τ=0.08: 96.10%  (96.45% ± 1.61%)
    τ=0.10: 97.24%  (97.17% ± 2.15%)  ← selected
    τ=0.12: 97.06%  (97.07% ± 2.72%)
    τ=0.15: 96.19%  (96.12% ± 4.11%)

  TEST split results (seq 0007–0020, τ=0.10):
    Micro-avg recall:  95.68%
    Per-seq mean:      94.39% ± 8.65%

  Low-recall sequences (stationary objects):
    Seq 0012: 74.7%
    Seq 0015: 73.5%

---

## ROI AREA — CORRECTED May 2026

  IMPORTANT: Two valid measurements, different scope.

  A) Active-frame median (Colab, seq 0006, n=270):
       Median ROI: 47.3%  (812×300px on 1242×375)
       Mean ROI:   48.2% ± 25.4%
       Range:      5.5% – 100.0%
     → Use for: efficiency tables, speedup claims

  B) Full test split (seq 0007–0020, n=6149 frames):
       Median ROI: 100.0%
       Mean ROI:    85.6%
       Active frames (ROI<99%): 45.1% (2772 frames)
     → Use for: operating regime table, honest framing

  C) DAVIS 2017 (90 sequences):
       Effective ROI: ~0-5%  (99.8% pixel sparsity)
       Active frames: 100%
     → Use for: primary efficiency demonstration

  RULE: never cite 47.3% without "on active frames"

---

## GPU EFFICIENCY — NVIDIA Tesla T4

  Source: Colab benchmark, seq 0006, τ=0.10
  Method: CUDA events, n=200, warmup=50

  ### YOLO11m (primary detector)
    Dense:   31.76 ± 0.97ms   (31.5 FPS)
    Sparse:  17.37 ± 0.53ms   (57.6 FPS)
    Speedup: 1.83× (conditional, active frames)
    
    Effective speedup (all KITTI test frames):
      = 0.451 × 1.83 + 0.549 × 1.0 = 1.37×

  ### Energy (YOLO11m, power via nvidia-smi 20Hz)
    Dense:   2107 mJ
    Sparse:  1294 mJ
    Saving:  −38.6%
    Power:   64.3W → 64.2W (~neutral)
    Variance: 10.6× more stable (sparse)

  ### Resolution Scaling (conditional on 47.3% ROI)
    KITTI: 31.3→19.7ms   1.59×  −37.1%
    1080p: 127.0→70.8ms  1.79×  −44.9%
    4K:    564.5→303.8ms 1.86×  −47.1%  (3.3 FPS sparse)

  NOTE: Previous locked number was 1.63× speedup
  (from earlier Colab run). New measurement is 1.83×
  on same hardware, same ROI size (~47%).
  Use 1.83× — it is from the cleaner Cell 6 measurement
  using pure CUDA events without sensing overhead included.

---

## DETECTOR AGNOSTICISM — T4 GPU, seq 0006, τ=0.10

  Median ROI: 47.3%

  YOLO11m (one-stage CNN):
    Dense mAP@50:  61.96%
    Sparse mAP@50: 71.81%
    Retention:     115.9%   ← foveal advantage
    Dense lat:     31.76ms
    Sparse lat:    17.37ms
    Speedup:       1.83×

  RT-DETR-L (transformer):
    Dense mAP@50:  78.61%
    Sparse mAP@50: 78.52%
    Retention:     99.9%
    Dense lat:     158.41 ± 7.86ms
    Sparse lat:     50.46 ± 8.08ms
    Speedup:       3.14×

  KEY INSIGHT: RT-DETR speedup (3.14×) > YOLO (1.83×)
  Reason: transformer attention scales quadratically
  with input tokens — spatial gating gives super-linear
  efficiency gains for transformer detectors.

---

## SENSING OVERHEAD — Apple M3 Pro (CPU-bound ops)

  NOTE: Sensing uses CPU-bound OpenCV ops.
  T4 total was 58ms (CPU underclocked on server).
  Mac M3 Pro is correct platform for this measurement.

  VOS (ORB+RANSAC):     7.93ms   35.7%
  DoG spatial filter:  11.88ms   53.4%
  Leaky IIR:            0.26ms    1.2%
  Global inhibition:    0.54ms    2.4%
  SAC tail:             0.90ms    4.1%
  ROI extraction:       0.75ms    3.3%
  ─────────────────────────────────────
  Total (full):        22.25ms  100.0%
  Total (no VOS):      14.32ms   (static camera)

  YOLO dense ref:      32.26ms
  Net saving:          ~10ms per active frame

---

## MOTION METHOD COMPARISON — Mac M3 Pro, 100 KITTI frames

  Method      Sparsity        Latency         Ego-robust
  RetiGate    97.1 ± 1.6%    25.1 ± 3.2ms    YES
  FrameDiff   50.2 ± 4.0%     0.6 ± 0.2ms    NO
  MOG2        73.3 ± 7.9%     2.3 ± 0.4ms    NO
  DISFlow      1.2 ± 1.8%    10.9 ± 0.6ms    NO
  Farneback    7.6 ± 2.8%    34.9 ± 1.3ms    NO
  RAFT         1.2 ± 2.1%   723.6 ± 33.3ms   NO (CPU)

---

## GENERALIZATION — No retuning, τ=0.10

  KITTI Tracking (21 seq):  Sparsity=98.25%  Recall=95.68%
  DAVIS 2017    (90 seq):   Sparsity=99.8%   IoG-Recall=93.9%
  Middlebury    (12 seq):   Sparsity=98.6%   Coverage=100%

  τ=0.20 on DAVIS: recall collapses to 22.6% (limitation)
  DAVIS frames: 6208 annotated objects, 13,556 total

---

## ABLATION — KITTI Tracking, 2000 frames, τ=0.10

  Variant          Sparsity    DSI     Note
  Baseline (full)  98.25%      0.046
  w/o Temporal     82.80%      0.001   −15.45pp (most critical)
  w/o Global inh   94.66%      0.032   −3.59pp
  w/o SAC tail     98.25%      0.000   sole directionality source
  w/o DoG filter   99.20%      0.315   artefactual DSI (see paper)
  w/o VOS          98.20%      0.046   ≈baseline (KITTI limitation)

---

## GATING STRATEGY — KITTI test split (seq 0007–0020)

  Single box:
    Recall:    95.44%
    ROI area:  88.9%
    Savings:   11.1%
    Note: high recall because ROI covers ~89% of frame

  Multi-cluster (top-5, max_area_frac=0.15):
    Recall:    58.28%
    ROI area:  44.3%
    Savings:   55.7%
    Note: tight clusters miss objects outside active pixels
    Failure mode: cluster_too_small (98.9% of misses)

---

## OPERATING REGIME — KITTI test split

  Full test split (n=6149 frames):
    Median ROI:      100.0%
    Mean ROI:         85.6%
    Active frames:    45.1%

  Per scene type:
    Scene           Seqs  Median ROI  Active%
    Static camera   90    ~0-5%       100%    (DAVIS)
    Slow ego-motion  4    ~32%         81%    (0006,0012,0015,0020)
    Fast ego-motion 10    ~100%        33%    (highway)
    All KITTI test  14    100%         45%

  Effective speedup formula:
    eff = active_frac × conditional_speedup
          + (1-active_frac) × 1.0
    YOLO11m: 0.451 × 1.83 + 0.549 × 1.0 = 1.37×

---

## IIR MEMORY — Synthetic simulation (KITTI resolution)

  Object stops at frame 30, measured to <10% ROI area:
    λ=0.05:  22 frames retained  (2.2s at 10FPS)
    λ=0.10:  10 frames retained  (1.0s at 10FPS) ← chosen
    λ=0.20:   5 frames retained  (0.5s at 10FPS)
    λ=0.50:   1 frame  retained  (0.1s at 10FPS)

---

## TEMPORAL STABILITY — Active seqs (0006,0012,0015,0020)

  Metric: frame-to-frame IoU of consecutive ROIs

    λ=0.05:  IoU=87.0%  Centroid Δ=52.3px
    λ=0.10:  IoU=90.4%  Centroid Δ=39.2px  ← chosen
    λ=0.20:  IoU=94.7%  Centroid Δ=22.4px
    λ=0.50:  IoU=96.4%  Centroid Δ=12.7px

---

## NUMBERS THAT CHANGED IN SPRINT 4

  Old → New (reason):
  
  Speedup: 1.61× → 1.83×
    Old was from earlier Colab cell including sensing.
    New is pure CUDA-event inference speedup.
    Use 1.83× everywhere.

  ROI area: 47.9% → 47.3%
    Negligible difference. Use 47.3% (more recent).

  Effective speedup: NEW number = 1.37×
    Accounts for 45.1% active frame rate on KITTI test.
    Always pair with "on active frames: 1.83×".

  τ in golden_baseline(): 0.05 (code default)
    All experiments override to τ=0.10.
    Paper always states τ=0.10.
    Code default is irrelevant to paper.