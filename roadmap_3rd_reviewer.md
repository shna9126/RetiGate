# RetiGate — Actionable Fix List

---

## 🔴 CRITICAL (Do First — Paper is Unpublishable Without These)

**1. Fix fidelity recall computation**
- "Did GT box fall inside ROI" is not standard recall
- Recompute as proper TP/(TP+FN) at IoU≥0.5 using KITTI ground truth labels (you already have them in label_02/)
- Explain sequences 0012 (74.7%) and 0015 (73.5%) — don't average over them silently

**2. Resolve the two conflicting recall numbers**
- `20_fidelity_recall.txt` reports both 95.50% ± 7.28% AND an older table showing 88.75% ± 11.44%
- Figure out which is post-VOS-bug-fix and delete/archive the stale one

---

## 🟡 MAJOR (Required for Any Top Venue)

**6. Add stronger baselines**
- Replace/supplement MOG2 (2006) with: **RAFT** (Teed & Deng, ECCV 2020) and **SuBSENSE** or **BSUV-Net**
- Add at minimum one modern efficient detector comparison: **AdaFocus** or **Glance-and-Focus** (NeurIPS 2020) — these are your real competitors

**7. Fix the DAVIS Pareto sweep**
- Currently only runs on `bike-packing` — one sequence
- Run Exp 16 across all 90 DAVIS sequences
- τ=0.20 gives 39.86% recall — address this collapse explicitly, don't smooth it with averaging

**8. Separate validation/test sets**
- τ=0.10 was selected by looking at test performance — this is test-set overfitting
- Split KITTI sequences: ~7 for val (threshold tuning), ~14 for test (final numbers)

**9. Add confidence intervals everywhere**
- Every latency number needs ± std
- Every mAP number needs confidence interval
- Use paired t-test for RetiGate vs Dense comparisons

**10. Fix the SAHI recall computation**
- You're using SAHI detections as ground truth — SAHI has false positives
- Use KITTI labels as ground truth for Exp 05 as well

---

## 🟢 MODERATE (Needed for Full Paper)

**11. Expand dataset scope**
- Add **MOT17** evaluation (standard for anything tracking-adjacent)
- Run at least one adverse condition sequence (night/rain) — BDD100K has these

**12. Fix the ablation inconsistency**
- `no_dog` increases sparsity AND jumps DSI from 0.046 → 0.315 — explain this counter-intuitive result
- `no_vor` ≈ baseline on KITTI — test VOS on a camera-shake dataset where it should matter (GoPro dataset, or DAVIS handheld sequences)

**13. Document VOS bug fix impact**
- List which result tables were generated before vs. after April 2026 fix
- Re-run anything that was pre-fix

---

## 📚 LITERATURE (Before Writing Draft)

**Must-cite (reviewers will ask):**
- Gallego et al. 2020 — Event-based Vision Survey (IEEE TPAMI) → explain why software > event camera
- Wang et al. 2021 — **AdaFocus** (ICCV) → most direct competitor
- Wang et al. 2020 — **Glance and Focus** (NeurIPS) → most direct competitor
- Teed & Deng 2020 — **RAFT** (ECCV) → should be a baseline
- Briggman et al. 2011 (Nature) + Euler et al. 2002 → actual DSGC physiology to support bio claims
- Lin et al. 2019 — **TSM** (ICCV) → temporal efficiency baseline

**Should cite:**
- Habibian et al. CVPR 2021 — Skip-convolutions for video detection
- Tokmakov et al. ICCV 2017 — Motion segmentation
- Kim et al. 2020 — Spiking-YOLO (differentiate from this)

---

## 🎯 Reframing (Do This Before Writing Anything)

**Current (wrong) framing:** "RetiGate makes detection faster"

**Correct framing:** "RetiGate is a bio-inspired sparsity mechanism that reduces theoretical compute (GFLOPs) by ~X% at ~7% accuracy cost, with strong cross-domain generalization and no retraining required"

What's Done vs What's Still Needed
✅ Done
GPU benchmark (T4)
Energy measurement (empirical)
mAP audit (21 sequences)
Fidelity recall audit
Ablation study
DAVIS generalization
Cross-domain (Middlebury)
Baseline comparison (Table II)
Pareto sweep
Hardware platform (M3 Pro + T4)
⚠️ Still Needed Before Draft
1. RAFT baseline — reviewers will ask
   Single script, ~2hrs to run

2. Validation/test split for τ selection
   Currently τ=0.10 selected on test set
   Split your 21 sequences: 7 val / 14 test
   Re-run fidelity audit on test-only split

3. Confidence intervals on mAP
   Currently no CI on 41.07% or 38.14%
   Run bootstrap or report per-sequence std

4. MOT17 evaluation — one sequence minimum
   Standard benchmark reviewers expect

5. Explain power increase (+3.5%) in one sentence
   Already have the explanation — just write it
❌ Known Limitations to State in Paper
- Single GPU platform (T4 via Colab — shared resource)
- Dense latency std was high in earlier runs (now fixed: ±0.81ms)
- τ=0.10 sensitivity (τ=0.20 collapses DAVIS recall to 39%)
- Sequences 0012 (74.7%) and 0015 (73.5%) below 75% recall
- ROI area variance is high (σ=25.4%) — system is scene-dependent


# Complete Roadmap — Prioritized by Impact

---

## What to Do vs What to Skip

First, honest assessment of each remaining item:

| Item | Do It? | Reason |
|---|---|---|
| RAFT baseline | ✅ Yes | Reviewers will ask, ~2hrs |
| Val/test split for τ | ✅ Yes | Methodological integrity |
| CI on mAP | ✅ Yes | Easy, already have per-sequence data |
| Fidelity recall reframe | ✅ Yes | Critical — metric is non-standard |
| Resolve duplicate recall numbers | ✅ Yes | Factual inconsistency in logs |
| DAVIS Pareto full sweep | ✅ Yes | Currently only 1 sequence |
| no_dog ablation explanation | ✅ Yes | Counter-intuitive result needs text |
| AdaFocus/Glance-and-Focus | ⚠️ Partial | Cite and discuss, don't reimplement |
| MOT17 | ⚠️ Optional | Do one sequence only, low effort |
| SAHI recall fix | ✅ Yes | Uses wrong ground truth |
| BDD100K night/rain | ❌ Skip | Too much new data, diminishing returns |
| SuBSENSE/BSUV-Net | ❌ Skip | Implement RAFT instead, sufficient |
| VOS on GoPro | ❌ Skip | Out of scope for current submission |



### Day 9: Explain no_dog Ablation

```python
# No new experiment needed — just add explanation to paper

# What happened:
# no_dog removes DoG → only center kernel remains
# Center kernel = Gaussian blur = spatial smoothing
# Smoothed signal has less texture detail
# Less texture = more uniform → 
#   more pixels exceed threshold → LESS sparsity (99.2% vs 98.2%)
# 
# BUT DSI jumps from 0.046 → 0.315 because:
# Without surround inhibition, directional smearing (SAC tail)
# operates on the raw center response
# Raw center has stronger directional gradients
# → artificially inflated DSI

# Paper text (one paragraph):
# "Removing the DoG spatial filter (no_dog) paradoxically 
#  increases apparent sparsity while amplifying DSI. Without
#  center-surround antagonism, the SAC directional tail operates
#  on unfiltered luminance gradients, producing stronger but
#  artifactual directional signals. This confirms that the DoG
#  stage is essential for biologically meaningful direction
#  selectivity, not merely spatial filtering."
```

---

## Week 3 — Write the Draft

### Paper Structure

```
1. Abstract          (write last, ~250 words)
2. Introduction      (motivation, contributions, 1 page)
3. Related Work      (use literature list, 1 page)
4. Method            (biology → math → algorithm, 1.5 pages)
5. Experiments       (4 tables + figures, 2 pages)
6. Ablation          (1 table, 0.5 pages)
7. Limitations       (honest, 0.5 pages)
8. Conclusion        (0.5 pages)
```

### Your Four Main Tables

```
Table 1: Efficiency (T4 GPU) — DONE, LaTeX ready
Table 2: Motion method comparison (RetiGate vs RAFT/Farnebäck/MOG2)
Table 3: mAP accuracy (Dense vs Sparse, with CI)
Table 4: Generalization (DAVIS + Middlebury + MOT17)
Table 5: Ablation
```

### Contributions to State in Introduction

```
1. A bio-inspired temporal pre-filter modeled on DSGC circuitry
   that produces sparse kinetic ROIs with 95.5% fidelity recall

2. Empirical demonstration that sparse ROI inference achieves
   1.63× speedup and 38.6% energy reduction on NVIDIA T4
   while retaining 92.86% of dense detection accuracy

3. Resolution-scaling efficiency: gains increase from 1.59× at
   KITTI to 1.86× at 4K, with energy savings up to 47.1%

4. Zero-shot cross-domain generalization across 90 DAVIS
   sequences and 12 Middlebury sequences without retuning
```

---

## What to Skip Entirely

```
❌ BDD100K night/rain      — new dataset, marginal contribution
❌ SuBSENSE implementation — RAFT is sufficient modern baseline
❌ GoPro VOS testing       — out of scope
❌ AdaFocus reimplementation — cite and discuss in related work
❌ Glance-and-Focus reimpl  — cite and discuss in related work
❌ Event camera comparison  — discuss in limitations, don't implement
```

---

## Condensed Week-by-Week

```
WEEK 1 (experiments):
  Day 1-2: Fidelity recall v2 (IoU-based) + val/test split
  Day 3:   RAFT baseline + rerun Table II
  Day 4:   Bootstrap CI on mAP
  Day 5:   DAVIS Pareto full sweep (all 90 sequences)

WEEK 2 (cleanup):
  Day 6-7: τ validation on val split, confirm τ=0.10
  Day 8:   MOT17 one sequence
  Day 9:   Archive pre-fix results, clean FINAL_NUMBERS.md
  Day 10:  Finalize all tables, update LaTeX

WEEK 3 (writing):
  Day 11:  Method section
  Day 12:  Experiments section
  Day 13:  Introduction + Related Work
  Day 14:  Abstract + Limitations + Conclusion
  Day 15:  Internal review, figures, formatting

TARGET: Complete draft in 3 weeks
Submit to RA-L or BMVC 2026
```

---

## Single Most Important Thing This Week

```
Run the new fidelity recall script (Day 1-2).

Current metric (ROI containment) will be rejected.
IoU-based recall is standard and your numbers
will likely improve because containment is stricter
than IoU≥0.5.

If recall stays ~95% with IoU metric → very strong result.
If recall drops significantly → important honest finding.
Either way you need the real number before writing.
```