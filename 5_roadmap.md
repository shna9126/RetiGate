

```
Figure 1: Teaser (existing)           ← keep
Figure 2: Bio analogy (existing)      ← keep
Figure 3: Pipeline stages (existing)  ← keep
Figure 4: Qualitative results         ← fix ROI %
Figure 5: Pareto + ablation           ← keep
Figure 6: Accuracy-efficiency Pareto  ← NEW (Week 3)
Figure 7: Temporal stability          ← NEW (Week 3)
Figure 8: IIR memory analysis         ← NEW (Week 2)
Figure 9: Latency breakdown pie       ← NEW (Week 3)
```

**You will exceed 8 pages. Strategy:**
```
Move Figures 7-9 to Supplementary Material.
Keep Figure 6 (Pareto curve) in main paper —
it directly addresses the fatal weakness.

Summary: main paper stays 8 pages,
supplementary adds 2 pages of supporting evidence.
WACV allows supplementary material.
```

---

### New Tables Plan (Final Paper)

```
Table 1: Parameters (existing)        ← keep
Table 2: τ grid search (existing)     ← keep
Table 3: System accuracy              ← remove mAP@0.75 col
Table 4: Motion comparison (existing) ← keep
Table 5: Efficiency on T4 (existing)  ← keep
Table 6: Resolution scaling (existing)← keep
Table 7: Cross-domain (existing)      ← keep
Table 8: Ablation (existing)          ← keep
Table 9: Multi-cluster vs single box  ← NEW (Week 1)
Table 10: Fallback analysis           ← NEW (Week 1)
Table 11: Detector agnosticism        ← NEW (Week 2)
Table 12: Edge hardware (Jetson)      ← NEW (Week 2)
```

**You will exceed 8 pages. Strategy:**
```
Merge Tables 11 and 12 into one compact table.
Move Table 10 (fallback) inline as text + key numbers.
Keep Tables 9, 11+12 in main paper.
Move Table 10 to supplementary.
```

---

## Complete Sprint 4 Deliverables

```
EXPERIMENTS (in priority order):
  Week 1, Day 1-2: 23_multicluster_vs_singlebox.py done
  Week 1, Day 3-4: 24_fallback_analysis.py done
  Week 1, Day 5:   26_iir_memory_analysis.py done
  Week 2, Day 1-2: 27_jetson_benchmark.py ignore
  Week 2, Day 3-4: 28_detector_agnosticism.py done
  Week 3, Day 1-2: 25_pareto_curve.py redo/done
  Week 3, Day 3-4: 29_temporal_stability.py
  Week 3, Day 5:   30_latency_breakdown.py

PAPER CHANGES (in priority order):
  Fix "pifno" sentence immediately
  Fix Table 3 mAP@0.75 column
  Add design philosophy paragraph (3.1)
  Add dataset choice paragraph (4.1)
  Add detector agnosticism paragraph (4.X)
  Add IIR memory defense paragraph (3.5)
  Add Jetson numbers to Table 5 or new table
  Add multi-cluster Table 9
  Add Figure 6 (Pareto curve)
  Regenerate Figure 4 with correct ROI %
  Fix τ formatting throughout
  Fix "golden baseline" formatting

TEXT ADDITIONS (no experiments needed):
  VOS latency breakdown note in Section 3.3
  Temporal stability note in Section 3.5
  Classical CV defense in Section 3.1
  Dataset age justification in Section 4.1
```

---

## Revised Score Projection

```
Current paper (before Sprint 4):  5/10 borderline
After Week 1 fixes only:          6/10 weak accept
After Week 2 additions:           6.5/10 weak accept
After Week 3 + Week 4 full:       7/10 accept

The single biggest score jump comes from:
  1. Multi-cluster experiment (−1 fatal weakness)
  2. Pareto curve with learned baselines (−1 fatal)
  3. Jetson benchmark (transforms "T4 only" critique)

The rest is polish that builds reviewer confidence.
```

Start here → reason

1. sec/0_abstract.tex   
   Short. Sets the tone. Numbers cascade from here.

2. sec/4_experiments.tex  
   Longest. Most changes. Get this right first.

3. sec/3_method.tex       
   Add IIR memory + temporal stability sentences.
   Minor changes only.

4. sec/5_limitations.tex  
   Rewrite stationary object paragraph.

5. sec/6_conclusion.tex   
   Update numbers. Add operating regime sentence.

6. sec/1_intro.tex        
   Update contributions list with correct numbers.

7. sec/2_related.tex      
   Likely no changes needed.