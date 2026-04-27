🛠️ Phase 1: Architectural Consistency (The VOR Fix)
The Problem: VOR (Vestibulo-Ocular Reflex) is a major "Contribution," but it’s currently a "hack" in the ablation script instead of a formal part of the RetinaCore package.

Task 1.1: Move the ORB-Homography stabilization logic into retigate/core/retina.py.

Task 1.2: Update RetinaCore.golden_baseline() to accept a use_vor=True flag.

Task 1.3: Rename the feature in the paper to "Vestibulo-Ocular Stabilization (VOS)"—it sounds more technical and aligns with the biological analogy.

📊 Phase 2: Rigorous mAP Alignment (The "Dense" Baseline)
The Problem: We reported 50.23% mAP, but we didn't show what "Normal YOLO" gets on the same KITTI Tracking frames. This makes the 50% look "lonely."

Task 2.1: Modify 14_kitti_map_audit.py to run two passes:

pass_dense: Standard YOLO on the full 1242x375 frame.

pass_sparse: RetiGate + YOLO on the ROI crop.

Task 2.2: Calculate the "Accuracy Retention" metric:

Retention= 
mAP 
dense
/
mAP 
sparse
​
If dense is 55% and sparse is 50%, you have 90% accuracy retention with 95% power saving—that is a massive win.
Run the dual-pass mAP experiment on KITTI Tracking this week. Get the dense vs sparse mAP numbers. The retention ratio tells you your story.

If retention is above 90%: Lead with accuracy. Aim higher than BMVC.
If retention is 75-90%: WACV is the right target. Honest story.
If retention is below 75%: BMVC with clear limitations framing.

⚡ Phase 3: The "Honest" Efficiency Narrative
The Problem: On an M3 Pro CPU, Python overhead makes RetiGate slower than dense YOLO. We must stop claiming "Wall-clock Speedup" for the single-threaded version.

Task 3.1: Update 08_power_measurement.py and 13_final_synthesis.py to distinguish between Theoretical GFLOPs (ASIC-target) and Empirical Latency (Python-reference).

Task 3.2: Double down on Async Pipelining. Since 18_async.txt shows the effective latency is max(sensing, inference), we use that as our primary speed claim.

Task 3.3: Remove the "5.2x" and "94% Energy" claims from the summary. Replace with: "94.91% Theoretical GFLOPs reduction; Empirical Python overhead: 114 mJ/frame."

🎯 Phase 4: Multi-Object Robustness (The "SAHI" Challenge)
The Problem: A single ROI misses scattered objects. The reviewer noticed the "Recall Cliff" in the SAHI showdown.

Task 4.1: Standardize RetinaCore.get_roi_clusters(top_n=3).

Task 4.2: Re-run the SAHI comparison using Top-3 ROIs. This will fix the 9% recall frames while still being much faster than SAHI's 20-30 slices.

🧹 Phase 5: Result Synthesis v4.0 (The Final Clean-up)
The Problem: Exp 04 (Scene Flow) is generating "junk" data because the labels don't match.

Task 5.1: Delete experiments/archive/04_detection_mAP.py. It is misleading.

Task 5.2: Use Experiment 14 (KITTI Tracking) as your only accuracy claim. It is the most rigorous and the dataset is cleaner.

What would give this CVPR/ICCV-level punch? I'll be direct. One of these would do it:

A formal proof or tight empirical analysis showing why the biological constants (λ=0.1, ω=1.5, α=0.5) are optimal — not just "we swept them," but a theory of why the retina's specific parameters generalize.
A demonstration that RetiGate as a pre-filter actually improves mAP over dense YOLO on small distant objects specifically — if the retention ratio on map_small is favorable, that is the headline.
A comparison showing that RetiGate outperforms SAHI not just on latency but on mAP for moving objects specifically — static objects SAHI handles fine; moving objects at distance are where motion-adaptive cropping should shine.

None of these require new architecture. They require sharper analysis of what you already have.


BMVC prep
Necessary (paper cannot go in without this)
1. Dense baseline mAP on KITTI Tracking — 1 day
Right now your 50.23% mAP is "lonely" as you said. Every reviewer will ask: what does plain YOLO get on the same sequences? Without this number you cannot claim accuracy retention. This is the single most important remaining experiment.
Run 14_kitti_map_audit.py with both dense and sparse pass. Get the paired numbers. This is one script modification and one overnight compute run.
2. VOR into core architecture — 3 hours
Move vor_stabilize into retigate/core/retina.py with a use_vor=True flag. Update golden_baseline(). Update the ablation script to use it. This is a code hygiene fix that makes the architecture coherent. Without it, a reviewer who reads the code sees VOR as a hack.
3. Freeze the honest efficiency narrative — 2 hours
Update 13_final_synthesis.py to distinguish theoretical GFLOPs from empirical powermetrics. Produce one clean summary table. Remove all 5.2x and 94% energy claims from any document that will become paper material.
Nice-to-have but not blocking submission
Multi-cluster SAHI re-run — improves the paper but does not change the core claim. Can go in Future Work.
4K scaling — interesting but a single-frame result. Not robust enough for a claim.
Additional failure mode visualizations — you already have them from 09_failure_cases.py. Enough.