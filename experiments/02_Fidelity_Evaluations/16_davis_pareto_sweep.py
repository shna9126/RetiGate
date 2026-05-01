#!/usr/bin/env python3
"""
16_davis_pareto_sweep_v2.py

DAVIS 2017 Pareto Sweep — all sequences.
Metric: IoG-Recall (fraction of GT object area covered by ROI).
This is correct for a gating system — measures ROI coverage,
not downstream detection quality (YOLO removed intentionally).

For each threshold τ, reports:
  - Mean pixel sparsity across all frames
  - IoG-Recall: fraction of GT objects covered by ROI (IoG >= 0.5)
  - Per-sequence breakdown for diagnostic output
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from retigate import RetinaCore
from retigate.datasets.davis import DAVISDataset

# --- CONFIGURATION ---
DAVIS_ROOT = Path("data/davis/DAVIS")
IMAGE_DIR  = DAVIS_ROOT / "JPEGImages" / "480p"
ANNOT_DIR  = DAVIS_ROOT / "Annotations" / "480p"

IOG_THRESHOLD = 0.5   # fraction of GT object that must be in ROI


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def get_boxes_from_mask(mask_path):
    """
    Convert DAVIS segmentation mask to bounding boxes.
    Each unique non-zero pixel value = one object instance.
    Returns list of [x1, y1, x2, y2].
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    boxes = []
    for uid in np.unique(mask):
        if uid == 0:
            continue   # background
        ys, xs = np.where(mask == uid)
        if len(xs) == 0:
            continue
        boxes.append([
            int(np.min(xs)), int(np.min(ys)),
            int(np.max(xs)), int(np.max(ys))
        ])
    return boxes


def iog(gt_box, roi, threshold=IOG_THRESHOLD):
    """
    Intersection over GT Area.
    Measures what fraction of the GT object is inside the ROI.
    Returns True if coverage >= threshold.

    This is the correct metric for a gating/proposal system.
    Reference: Hosang et al., PAMI 2016.
    """
    if roi is None:
        return False

    ix1 = max(gt_box[0], roi[0])
    iy1 = max(gt_box[1], roi[1])
    ix2 = min(gt_box[2], roi[2])
    iy2 = min(gt_box[3], roi[3])

    inter   = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    if gt_area <= 0:
        return False

    return (inter / gt_area) >= threshold


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    all_sequences = DAVISDataset.list_sequences()
    print(f"DAVIS sequences found: {len(all_sequences)}")
    print(f"Metric: IoG-Recall (IoG >= {IOG_THRESHOLD})")
    print(f"No YOLO — pure gating coverage evaluation\n")

    thresholds   = [0.01, 0.05, 0.10, 0.20]
    final_results = []

    for tau in thresholds:
        print(f"\n{'='*55}")
        print(f"Evaluating τ = {tau}")
        print(f"{'='*55}")

        # Accumulators — micro-average across all objects
        total_tp       = 0
        total_gt_count = 0

        # Sparsity across all frames
        all_sparsity   = []

        # Per-sequence for diagnostic
        seq_records    = []

        for sequence in tqdm(all_sequences,
                             desc=f"τ={tau}",
                             leave=False):
            # Fresh retina per sequence
            retina           = RetinaCore.golden_baseline()
            retina.threshold = tau

            img_paths = sorted(
                list((IMAGE_DIR / sequence).glob("*.jpg"))
            )

            seq_tp       = 0
            seq_gt_count = 0
            seq_sparsity = []

            for img_p in img_paths:
                img = cv2.imread(str(img_p))
                if img is None:
                    continue

                mask_p = (ANNOT_DIR / sequence /
                          img_p.name.replace(".jpg", ".png"))
                gt_boxes = get_boxes_from_mask(mask_p)

                # Process through retina regardless
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rout = retina.process_frame(gray)
                roi  = retina.get_roi_bbox(
                    rout, frame_shape=img.shape
                )
                seq_sparsity.append(rout['sparsity'] * 100)

                if not gt_boxes:
                    continue   # no GT objects this frame

                # Count how many GT objects are covered by ROI
                for gt_box in gt_boxes:
                    seq_gt_count += 1
                    if iog(gt_box, roi):
                        seq_tp += 1

            # Sequence summary
            seq_recall = (seq_tp / seq_gt_count * 100
                          if seq_gt_count > 0 else 0.0)
            seq_records.append({
                'sequence':    sequence,
                'recall':      seq_recall,
                'sparsity':    np.mean(seq_sparsity),
                'tp':          seq_tp,
                'gt_count':    seq_gt_count,
            })

            total_tp       += seq_tp
            total_gt_count += seq_gt_count
            all_sparsity.extend(seq_sparsity)

        # Aggregate
        micro_recall    = (total_tp / total_gt_count * 100
                           if total_gt_count > 0 else 0.0)
        seq_df          = pd.DataFrame(seq_records)
        mean_seq_recall = seq_df['recall'].mean()
        std_seq_recall  = seq_df['recall'].std()

        # Worst sequences — useful for paper
        worst = seq_df.nsmallest(3, 'recall')

        print(f"\n  Micro-avg recall  : {micro_recall:.2f}%")
        print(f"  Mean seq recall   : "
              f"{mean_seq_recall:.2f}% ± {std_seq_recall:.2f}%")
        print(f"  Mean sparsity     : {np.mean(all_sparsity):.2f}%")
        print(f"  Total GT objects  : {total_gt_count}")
        print(f"  Worst sequences   :")
        for _, row in worst.iterrows():
            print(f"    {row['sequence']:<25}: "
                  f"{row['recall']:.1f}%")

        final_results.append({
            'Threshold':         tau,
            'Sparsity (%)':      np.mean(all_sparsity),
            'Sparsity_std':      np.std(all_sparsity),
            'Recall_micro (%)':  micro_recall,
            'Recall_mean (%)':   mean_seq_recall,
            'Recall_std (%)':    std_seq_recall,
            'N_sequences':       len(all_sequences),
            'N_frames':          len(all_sparsity),
            'N_gt_objects':      total_gt_count,
        })

        # Save per-sequence breakdown
        out_dir = Path('results/04_Ablations')
        out_dir.mkdir(parents=True, exist_ok=True)
        seq_df.to_csv(
            out_dir / f'davis_pareto_perseq_tau{tau}.csv',
            index=False
        )

    # ── Final table ──────────────────────────────────────────
    df = pd.DataFrame(final_results)

    print("\n\n" + "="*65)
    print("TABLE VI-B: DAVIS PARETO FRONTIER (All Sequences)")
    print("="*65)
    print(df[[
        'Threshold', 'Sparsity (%)', 'Recall_micro (%)',
        'Recall_mean (%)', 'Recall_std (%)', 'N_sequences'
    ]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ── Save ─────────────────────────────────────────────────
    df.to_csv(
        'results/04_Ablations/table6b_davis_pareto_v2.csv',
        index=False
    )
    print("\nSaved → results/04_Ablations/table6b_davis_pareto_v2.csv")

    # ── Paper summary ────────────────────────────────────────
    # Find best τ row
    best = df.loc[df['Recall_micro (%)'].idxmax()]

    print(f"""
{'='*65}
PAPER NUMBERS — DAVIS PARETO SWEEP
{'='*65}
Sequences : {len(all_sequences)} DAVIS 2017 480p sequences
Metric    : IoG-Recall (IoG >= {IOG_THRESHOLD}, Hosang et al. 2016)
Frames    : {df['N_frames'].iloc[0]} total annotated frames

τ      | Sparsity      | Recall (micro) | Recall (mean±std)
-------|---------------|----------------|------------------""")

    for _, row in df.iterrows():
        print(
            f"{row['Threshold']:.2f}   | "
            f"{row['Sparsity (%)']:.1f}%"
            f" ± {row['Sparsity_std']:.1f}%  | "
            f"{row['Recall_micro (%)']:.1f}%           | "
            f"{row['Recall_mean (%)']:.1f}%"
            f" ± {row['Recall_std (%)']:.1f}%"
        )

    print(f"""
Golden constant (τ=0.10):
  Sparsity : {df[df['Threshold']==0.10]['Sparsity (%)'].values[0]:.1f}%
  Recall   : {df[df['Threshold']==0.10]['Recall_micro (%)'].values[0]:.1f}%

Paper sentence:
"At τ=0.10, RetiGate achieves
{df[df['Threshold']==0.10]['Recall_micro (%)'].values[0]:.1f}% IoG-Recall
across all {len(all_sequences)} DAVIS sequences with
{df[df['Threshold']==0.10]['Sparsity (%)'].values[0]:.1f}% pixel sparsity,
without any dataset-specific retuning."
{'='*65}
""")


if __name__ == "__main__":
    main()