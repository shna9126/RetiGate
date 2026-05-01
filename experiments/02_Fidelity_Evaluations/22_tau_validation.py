#!/usr/bin/env python3
"""
22_tau_validation.py

Validates τ=0.10 selection on held-out val split.
Val:  sequences 0000–0006 (used ONLY for τ selection)
Test: sequences 0007–0020 (final reported numbers)

Metric: GT-Containment Recall (GT box inside ROI ± 5px)
Same metric as 20_kitti_tracking_audit_v3.py
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from retigate import RetinaCore

# --- CONFIGURATION ---
DATA_ROOT = Path("data/kitti/data_tracking_image")
IMG_DIR   = DATA_ROOT / "image_02"
LBL_DIR   = DATA_ROOT / "label_02"
MARGIN    = 5  # px boundary tolerance

LABEL_MAPPING = {
    "Car": 0, "Van": 0, "Truck": 0,
    "Pedestrian": 1, "Person_sitting": 1,
    "Cyclist": 2
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def parse_labels(label_path):
    try:
        df = pd.read_csv(label_path, sep=' ', header=None)
        df = df[[0, 2, 6, 7, 8, 9]]
        df.columns = ['frame', 'type', 'x1', 'y1', 'x2', 'y2']
        return df[df['type'].isin(LABEL_MAPPING.keys())]
    except:
        return pd.DataFrame()


def gt_contained(gt_box, roi, margin=MARGIN):
    """GT box fully inside ROI with margin tolerance."""
    if roi is None:
        return False
    return (
        gt_box[0] >= roi[0] - margin and
        gt_box[1] >= roi[1] - margin and
        gt_box[2] <= roi[2] + margin and
        gt_box[3] <= roi[3] + margin
    )


def run_sequences(seq_list, tau):
    """
    Run fidelity audit on given sequences with given tau.
    Returns micro-average recall and per-sequence recalls.
    """
    all_tp      = 0
    all_fn      = 0
    seq_recalls = []

    for seq in seq_list:
        label_file = LBL_DIR / f"{seq}.txt"
        if not label_file.exists():
            continue

        retina           = RetinaCore.golden_baseline()
        retina.threshold = tau

        gt_df     = parse_labels(label_file)
        img_paths = sorted(list((IMG_DIR / seq).glob("*.png")))

        seq_tp = 0
        seq_fn = 0

        for i, img_path in enumerate(
            tqdm(img_paths,
                 desc=f"  seq={seq} τ={tau}",
                 leave=False)
        ):
            frame_gt = gt_df[gt_df['frame'] == i]
            if frame_gt.empty:
                continue

            img  = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rout = retina.process_frame(gray)
            roi  = retina.get_roi_bbox(
                rout, frame_shape=img.shape
            )

            for _, gt in frame_gt.iterrows():
                gt_box = [gt.x1, gt.y1, gt.x2, gt.y2]
                if gt_contained(gt_box, roi):
                    seq_tp += 1
                else:
                    seq_fn += 1

        total = seq_tp + seq_fn
        if total > 0:
            seq_recalls.append(seq_tp / total * 100)

        all_tp += seq_tp
        all_fn += seq_fn

    micro = (all_tp / (all_tp + all_fn) * 100
             if (all_tp + all_fn) > 0 else 0.0)

    return micro, seq_recalls


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    sequences = sorted([
        d.name for d in IMG_DIR.iterdir()
        if d.is_dir()
    ])

    val_seqs  = sequences[:7]   # 0000–0006
    test_seqs = sequences[7:]   # 0007–0020

    thresholds = [0.05, 0.08, 0.10, 0.12, 0.15]

    print("KITTI TRACKING — τ VALIDATION")
    print(f"Val sequences:  {val_seqs}")
    print(f"Test sequences: {test_seqs}")
    print(f"Metric: GT-Containment Recall (±{MARGIN}px)")
    print()

    # ── PHASE 1: Grid search on VAL split only ───────────────
    print("=" * 55)
    print("PHASE 1: τ GRID SEARCH ON VAL SPLIT (0000–0006)")
    print("=" * 55)

    val_results = []

    for tau in thresholds:
        print(f"\nτ = {tau}")
        micro, seq_recalls = run_sequences(val_seqs, tau)

        val_results.append({
            'tau':        tau,
            'micro_recall': micro,
            'mean_recall':  np.mean(seq_recalls),
            'std_recall':   np.std(seq_recalls),
        })

        print(f"  Micro-avg recall: {micro:.2f}%")
        print(f"  Mean per-seq:     "
              f"{np.mean(seq_recalls):.2f}%"
              f" ± {np.std(seq_recalls):.2f}%")

    val_df = pd.DataFrame(val_results)

    print("\n" + "=" * 55)
    print("VAL RESULTS SUMMARY")
    print("=" * 55)
    print(val_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    ))

    # Select best τ on val
    best_row  = val_df.loc[val_df['micro_recall'].idxmax()]
    best_tau  = best_row['tau']

    print(f"\nBest τ on val split: {best_tau}")
    print(f"Val recall at best τ: {best_row['micro_recall']:.2f}%")

    # Confirm τ=0.10 is selected (or note if different)
    if best_tau == 0.10:
        print("✅ τ=0.10 confirmed as optimal on val split.")
    else:
        print(f"⚠️  Val selects τ={best_tau}, not 0.10.")
        print(f"   Check if difference is significant.")
        print(f"   Val recall at τ=0.10: "
              f"{val_df[val_df['tau']==0.10]['micro_recall'].values[0]:.2f}%")
        print(f"   Val recall at τ={best_tau}: "
              f"{best_row['micro_recall']:.2f}%")

    # ── PHASE 2: Final evaluation on TEST split ──────────────
    print("\n" + "=" * 55)
    print(f"PHASE 2: FINAL TEST EVALUATION (τ={best_tau})")
    print("Sequences 0007–0020 — never seen during τ selection")
    print("=" * 55)

    test_micro, test_seq_recalls = run_sequences(
        test_seqs, best_tau
    )

    print(f"\nTest micro-avg recall : {test_micro:.2f}%")
    print(f"Test mean per-seq     : "
          f"{np.mean(test_seq_recalls):.2f}%"
          f" ± {np.std(test_seq_recalls):.2f}%")

    # ── Also run test at τ=0.10 if best_tau differs ─────────
    if best_tau != 0.10:
        print(f"\nAlso running test at τ=0.10 for comparison...")
        test_micro_010, test_recalls_010 = run_sequences(
            test_seqs, 0.10
        )
        print(f"Test recall at τ=0.10: {test_micro_010:.2f}%")

    # ── Save ─────────────────────────────────────────────────
    out_dir = Path("results/02_Accuracy_Logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    val_df.to_csv(
        out_dir / "tau_validation_val_split.csv",
        index=False
    )

    pd.DataFrame({
        'split':    'test',
        'tau':      best_tau,
        'sequence': test_seqs[:len(test_seq_recalls)],
        'recall':   test_seq_recalls,
    }).to_csv(
        out_dir / "tau_validation_test_split.csv",
        index=False
    )

    # ── Paper summary ────────────────────────────────────────
    print(f"""
{'='*60}
PAPER NUMBERS — τ VALIDATION
{'='*60}
Val split  (0000–0006): τ grid search
Test split (0007–0020): final reported numbers

Val grid search results:
{val_df.to_string(index=False, float_format=lambda x: f'{x:.2f}')}

Selected τ: {best_tau} (highest micro-recall on val)

Test results (τ={best_tau}):
  Micro-avg recall : {test_micro:.2f}%
  Mean per-seq     : {np.mean(test_seq_recalls):.2f}% ± {np.std(test_seq_recalls):.2f}%

Paper sentence:
"τ was selected by grid search over
{{0.05, 0.08, 0.10, 0.12, 0.15}} on a held-out
validation split (sequences 0000–0006).
τ={best_tau} achieved {best_row['micro_recall']:.1f}% recall on val.
Final fidelity recall on the test split
(sequences 0007–0020): {test_micro:.1f}%."
{'='*60}
""")


if __name__ == "__main__":
    main()