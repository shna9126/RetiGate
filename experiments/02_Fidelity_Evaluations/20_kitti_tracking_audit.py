#!/usr/bin/env python3
"""
Fidelity Recall Audit — KITTI Tracking (21 sequences)
Metric: GT-Containment Recall (GT box inside ROI ± 5px margin)
        This is a valid proposal recall metric, stricter than IoG≥0.5.
        Reference: equivalent to IoG≥1.0 with 5px boundary tolerance.

Adds vs original:
  - Val/test split (sequences 0000-0006 val, 0007-0020 test)
  - IoG≥0.5 as secondary metric for reviewer comparison
  - Per-sequence CSV output
  - Paper-ready summary
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

LABEL_MAPPING = {
    "Car": 0, "Van": 0, "Truck": 0,
    "Pedestrian": 1, "Person_sitting": 1,
    "Cyclist": 2
}

TAU    = 0.10   # Golden constant — confirmed on val split
MARGIN = 5      # px boundary tolerance (same as original)

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def parse_tracking_labels(label_path):
    """
    KITTI Tracking Format:
    Col 0: Frame ID
    Col 2: Object Type
    Col 6-9: BBox (left, top, right, bottom)
    """
    try:
        df = pd.read_csv(label_path, sep=' ', header=None)
        df = df[[0, 2, 6, 7, 8, 9]]
        df.columns = ['frame', 'type', 'x1', 'y1', 'x2', 'y2']
        return df[df['type'].isin(LABEL_MAPPING.keys())]
    except Exception as e:
        print(f"  Error parsing {label_path}: {e}")
        return pd.DataFrame()


def gt_contained(gt_box, roi, margin=MARGIN):
    """
    Primary metric: is the GT box fully contained within the ROI?
    Equivalent to IoG≥1.0 with boundary tolerance.
    This is STRICTER than IoG≥0.5 — conservative estimate.

    gt_box: [x1, y1, x2, y2]
    roi:    (x1, y1, x2, y2) tuple or None
    """
    if roi is None:
        return False
    return (
        gt_box[0] >= roi[0] - margin and
        gt_box[1] >= roi[1] - margin and
        gt_box[2] <= roi[2] + margin and
        gt_box[3] <= roi[3] + margin
    )


def iog(gt_box, roi, threshold=0.5):
    """
    Secondary metric: Intersection over GT Area >= threshold.
    More lenient than containment — handles partial coverage.
    Used as cross-check for reviewers.

    Reference: Hosang et al., PAMI 2016,
    'What makes for effective detection proposals?'
    """
    if roi is None:
        return False

    ix1 = max(gt_box[0], roi[0])
    iy1 = max(gt_box[1], roi[1])
    ix2 = min(gt_box[2], roi[2])
    iy2 = min(gt_box[3], roi[3])

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    if gt_area <= 0:
        return False

    return (inter / gt_area) >= threshold


# -------------------------------------------------------------------
# CORE AUDIT
# -------------------------------------------------------------------

def run_split(seq_list, split_name, tau=TAU):
    """
    Run fidelity audit on a list of sequences.
    Reports both containment recall and IoG≥0.5 recall.
    Returns per-sequence results for CSV export.
    """
    records = []

    for seq in seq_list:
        label_file = LBL_DIR / f"{seq}.txt"
        if not label_file.exists():
            print(f"  WARNING: {label_file} not found, skipping.")
            continue

        # Fresh retina per sequence — clears all temporal buffers
        retina           = RetinaCore.golden_baseline()
        retina.threshold = tau

        gt_df     = parse_tracking_labels(label_file)
        img_paths = sorted(list((IMG_DIR / seq).glob("*.png")))

        seq_contain_tp = 0
        seq_iog_tp     = 0
        seq_total_gt   = 0

        for i, img_path in enumerate(
            tqdm(img_paths, desc=f"  {split_name} {seq}",
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
                seq_total_gt += 1

                if gt_contained(gt_box, roi):
                    seq_contain_tp += 1

                if iog(gt_box, roi, threshold=0.5):
                    seq_iog_tp += 1

        if seq_total_gt > 0:
            contain_recall = seq_contain_tp / seq_total_gt * 100
            iog_recall     = seq_iog_tp     / seq_total_gt * 100

            print(
                f"  {seq}: "
                f"Containment={contain_recall:.2f}%  "
                f"IoG≥0.5={iog_recall:.2f}%  "
                f"(GT={seq_total_gt})"
            )

            records.append({
                "split":           split_name,
                "sequence":        seq,
                "total_gt":        seq_total_gt,
                "contain_tp":      seq_contain_tp,
                "iog_tp":          seq_iog_tp,
                "contain_recall":  contain_recall,
                "iog_recall":      iog_recall,
            })

    # Summary
    if records:
        df = pd.DataFrame(records)
        c_mean = df["contain_recall"].mean()
        c_std  = df["contain_recall"].std()
        c_min  = df["contain_recall"].min()
        c_max  = df["contain_recall"].max()
        i_mean = df["iog_recall"].mean()
        i_std  = df["iog_recall"].std()

        print(f"\n  {split_name} SUMMARY (τ={tau}):")
        print(f"    Containment : {c_mean:.2f}% ± {c_std:.2f}%  "
              f"[{c_min:.1f}% – {c_max:.1f}%]")
        print(f"    IoG≥0.5     : {i_mean:.2f}% ± {i_std:.2f}%")
        return df

    return pd.DataFrame()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    sequences = sorted([
        d.name for d in IMG_DIR.iterdir() if d.is_dir()
    ])

    # Clean val/test split
    # Val  (0000–0006): used only to confirm τ=0.10
    # Test (0007–0020): final reported numbers
    val_seqs  = sequences[:7]
    test_seqs = sequences[7:]

    print("KITTI TRACKING FIDELITY AUDIT")
    print(f"Metric : GT-Containment Recall (±{MARGIN}px margin)")
    print(f"Cross  : IoG≥0.5 (secondary, for reviewer reference)")
    print(f"τ      : {TAU}")
    print(f"Val    : {val_seqs}")
    print(f"Test   : {test_seqs}")

    # --- VALIDATION SPLIT ---
    print("\n" + "="*60)
    print("VALIDATION SET (τ confirmation, not reported)")
    print("="*60)
    val_df = run_split(val_seqs, "VAL")

    # --- TEST SPLIT ---
    print("\n" + "="*60)
    print("TEST SET — FINAL REPORTED NUMBERS")
    print("="*60)
    test_df = run_split(test_seqs, "TEST")

    # --- FULL 21-SEQ AGGREGATE ---
    all_df = pd.concat([val_df, test_df], ignore_index=True)

    print("\n" + "="*60)
    print("FULL 21-SEQUENCE AGGREGATE (for paper)")
    print("="*60)
    print(
        f"  Containment Recall : "
        f"{all_df['contain_recall'].mean():.2f}% "
        f"± {all_df['contain_recall'].std():.2f}%"
    )
    print(
        f"  IoG≥0.5 Recall     : "
        f"{all_df['iog_recall'].mean():.2f}% "
        f"± {all_df['iog_recall'].std():.2f}%"
    )
    print(
        f"  Lowest sequence    : "
        f"{all_df['contain_recall'].min():.2f}% "
        f"(seq {all_df.loc[all_df['contain_recall'].idxmin(), 'sequence']})"
    )
    print(
        f"  Highest sequence   : "
        f"{all_df['contain_recall'].max():.2f}% "
        f"(seq {all_df.loc[all_df['contain_recall'].idxmax(), 'sequence']})"
    )

    # --- SAVE ---
    out_dir = Path("results/02_Accuracy_Logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fidelity_recall_v3.csv"
    all_df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    # --- PAPER SUMMARY ---
    test_c = test_df["contain_recall"]
    test_i = test_df["iog_recall"]

    print(f"""
{'='*60}
NUMBERS FOR PAPER
{'='*60}
Metric:   GT-Containment Recall (GT box inside ROI ±5px)
          Stricter than IoG≥0.5 → conservative lower bound.
τ:        {TAU} (selected on val sequences 0000–0006)
Dataset:  KITTI Tracking, {len(all_df)} sequences

Test split (0007–0020) — PRIMARY REPORTED NUMBER:
  Containment : {test_c.mean():.2f}% ± {test_c.std():.2f}%
  IoG≥0.5     : {test_i.mean():.2f}% ± {test_i.std():.2f}%

Full 21-seq aggregate:
  Containment : {all_df['contain_recall'].mean():.2f}% ± {all_df['contain_recall'].std():.2f}%
  IoG≥0.5     : {all_df['iog_recall'].mean():.2f}% ± {all_df['iog_recall'].std():.2f}%

Paper sentence:
"RetiGate achieves {test_c.mean():.1f}% gating fidelity recall
on held-out KITTI Tracking sequences (0007–0020), measured
as the fraction of ground-truth objects fully contained
within the predicted ROI (±5px boundary tolerance).
Cross-validated with IoG≥0.5: {test_i.mean():.1f}%."
{'='*60}
""")


if __name__ == "__main__":
    main()