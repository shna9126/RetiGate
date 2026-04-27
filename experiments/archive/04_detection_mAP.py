#!/usr/bin/env python3
"""
Experiment 04: Detection mAP — Dense YOLO11n vs RetiGate Multi-Cluster Foveation

Dense pipeline:
    YOLO11n on the full frame at imgsz=(384,1280).

Sparse (Gated) pipeline — Phase 2 multi-cluster foveation:
    1. Temporal priming: reset retina, feed _11 frame, then process _10.
    2. Cluster-based slicing: top-3 largest motion clusters from get_roi_clusters.
    3. Foveal inference loop: for each cluster crop (x1,y1,x2,y2) →
         - resize to 640×640 with INTER_CUBIC
         - run YOLO11n on the zoomed patch
         - map back:  global_x = pred_x * (crop_w / 640) + x1
                      global_y = pred_y * (crop_h / 640) + y1
    4. Global aggregation: merge dense + foveal predictions, NMS @ IoU=0.5.

GT filter: Easy + Moderate only (truncation ≤ 0.3, occlusion ≤ 1).
Classes:   Car (COCO 2) and Pedestrian (COCO 0) only.

Output:
    results/table4_detection_mAP.csv
    results/table4_per_frame.csv

Usage:
    python experiments/04_detection_mAP.py --n 100
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms as tv_nms
from tqdm import tqdm
from ultralytics import YOLO

from retigate import RetinaCore
from retigate.datasets.kitti import KITTIDataset
from retigate.metrics import DetectionEvaluator

YOLO_FILTER      = [0, 2]        # COCO: person=0, car=2
YOLO_IMGSZ_DENSE = (384, 1280)   # 12×32, 40×32 — KITTI native aspect
FOVEA_SIZE       = 640           # each cluster crop is resized to this
FOVEA_PAD        = 15            # TIGHTER padding for HIGHER zoom magnification
MAX_CLUSTERS     = 3             # top-3 largest motion clusters per frame
NMS_IOU          = 0.5

# VOR (Vestibulo-Ocular Reflex) stabilization
VOR_WARMUP_ITERS = 10            # feed stabilized _11 this many times
_ORB     = cv2.ORB_create(500)
_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def yolo_preds_to_eval(boxes_obj):
    """(xyxy, scores, coco_labels) filtered to YOLO_FILTER classes."""
    if boxes_obj is None or len(boxes_obj) == 0:
        return (torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros(0, dtype=torch.float32),
                torch.zeros(0, dtype=torch.long))
    xyxy     = boxes_obj.xyxy.cpu().float()
    scores   = boxes_obj.conf.cpu().float()
    coco_cls = boxes_obj.cls.cpu().long()
    valid = torch.tensor([int(c) in YOLO_FILTER for c in coco_cls],
                         dtype=torch.bool)
    return xyxy[valid], scores[valid], coco_cls[valid]


def vor_stabilize(gray_ref: np.ndarray, gray_warp: np.ndarray) -> np.ndarray:
    """
    Warp gray_warp to align with gray_ref using ORB homography (VOR).
    Cancels ego-motion so only independently moving objects produce
    differential signal in the retina.
    Falls back to gray_warp unchanged if homography cannot be estimated.
    """
    kp1, des1 = _ORB.detectAndCompute(gray_ref,  None)
    kp2, des2 = _ORB.detectAndCompute(gray_warp, None)
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return gray_warp
    matches = _MATCHER.match(des1, des2)
    if len(matches) < 4:
        return gray_warp
    matches = sorted(matches, key=lambda m: m.distance)[:50]
    pts_ref  = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_warp = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask  = cv2.findHomography(pts_warp, pts_ref, cv2.RANSAC, 5.0)
    if H is None:
        return gray_warp
    h, w = gray_ref.shape
    return cv2.warpPerspective(gray_warp, H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


# ── pipelines ─────────────────────────────────────────────────────────────────

def run_dense(model, img_bgr):
    t0  = time.perf_counter()
    res = model(img_bgr, classes=YOLO_FILTER, verbose=False,
                imgsz=YOLO_IMGSZ_DENSE)[0]
    ms = (time.perf_counter() - t0) * 1000
    return (*yolo_preds_to_eval(res.boxes), ms)


def run_foveal_clusters(model, img_bgr, retina, gray):
    """
    Foveal inference on top-3 motion clusters.
    Returns (cluster_boxes, cluster_scores, cluster_labels,
             latency_ms, sparsity, clusters, n_clusters).
    cluster_boxes are already in full-image coordinates.
    """
    t0       = time.perf_counter()
    rout     = retina.process_frame(gray)
    sparsity = rout['sparsity']
    clusters = retina.get_roi_clusters(rout, pad=FOVEA_PAD, frame_shape=img_bgr.shape)

    empty = torch.zeros((0, 4), dtype=torch.float32)

    if not clusters:
        ms = (time.perf_counter() - t0) * 1000
        return (empty, torch.zeros(0), torch.zeros(0, dtype=torch.long),
                ms, sparsity, [], 0)

    # Top-3 by area
    clusters = sorted(clusters,
                      key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                      reverse=True)[:MAX_CLUSTERS]

    all_boxes, all_scores, all_labels = [], [], []

    for (x1, y1, x2, y2) in clusters:
        crop_h = y2 - y1;  crop_w = x2 - x1
        if crop_h <= 0 or crop_w <= 0:
            continue

        # ── High-res crop ──────────────────────────────────────────────
        crop   = img_bgr[y1:y2, x1:x2]
        zoomed = cv2.resize(crop, (FOVEA_SIZE, FOVEA_SIZE),
                            interpolation=cv2.INTER_CUBIC)

        # ── Inference on zoomed patch ──────────────────────────────────
        res = model(zoomed, classes=YOLO_FILTER, verbose=False,
                    imgsz=FOVEA_SIZE)[0]
        boxes, scores, labels = yolo_preds_to_eval(res.boxes)

        if boxes.shape[0] > 0:
            # ── Coordinate transform back to global frame ──────────────
            # global_x = pred_x * (crop_w / FOVEA_SIZE) + x1
            # global_y = pred_y * (crop_h / FOVEA_SIZE) + y1
            sx = crop_w / FOVEA_SIZE
            sy = crop_h / FOVEA_SIZE
            boxes[:, 0] = boxes[:, 0] * sx + x1
            boxes[:, 1] = boxes[:, 1] * sy + y1
            boxes[:, 2] = boxes[:, 2] * sx + x1
            boxes[:, 3] = boxes[:, 3] * sy + y1
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

    ms = (time.perf_counter() - t0) * 1000
    n  = len(clusters)

    if not all_boxes:
        return (empty, torch.zeros(0), torch.zeros(0, dtype=torch.long),
                ms, sparsity, clusters, n)

    cat_b = torch.cat(all_boxes)
    cat_s = torch.cat(all_scores)
    cat_l = torch.cat(all_labels)

    # NMS across foveal zones to remove duplicates
    keep = tv_nms(cat_b, cat_s, iou_threshold=NMS_IOU)
    return (cat_b[keep], cat_s[keep], cat_l[keep], ms, sparsity, clusters, n)


# ── main ──────────────────────────────────────────────────────────────────────

def main(n_frames: int = 100):
    Path('results').mkdir(exist_ok=True)

    print("Loading KITTI dataset...")
    ds = KITTIDataset()
    if len(ds) == 0:
        sys.exit("ERROR: KITTIDataset is empty — check data/kitti/ layout.")

    image_paths = ds.image_paths[:n_frames]
    print(f"  {len(image_paths)} frames selected")

    print("Loading YOLO11n...")
    model  = YOLO('yolo11n.pt')
    retina = RetinaCore.golden_baseline()

    eval_dense  = DetectionEvaluator()
    eval_sparse = DetectionEvaluator()
    per_frame   = []

    # --- WACV TUNED PARAMETERS ---
    CONF_DENSE  = 0.15  # Lowered to capture all candidates for the union
    CONF_FOVEAL = 0.25  # Lowered to allow zoomed small objects to pass

    for img_path in tqdm(image_paths, desc='Evaluating'):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. VOR Temporal priming
        retina.reset_memory()
        img_11 = cv2.imread(str(img_path).replace('_10.png', '_11.png'))
        if img_11 is not None:
            gray_11 = cv2.cvtColor(img_11, cv2.COLOR_BGR2GRAY)
            gray_11_stab = vor_stabilize(gray, gray_11)
            for _ in range(VOR_WARMUP_ITERS):
                retina.process_frame(gray_11_stab)

        # 2. Load Ground Truth
        gt_np, gc_np = ds._load_label(img_path.stem)
        gt_boxes  = torch.from_numpy(gt_np).float()
        gt_labels = torch.from_numpy(gc_np).long()

        # 3. Dense Pipeline (imgsz=1280, conf=0.20)
        d_boxes, d_scores, d_labels, d_lat = run_dense(model, img_bgr)
        # Apply dense confidence gate
        mask_d = d_scores >= CONF_DENSE
        d_boxes, d_scores, d_labels = d_boxes[mask_d], d_scores[mask_d], d_labels[mask_d]

        # 4. Foveal Cluster Pipeline
        c_boxes, c_scores, c_labels, c_lat, sparsity, clusters, n_clusters = \
            run_foveal_clusters(model, img_bgr, retina, gray)
        # Apply foveal confidence gate
        mask_c = c_scores >= CONF_FOVEAL
        c_boxes, c_scores, c_labels = c_boxes[mask_c], c_scores[mask_c], c_labels[mask_c]

        # 5. RECALL-FIRST CONSENSUS AGGREGATION
        # If we have foveal detections, we prioritize them over dense ones
        if c_boxes.shape[0] > 0:
            final_b = [c_boxes]
            final_s = [c_scores]
            final_l = [c_labels]
            
            if d_boxes.shape[0] > 0:
                from torchvision.ops import box_iou
                ious = box_iou(d_boxes, c_boxes)
                max_iou, _ = ious.max(dim=1)
                # Keep dense detections only if they don't overlap with our foveal "fixations"
                keep_dense = max_iou < 0.3 
                final_b.append(d_boxes[keep_dense])
                final_s.append(d_scores[keep_dense])
                final_l.append(d_labels[keep_dense])
            
            s_boxes = torch.cat(final_b)
            s_scores = torch.cat(final_s)
            s_labels = torch.cat(final_l)
        else:
            s_boxes, s_scores, s_labels = d_boxes, d_scores, d_labels

        # Final cleanup NMS
        if s_boxes.shape[0] > 0:
            keep = tv_nms(s_boxes, s_scores, iou_threshold=0.4)
            s_boxes, s_scores, s_labels = s_boxes[keep], s_scores[keep], s_labels[keep]

        # 6. Update Metrics
        if gt_boxes.shape[0] > 0:
            eval_dense.update(d_boxes, d_scores, d_labels, gt_boxes, gt_labels)
            eval_sparse.update(s_boxes, s_scores, s_labels, gt_boxes, gt_labels)

        # 7. Per-frame log
        if clusters:
            rx1 = min(c[0] for c in clusters); ry1 = min(c[1] for c in clusters)
            rx2 = max(c[2] for c in clusters); ry2 = max(c[3] for c in clusters)
        else:
            rx1 = ry1 = rx2 = ry2 = None

        per_frame.append({
            'frame':          img_path.stem,
            'n_gt':           gt_boxes.shape[0],
            'n_dense_pred':   d_boxes.shape[0],
            'n_foveal_pred':  c_boxes.shape[0],
            'n_combined_pred': s_boxes.shape[0],
            'n_clusters':     n_clusters,
            'sparsity_pct':   round(float(sparsity) * 100, 2),
            'dense_lat_ms':   round(d_lat, 2),
            'foveal_lat_ms':  round(c_lat, 2),
            'total_sparse_lat_ms': round(d_lat + c_lat, 2),
            'roi_x1': rx1, 'roi_y1': ry1, 'roi_x2': rx2, 'roi_y2': ry2,
        })

        # ── VOR audit (first 25 frames) ──────────────────────────────────
        if n_frames <= 25:
            gt_centroids_y = ((gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
                              if gt_boxes.shape[0] > 0 else torch.tensor([]))
            covered = 0
            for (cx1, cy1, cx2, cy2) in clusters:
                for cy in gt_centroids_y:
                    if cy1 <= float(cy) <= cy2:
                        covered += 1
                        break
            cluster_ys = [(c[1], c[3]) for c in clusters]
            print(f"  {img_path.stem}  clusters={cluster_ys}"
                  f"  gt_n={gt_boxes.shape[0]}  covered={covered}")

    # --- END OF LOOP ---
    print("\nComputing final mAP...")
    dm = eval_dense.compute()
    sm = eval_sparse.compute()

    df_pf = pd.DataFrame(per_frame)
    if df_pf.empty:
        sys.exit("ERROR: No data processed. check per_frame logic.")

    # Logging and saving logic...
    # (The rest of your script for printing tables and saving CSVs is correct)

    # ── Aggregate metrics ──────────────────────────────────────────────────
    print("\nComputing mAP…")
    dm = eval_dense.compute()
    sm = eval_sparse.compute()

    df_pf         = pd.DataFrame(per_frame)
    mean_sparsity = df_pf['sparsity_pct'].mean()
    total_gt      = df_pf['n_gt'].sum()
    mean_clusters = df_pf['n_clusters'].mean()
    frames_no_roi = df_pf['roi_x1'].isna().sum()

    col_w = 28
    all_keys = sorted(set(dm) | set(sm))
    print(f"\n{'Metric':<{col_w}} {'Dense':>10} {'Sparse (Gated)':>15}")
    print("─" * (col_w + 27))
    for k in all_keys:
        print(f"{k:<{col_w}} {dm.get(k, float('nan')):>10.4f}"
              f" {sm.get(k, float('nan')):>15.4f}")
    print("─" * (col_w + 27))
    mean_d_lat = df_pf['dense_lat_ms'].mean()
    mean_s_lat = df_pf['total_sparse_lat_ms'].mean()
    print(f"{'mean_latency_ms':<{col_w}} {mean_d_lat:>10.1f} {mean_s_lat:>15.1f}")
    print(f"{'mean_sparsity_pct':<{col_w}} {'N/A':>10} {mean_sparsity:>14.1f}%")
    print(f"\nFrames evaluated:      {len(per_frame)}")
    print(f"Total GT objects:      {total_gt}")
    print(f"Mean clusters/frame:   {mean_clusters:.1f}")
    print(f"Frames with no ROI:    {frames_no_roi}")

    # ── Verify hero-result condition ──────────────────────────────────────
    ds_small = dm.get('map_small', float('nan'))
    ss_small = sm.get('map_small', float('nan'))
    print(f"\nmap_small  Dense={ds_small:.4f}  Sparse={ss_small:.4f}")
    if not (ds_small != ds_small) and not (ss_small != ss_small):  # not NaN
        if ss_small > ds_small:
            print("✓ Hero condition met: Sparse map_small > Dense map_small")
        else:
            print("✗ Hero condition NOT met: Sparse map_small did not exceed Dense")

    # ── Save CSV ───────────────────────────────────────────────────────────
    summary_rows = [{'metric': k,
                     'dense': dm.get(k, float('nan')),
                     'sparse_gated': sm.get(k, float('nan'))}
                    for k in all_keys]
    summary_rows += [
        {'metric': 'mean_latency_ms',
         'dense': round(mean_d_lat, 2), 'sparse_gated': round(mean_s_lat, 2)},
        {'metric': 'mean_sparsity_pct',
         'dense': 0.0, 'sparse_gated': round(mean_sparsity, 2)},
    ]

    out_summary = Path('results/table4_detection_mAP.csv')
    out_frames  = Path('results/table4_per_frame.csv')
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    df_pf.to_csv(out_frames, index=False)
    print(f"\nSaved → {out_summary}")
    print(f"Saved → {out_frames}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exp 04: Detection mAP')
    parser.add_argument('--n', type=int, default=100,
                        help='Number of KITTI frames (default: 100)')
    args = parser.parse_args()
    main(n_frames=args.n)
