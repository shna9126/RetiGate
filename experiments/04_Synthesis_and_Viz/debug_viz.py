#!/usr/bin/env python3
"""
Visual audit for frame 000010_10.

Draws:
  Green  — GT boxes (after truncation/occlusion filter)
  Red    — YOLO11n predictions (imgsz=(384,1280))
  Blue   — RetiGate ROI crop boundary

Output: figures/debug_viz_000010.png

Usage:
    python experiments/debug_viz.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from retigate import RetinaCore
from retigate.datasets.kitti import KITTIDataset

YOLO_FILTER   = [0, 2]          # person, car only
YOLO_IMGSZ    = (384, 1280)     # dense full-frame inference size
TARGET_STEM   = '000010_10'
OUT_PATH      = Path('figures/debug_viz_000010.png')

GREEN = (0, 200, 0)
RED   = (0, 0, 220)
BLUE  = (220, 80, 0)
FONT  = cv2.FONT_HERSHEY_SIMPLEX


def draw_box(img, x1, y1, x2, y2, color, label='', thickness=2):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        cv2.rectangle(img, (int(x1), int(y1) - th - 4),
                      (int(x1) + tw + 2, int(y1)), color, -1)
        cv2.putText(img, label, (int(x1) + 1, int(y1) - 3),
                    FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ds    = KITTIDataset()
    model = YOLO('yolo11n.pt')
    retina = RetinaCore.golden_baseline()

    # Find target frame index
    target_idx = next(
        (i for i, p in enumerate(ds.image_paths) if p.stem == TARGET_STEM), None)
    if target_idx is None:
        sys.exit(f"Frame {TARGET_STEM} not found in dataset.")

    # Warm up retina on preceding frames so amacrine state is meaningful
    print(f"Warming up retina on frames 0–{target_idx - 1}…")
    for p in ds.image_paths[:target_idx]:
        img = cv2.imread(str(p))
        retina.process_frame(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # Load target frame
    target_path = ds.image_paths[target_idx]
    img_bgr = cv2.imread(str(target_path))
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    canvas  = img_bgr.copy()
    H, W    = canvas.shape[:2]

    # ── Ground truth ───────────────────────────────────────────────────
    gt_boxes, gt_cls = ds._load_label(target_path.stem)
    # Also load unfiltered GT to show what gets filtered
    gt_boxes_all, _ = ds._load_label(target_path.stem,
                                     max_truncation=1.0, max_occlusion=3)
    filtered_out = len(gt_boxes_all) - len(gt_boxes)

    COCO_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 7: 'truck'}
    for box, cls in zip(gt_boxes, gt_cls):
        draw_box(canvas, *box, GREEN,
                 label=f"GT:{COCO_NAMES.get(int(cls), str(cls))}", thickness=2)

    # ── YOLO predictions ───────────────────────────────────────────────
    res = model(img_bgr, classes=YOLO_FILTER, verbose=False, imgsz=YOLO_IMGSZ)[0]
    if res.boxes and len(res.boxes) > 0:
        for i in range(len(res.boxes)):
            x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy()
            conf = float(res.boxes.conf[i])
            cls  = int(res.boxes.cls[i])
            draw_box(canvas, x1, y1, x2, y2, RED,
                     label=f"{COCO_NAMES.get(cls, str(cls))} {conf:.2f}",
                     thickness=2)

    # ── RetiGate ROI ───────────────────────────────────────────────────
    rout = retina.process_frame(gray)
    roi  = retina.get_roi_bbox(rout, pad=40, frame_shape=img_bgr.shape)
    if roi:
        rx1, ry1, rx2, ry2 = roi
        roi_frac = (rx2 - rx1) * (ry2 - ry1) / (W * H) * 100
        draw_box(canvas, rx1, ry1, rx2, ry2, BLUE,
                 label=f"ROI {roi_frac:.0f}%", thickness=3)

    # ── Legend ─────────────────────────────────────────────────────────
    legend_items = [
        (GREEN, f"GT boxes: {len(gt_boxes)} shown  ({filtered_out} filtered)"),
        (RED,   f"YOLO preds: {len(res.boxes) if res.boxes else 0}  (imgsz={YOLO_IMGSZ})"),
        (BLUE,  f"RetiGate ROI  (sparsity {rout['sparsity']*100:.1f}%)"),
    ]
    lx, ly = 8, H - 10 - len(legend_items) * 22
    for color, text in legend_items:
        cv2.rectangle(canvas, (lx, ly - 12), (lx + 14, ly + 2), color, -1)
        cv2.putText(canvas, text, (lx + 18, ly), FONT, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, text, (lx + 18, ly), FONT, 0.5,
                    (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (lx + 18, ly), FONT, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        ly += 22

    OUT_PATH.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT_PATH), canvas)
    print(f"\nSaved → {OUT_PATH}")
    print(f"GT kept: {len(gt_boxes)}  GT filtered: {filtered_out}")
    print(f"YOLO predictions: {len(res.boxes) if res.boxes else 0}")
    print(f"RetiGate ROI: {roi}  ({roi_frac:.1f}% of frame)" if roi else "RetiGate ROI: None")

    # ── Quick IoU report ───────────────────────────────────────────────
    def iou(b1, b2):
        xi1=max(b1[0],b2[0]); yi1=max(b1[1],b2[1])
        xi2=min(b1[2],b2[2]); yi2=min(b1[3],b2[3])
        inter=max(0,xi2-xi1)*max(0,yi2-yi1)
        a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
        return inter/(a1+a2-inter+1e-6)

    if res.boxes and len(res.boxes) > 0 and len(gt_boxes) > 0:
        pred_np = res.boxes.xyxy.cpu().numpy()
        print(f"\nBest IoU per GT box (after filter, imgsz={YOLO_IMGSZ}):")
        for i, gb in enumerate(gt_boxes):
            best = max(iou(gb, pb) for pb in pred_np)
            cls_name = COCO_NAMES.get(int(gt_cls[i]), '?')
            print(f"  GT[{i}] {cls_name} ({gb[0]:.0f},{gb[1]:.0f},{gb[2]:.0f},{gb[3]:.0f})"
                  f"  best_iou={best:.3f}  {'✓' if best>=0.5 else '✗'}")


if __name__ == '__main__':
    main()
