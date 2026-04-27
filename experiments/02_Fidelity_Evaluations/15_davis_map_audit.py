#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Project Imports
from retigate import RetinaCore

# --- CONFIGURATION FOR DAVIS 2017 480p ---
DAVIS_ROOT = Path("data/davis/DAVIS")
IMAGE_DIR  = DAVIS_ROOT / "JPEGImages" / "480p"
ANNOT_DIR  = DAVIS_ROOT / "Annotations" / "480p"

def get_boxes_from_mask(mask_path):
    """Converts DAVIS segmentation mask to Bounding Boxes."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None: return []
    
    # Each unique color in DAVIS mask represents a different object
    unique_ids = np.unique(mask)
    boxes = []
    for uid in unique_ids:
        if uid == 0: continue # Skip background
        ys, xs = np.where(mask == uid)
        boxes.append([np.min(xs), np.min(ys), np.max(xs), np.max(ys)])
    return boxes

def calculate_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter / float(a1 + a2 - inter + 1e-6)

def main():
    model = YOLO('yolo11m.pt')
    retina = RetinaCore.golden_baseline()
    retina.threshold = 0.10 # Consistency check: Use same tau as KITTI
    
    # Pick a specific sequence (e.g., 'bike-packing') for temporal continuity
    sequence = "bike-packing"
    img_paths = sorted(list((IMAGE_DIR / sequence).glob("*.jpg")))
    
    results = []
    print(f">>> TASK A (Interim): Formal mAP Audit on DAVIS [{sequence}]")

    for img_path in tqdm(img_paths):
        img = cv2.imread(str(img_path))
        mask_path = ANNOT_DIR / sequence / img_path.name.replace(".jpg", ".png")
        gt_boxes = get_boxes_from_mask(mask_path)
        
        if not gt_boxes: continue

        # 1. Biological Filtering
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rout = retina.process_frame(gray)
        roi = retina.get_roi_bbox(rout)
        
        if roi is None:
            results.append({'rec': 0})
            continue

        # 2. Gated Inference
        crop = img[roi[1]:roi[3], roi[0]:roi[2]]
        res = model.predict(crop, verbose=False, conf=0.25)[0]
        
        # 3. Precision/Recall Validation
        tp = 0
        matched = set()
        for box in res.boxes.xyxy:
            det = [box[0]+roi[0], box[1]+roi[1], box[2]+roi[0], box[3]+roi[1]]
            best_iou, best_idx = 0, -1
            for i, gt in enumerate(gt_boxes):
                if i in matched: continue
                iou = calculate_iou(det, gt)
                if iou > best_iou: best_iou, best_idx = iou, i
            if best_iou >= 0.5:
                tp += 1
                matched.add(best_idx)

        results.append({'rec': tp / len(gt_boxes)})

    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print(f"DAVIS mAP AUDIT REPORT: {sequence}")
    print(f"Mean Recall@0.5: {df['rec'].mean()*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()