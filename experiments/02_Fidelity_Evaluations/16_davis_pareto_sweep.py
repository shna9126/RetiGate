#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Project Imports
from retigate import RetinaCore

# --- CONFIGURATION ---
DAVIS_ROOT = Path("data/davis/DAVIS")
IMAGE_DIR  = DAVIS_ROOT / "JPEGImages" / "480p"
ANNOT_DIR  = DAVIS_ROOT / "Annotations" / "480p"

def get_boxes_from_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None: return []
    unique_ids = np.unique(mask)
    boxes = []
    for uid in unique_ids:
        if uid == 0: continue
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
    thresholds = [0.01, 0.05, 0.10, 0.20]
    sequence = "bike-packing" # Our established baseline sequence
    img_paths = sorted(list((IMAGE_DIR / sequence).glob("*.jpg")))
    
    final_results = []
    print(f">>> Task B: DAVIS Pareto Sweep (Robustness Check)")

    for tau in thresholds:
        retina = RetinaCore.golden_baseline()
        retina.threshold = tau
        
        frame_recalls = []
        sparsity_vals = []
        
        for img_path in tqdm(img_paths, desc=f"Sweep Tau={tau}"):
            img = cv2.imread(str(img_path))
            mask_path = ANNOT_DIR / sequence / img_path.name.replace(".jpg", ".png")
            gt_boxes = get_boxes_from_mask(mask_path)
            if not gt_boxes: continue

            # 1. Retina Processing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rout = retina.process_frame(gray)
            roi = retina.get_roi_bbox(rout,frame_shape=img.shape)
            sparsity_vals.append(rout['sparsity'] * 100)
            
            if roi is None:
                frame_recalls.append(0)
                continue

            # 2. Inference & Matching
            crop = img[roi[1]:roi[3], roi[0]:roi[2]]
            res = model.predict(crop, verbose=False, conf=0.25)[0]
            
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
            
            frame_recalls.append(tp / len(gt_boxes))

        final_results.append({
            'Threshold': tau,
            'Sparsity (%)': np.mean(sparsity_vals),
            'Recall (%)': np.mean(frame_recalls) * 100
        })

    df = pd.DataFrame(final_results)
    print("\n" + "="*50)
    print("TABLE VI-B (ROBUST): DAVIS PARETO FRONTIER")
    print("="*50)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    df.to_csv('results/table6b_davis_pareto.csv', index=False)

if __name__ == "__main__":
    main()