#!/usr/bin/env python3
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

from retigate import RetinaCore
from retigate.metrics import DetectionEvaluator

# --- CONFIGURATION ---
DATA_ROOT = Path("data/kitti/data_tracking_image")
IMG_DIR = DATA_ROOT / "image_02"
LBL_DIR = DATA_ROOT / "label_02"

# Map KITTI strings to standard IDs for torchmetrics
CLASS_MAP = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

# Translation Layer: YOLO COCO ID -> KITTI Internal ID
# COCO: 0=person, 1=bicycle, 2=car
# KITTI: 0=Car, 1=Pedestrian, 2=Cyclist
YOLO_TO_KITTI = {
    0: 1, # Person -> Pedestrian
    1: 2, # Bicycle -> Cyclist
    2: 0, # Car -> Car
    5: 0, # Bus -> Car
    7: 0  # Truck -> Car
}

# Map all vehicle types to the "Car" ID (0)
LABEL_MAPPING = {
    "Car": 0, "Van": 0, "Truck": 0, "Tram": 0,
    "Pedestrian": 1, "Person_sitting": 1,
    "Cyclist": 2
}

def parse_tracking_labels(label_path):
    try:
        df = pd.read_csv(label_path, sep=' ', header=None)
        df = df[[0, 2, 6, 7, 8, 9]]
        df.columns = ['frame', 'type', 'x1', 'y1', 'x2', 'y2']
        # Only keep labels that are in our mapping
        df = df[df['type'].isin(LABEL_MAPPING.keys())].copy()
        # Convert the string types to our 0, 1, 2 IDs
        df['class_id'] = df['type'].map(LABEL_MAPPING)
        return df
    except:
        return pd.DataFrame()

def main():
    model = YOLO('yolo11m.pt')
    evaluator = DetectionEvaluator() # <-- Standardized Evaluator
    overall_mAP_scores = []
    
    sequences = sorted([d.name for d in IMG_DIR.iterdir() if d.is_dir()])
    print(f">>> TASK: Rigorous mAP Audit (COCO Protocol) on KITTI")

    for seq in sequences:
        label_file = LBL_DIR / f"{seq}.txt"
        if not label_file.exists(): continue
        
        evaluator.reset() # Clean slate for every sequence
        retina = RetinaCore.golden_baseline()
        retina.threshold = 0.10
        
        gt_df = parse_tracking_labels(label_file)
        img_paths = sorted(list((IMG_DIR / seq).glob("*.png")))
        
        for i, path in enumerate(tqdm(img_paths, desc=f"Seq {seq}", leave=False)):
            frame_gt = gt_df[gt_df['frame'] == i]
            if frame_gt.empty: continue
            
            img = cv2.imread(str(path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rout = retina.process_frame(gray)
            roi = retina.get_roi_bbox(rout, frame_shape=img.shape)
            
            # Prepare GT (Ensure CPU)
            gt_boxes = torch.tensor(frame_gt[['x1', 'y1', 'x2', 'y2']].values, dtype=torch.float32).cpu()
            gt_labels = torch.tensor(frame_gt['class_id'].values, dtype=torch.int64).cpu()

            if roi is None:
                evaluator.update(torch.empty((0, 4)), torch.empty(0), torch.empty(0, dtype=torch.int64), 
                                 gt_boxes, gt_labels)
                continue

            crop = img[roi[1]:roi[3], roi[0]:roi[2]]
            res = model.predict(crop, verbose=False, conf=0.40)[0]
            
            if len(res.boxes) > 0:
                # 1. Map boxes back to global image and ensure CPU
                pred_boxes = res.boxes.xyxy.clone().cpu()
                pred_boxes[:, [0, 2]] += roi[0]
                pred_boxes[:, [1, 3]] += roi[1]
                
                pred_scores = res.boxes.conf.cpu()
                
                # 2. TRANSLATION LAYER: Map YOLO labels to KITTI labels
                raw_labels = res.boxes.cls.cpu().tolist()
                mapped_labels = [YOLO_TO_KITTI.get(int(l), -1) for l in raw_labels]
                pred_labels = torch.tensor(mapped_labels, dtype=torch.int64)
                
                # 3. Filter out any detections not in our Big 3 (mapped to -1)
                valid_idx = pred_labels != -1
                pred_boxes = pred_boxes[valid_idx]
                pred_scores = pred_scores[valid_idx]
                pred_labels = pred_labels[valid_idx]
            else:
                pred_boxes = torch.empty((0, 4))
                pred_scores = torch.empty(0)
                pred_labels = torch.empty(0, dtype=torch.int64)

            evaluator.update(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)

        # Sequence Results (COCO mAP is the mean across IoU 0.5:0.95)
        stats = evaluator.compute()
        seq_map50 = stats['map_50'] * 100
        print(f"Sequence {seq}: mAP@0.5 = {seq_map50:.2f}%")
        overall_mAP_scores.append(seq_map50)

    print("\n" + "="*60)
    print("FINAL RIGOROUS AUDIT: KITTI TRACKING (mAP@0.5)")
    print("="*60)
    print(f"Mean System mAP: {np.mean(overall_mAP_scores):.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()