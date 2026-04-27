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
from retigate.datasets.kitti import KITTIDataset

# --- CONFIGURATION ---
DATA_ROOT = Path("data/kitti/data_tracking_image")
IMG_DIR = DATA_ROOT / "image_02"
LBL_DIR = DATA_ROOT / "label_02"

YOLO_TO_KITTI = {0: 1, 1: 2, 2: 0, 5: 0, 7: 0}
LABEL_MAPPING = {
    "Car": 0, "Van": 0, "Truck": 0, "Tram": 0,
    "Pedestrian": 1, "Person_sitting": 1,
    "Cyclist": 2
}

WARMUP_FRAMES = 8 # Slightly more aggressive warmup for lambda=0.1

def parse_tracking_labels(label_path):
    try:
        df = pd.read_csv(label_path, sep=' ', header=None)
        df = df[[0, 2, 6, 7, 8, 9]]
        df.columns = ['frame', 'type', 'x1', 'y1', 'x2', 'y2']
        df = df[df['type'].isin(LABEL_MAPPING.keys())].copy()
        df['class_id'] = df['type'].map(LABEL_MAPPING)
        return df
    except:
        return pd.DataFrame()

def process_yolo_results(res, offset=(0, 0)):
    if len(res.boxes) > 0:
        pred_boxes = res.boxes.xyxy.clone().cpu()
        pred_boxes[:, [0, 2]] += offset[0]
        pred_boxes[:, [1, 3]] += offset[1]
        
        pred_scores = res.boxes.conf.cpu()
        raw_labels = res.boxes.cls.cpu().tolist()
        mapped_labels = [YOLO_TO_KITTI.get(int(l), -1) for l in raw_labels]
        pred_labels = torch.tensor(mapped_labels, dtype=torch.int64)
        
        valid_idx = pred_labels != -1
        return pred_boxes[valid_idx], pred_scores[valid_idx], pred_labels[valid_idx]
    
    return torch.empty((0, 4)), torch.empty(0), torch.empty(0, dtype=torch.int64)

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO('yolo11m.pt').to(device)
    
    # Global Evaluators: Accumulate state across ALL sequences
    eval_dense = DetectionEvaluator()
    eval_sparse = DetectionEvaluator()
    
    sequences = sorted([d.name for d in IMG_DIR.iterdir() if d.is_dir()])
    print(f">>> TASK: Final mAP Audit (Warmup={WARMUP_FRAMES} frames)")

    for seq in sequences:
        label_file = LBL_DIR / f"{seq}.txt"
        if not label_file.exists(): continue
        
        retina = RetinaCore.golden_baseline(use_vos=True)
        gt_df = parse_tracking_labels(label_file)
        img_paths = sorted(list((IMG_DIR / seq).glob("*.png")))
        
        for i, path in enumerate(tqdm(img_paths, desc=f"Seq {seq}", leave=False)):
            img = cv2.imread(str(path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # --- 1. ALWAYS process through retina for temporal warmup ---
            rout = retina.process_frame(gray)
            
            # --- 2. Warmup Skip Logic ---
            if i < WARMUP_FRAMES:
                continue

            frame_gt = gt_df[gt_df['frame'] == i]
            if frame_gt.empty: continue
            
            gt_boxes = torch.tensor(frame_gt[['x1', 'y1', 'x2', 'y2']].values, dtype=torch.float32).cpu()
            gt_labels = torch.tensor(frame_gt['class_id'].values, dtype=torch.int64).cpu()

            # --- PASS 1: DENSE ---
            res_dense = model.predict(img, verbose=False, conf=0.40, device=device)[0]
            d_boxes, d_scores, d_labels = process_yolo_results(res_dense)
            eval_dense.update(d_boxes, d_scores, d_labels, gt_boxes, gt_labels)

            # --- PASS 2: SPARSE ---
            roi = retina.get_roi_bbox(rout, frame_shape=img.shape)
            
            if roi is not None:
                crop = img[roi[1]:roi[3], roi[0]:roi[2]]
                if crop.size > 0:
                    res_sparse = model.predict(crop, verbose=False, conf=0.40, device=device)[0]
                    s_boxes, s_scores, s_labels = process_yolo_results(res_sparse, offset=(roi[0], roi[1]))
                    eval_sparse.update(s_boxes, s_scores, s_labels, gt_boxes, gt_labels)
                else:
                    eval_sparse.update(torch.empty((0, 4)), torch.empty(0), torch.empty(0, dtype=torch.int64), gt_boxes, gt_labels)
            else:
                eval_sparse.update(torch.empty((0, 4)), torch.empty(0), torch.empty(0, dtype=torch.int64), gt_boxes, gt_labels)

    # FINAL COMPUTE
    m50_d = eval_dense.compute()['map_50'] * 100
    m50_s = eval_sparse.compute()['map_50'] * 100
    retention = (m50_s / m50_d) * 100 if m50_d > 0 else 0

    print("\n" + "="*65)
    print(f"FINAL AUDIT RESULTS (FULL 21-SEQUENCE AGGREGATE)")
    print(f"Dense mAP: {m50_d:.2f}% | Sparse mAP: {m50_s:.2f}%")
    print(f"ACCURACY RETENTION: {retention:.2f}%")
    print("="*65)
    
    Path('results/01_Synthesis').mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "Metric": "mAP@50", "Dense": f"{m50_d:.2f}%", 
        "Sparse": f"{m50_s:.2f}%", "Retention": f"{retention:.2f}%"
    }]).to_csv('results/01_Synthesis/table_accuracy_retention.csv', index=False)

if __name__ == "__main__":
    main()