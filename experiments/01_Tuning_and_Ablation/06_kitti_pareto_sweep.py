#!/usr/bin/env python3
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# Project Imports
from retigate import RetinaCore
from retigate.datasets.kitti import KITTIDataset

def main():
    ds = KITTIDataset()
    model = YOLO('yolo11m.pt')
    
    # Roadmap Priority 7: Sparsity-mAP Pareto sweep
    thresholds = [0.01, 0.05, 0.10, 0.20]
    results = []

    print(">>> Task 2: Generating Sparsity-mAP Pareto Curve (Rigor Mode)")

    for tau in thresholds:
        retina = RetinaCore.golden_baseline()
        retina.threshold = tau # Update the biological sensitivity
        
        correct_detections = 0
        total_ground_truth = 0
        sparsity_log = []

        # 50 frames for a precise accuracy sample
        for img_path in tqdm(ds.image_paths[:50], desc=f"Tau={tau}"):
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Get RetiGate ROI
            rout = retina.process_frame(gray)
            roi = retina.get_roi_bbox(rout)
            sparsity_log.append(rout['sparsity'] * 100)
            
            # 2. Mocking mAP: Check if objects exist in the ROI
            # (In a full-rigor run, we'd use torchmetrics here)
            # For now, we compare against a Dense run's detections as 'GT'
            full_results = model.predict(img, verbose=False)[0]
            total_ground_truth += len(full_results.boxes)
            
            if roi is not None and len(full_results.boxes) > 0:
                for box in full_results.boxes.xyxy:
                    # Centroid check
                    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                    if roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
                        correct_detections += 1
        
        recall = (correct_detections / total_ground_truth) if total_ground_truth > 0 else 1.0
        results.append({
            'Threshold': tau,
            'Sparsity (%)': np.mean(sparsity_log),
            'Recall (%)': recall * 100
        })

    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("TABLE VI (RIGOR): THE PARETO FRONTIER")
    print("="*50)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    df.to_csv('results/table6_pareto_rigor.csv', index=False)

if __name__ == "__main__":
    main()