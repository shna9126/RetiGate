#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Project Imports
from retigate import RetinaCore
from retigate.datasets.kitti import KITTIDataset
from retigate.baselines.sahi_baseline import SAHIBaseline

def main():
    ds = KITTIDataset()
    retina = RetinaCore.golden_baseline()
    sahi_bench = SAHIBaseline()
    
    Path('results/failures').mkdir(parents=True, exist_ok=True)
    
    print(">>> Experiment 09: Identifying Scientific Failure Cases")

    for img_path in tqdm(ds.image_paths[:300], desc="Scanning for Edge Cases"):
        img_bgr = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Get Ground Truth (SAHI)
        sahi_boxes, _ = sahi_bench.predict(str(img_path))
        
        # 2. Get RetiGate ROI
        retina.reset_memory()
        for _ in range(5): retina.process_frame(gray)
        rout = retina.process_frame(gray)
        roi = retina.get_roi_bbox(rout, pad=20)
        
        # 3. Logic: If SAHI found a car and RetiGate's ROI is empty or far away
        if len(sahi_boxes) > 0:
            if roi is None:
                # Total Miss Case: Object is there but Retina is blind to it
                label = "BLIND_MISS"
            else:
                # Centroid Check: Did the ROI actually cover the objects?
                captured = 0
                for box in sahi_boxes:
                    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                    if roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
                        captured += 1
                
                if captured == 0:
                    label = "SPATIAL_DISCONNECT"
                else:
                    continue # It's a success, skip it.

            # Save the failure for visual audit
            canvas = img_bgr.copy()
            # Draw SAHI detections in Red (What we missed)
            for b in sahi_boxes:
                cv2.rectangle(canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0,0,255), 2)
            
            # Draw RetiGate ROI in Green (Where we were looking instead)
            if roi:
                cv2.rectangle(canvas, (roi[0], roi[1]), (roi[2], roi[3]), (0,255,0), 3)
                
            cv2.putText(canvas, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imwrite(f"results/failures/{label}_{img_path.name}", canvas)

    print(f"\nFailure Analysis complete. Check 'results/failures/' for images.")

if __name__ == "__main__":
    main()