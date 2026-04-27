#!/usr/bin/env python3
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# Project Imports
from retigate import RetinaCore
from retigate.datasets.kitti import KITTIDataset
from retigate.baselines.sahi_baseline import SAHIBaseline

def main():
    ds = KITTIDataset()
    model = YOLO('yolo11m.pt')
    retina = RetinaCore.golden_baseline()
    retina.threshold = 0.10 # Using our newly discovered optimal point
    sahi_bench = SAHIBaseline(model_path='yolo11m.pt')
    
    # Legacy Baseline: MOG2
    back_sub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=False)

    results = []
    print(">>> STARTING THE GRAND TOURNAMENT (FINAL AUDIT)")

    for img_path in tqdm(ds.image_paths[:50], desc="Tournament"):
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- 1. DENSE YOLO ---
        t0 = time.perf_counter()
        d_res = model.predict(img, verbose=False)[0]
        t_dense = (time.perf_counter() - t0) * 1000
        d_objs = len(d_res.boxes)

        # --- 2. SAHI (Brute Force) ---
        t0 = time.perf_counter()
        _, s_objs = sahi_bench.predict(str(img_path))
        t_sahi = (time.perf_counter() - t0) * 1000

        # --- 3. MOG2 (Legacy CV) ---
        t0 = time.perf_counter()
        mask = back_sub.apply(img)
        # Simulate a crop on the largest MOG2 contour
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t_mog = (time.perf_counter() - t0) * 1000

        # --- 4. RETIGATE (Proposed) ---
        t0 = time.perf_counter()
        rout = retina.process_frame(gray)
        roi = retina.get_roi_bbox(rout)
        r_objs = 0
        if roi is not None:
            crop = img[roi[1]:roi[3], roi[0]:roi[2]]
            if crop.size > 0:
                res = model.predict(crop, verbose=False)[0]
                r_objs = len(res.boxes)
        t_reti = (time.perf_counter() - t0) * 1000

        results.append({
            'Dense_Lat': t_dense, 'SAHI_Lat': t_sahi, 'Reti_Lat': t_reti,
            'Dense_Det': d_objs, 'SAHI_Det': s_objs, 'Reti_Det': r_objs
        })

    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("THE FINAL TOURNAMENT: PERFORMANCE & FIDELITY")
    print("="*70)
    summary = {
        "Metric": ["Avg Latency (ms)", "Avg Objects Detected"],
        "Dense YOLO": [df['Dense_Lat'].mean(), df['Dense_Det'].mean()],
        "SAHI": [df['SAHI_Lat'].mean(), df['SAHI_Det'].mean()],
        "RetiGate": [df['Reti_Lat'].mean(), df['Reti_Det'].mean()]
    }
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

if __name__ == "__main__":
    main()