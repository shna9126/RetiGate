#!/usr/bin/env python3
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
# Project Imports
from retigate import RetinaCore
from retigate.datasets.kitti import KITTIDataset

def main():
    ds = KITTIDataset()
    retina = RetinaCore.golden_baseline()
    
    # YOLO-MOTF Proxy: Dense Farnebäck Flow
    # We use the standard parameters from their literature
    farneback_params = dict(
        pyr_scale=0.5, levels=3, winsize=15, 
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    results = []
    print(">>> FINAL BENCHMARK: RETIGATE vs. DENSE OPTICAL FLOW (MOTF)")

    # Test on 100 frames for statistical significance
    prev_gray = None
    
    for img_path in tqdm(ds.image_paths[:100], desc="Comparing Latency"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (1242, 375)) # Standardize resolution
        
        # --- 1. DENSE FLOW (YOLO-MOTF Baseline) ---
        t_flow = 0
        if prev_gray is not None:
            t0 = time.perf_counter()
            _ = cv2.calcOpticalFlowFarneback(prev_gray, img, None, **farneback_params)
            t_flow = (time.perf_counter() - t0) * 1000
        prev_gray = img

        # --- 2. RETIGATE (Proposed) ---
        t0 = time.perf_counter()
        _ = retina.process_frame(img)
        t_reti = (time.perf_counter() - t0) * 1000

        if t_flow > 0: # Skip the first frame
            results.append({
                'Farneback_ms': t_flow,
                'RetiGate_ms': t_reti
            })

    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("MOTION SENSING LATENCY COMPARISON")
    print("="*60)
    print(f"Avg Farnebäck (Dense Flow): {df['Farneback_ms'].mean():.2f} ms")
    print(f"Avg RetiGate (Bio-Retina):  {df['RetiGate_ms'].mean():.2f} ms")
    print(f"RetiGate Speedup Factor:    {df['Farneback_ms'].mean() / df['RetiGate_ms'].mean():.2f}x")
    print("-" * 60)
    
    # Save for Table II of the paper
    df.to_csv('results/table2_motion_latency.csv', index=False)

if __name__ == "__main__":
    main()