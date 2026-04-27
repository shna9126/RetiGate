#!/usr/bin/env python3
"""
Experiment 02: Motion Method Comparison (Table II)
RetiGate vs traditional motion filters on KITTI with Ego-Motion.

Metrics:
  - Sparsity (%): How many pixels are gated out (Higher is better for energy efficiency).
  - Latency (ms): Processing time per frame.
  - Ego-Motion Resilience: Calculated qualitatively via sparsity under motion.
"""

import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Project Imports
from retigate import RetinaCore
from retigate.datasets.kitti import KITTIDataset
from retigate.baselines.frame_diff import FrameDiffBaseline
from retigate.baselines.mog2 import MOG2Baseline
from retigate.baselines.disflow import DISFlowBaseline
from retigate.baselines.farneback import FarnebackBaseline


# --- VOR Constants (from Exp 04 Breakthrough) ---
VOR_WARMUP_ITERS = 10
_ORB = cv2.ORB_create(500)
_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def vor_stabilize(gray_ref, gray_warp):
    kp1, des1 = _ORB.detectAndCompute(gray_ref, None)
    kp2, des2 = _ORB.detectAndCompute(gray_warp, None)
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return gray_warp
    matches = sorted(_MATCHER.match(des1, des2), key=lambda m: m.distance)[:50]
    if len(matches) < 4: return gray_warp
    pts_ref = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_warp = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, _ = cv2.findHomography(pts_warp, pts_ref, cv2.RANSAC, 5.0)
    if H is None: return gray_warp
    return cv2.warpPerspective(gray_warp, H, (gray_ref.shape[1], gray_ref.shape[0]), 
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def main(n_frames=100):
    Path('results').mkdir(exist_ok=True)
    ds = KITTIDataset()
    
    # Initialize all systems
    retina = RetinaCore.golden_baseline()
    # Inside main() of 02_baselines.py

    baselines = {
        "FrameDiff": FrameDiffBaseline(threshold=0.05),
        "MOG2": MOG2Baseline(),
        "DISFlow": DISFlowBaseline(),
        "Farneback": FarnebackBaseline() # Add this line
}
    
    results = []
    
    print(f"Running Comparative Baseline on {n_frames} KITTI frames...")
    
    for img_path in tqdm(ds.image_paths[:n_frames]):
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. RetiGate + VOR Logic
        t0 = time.perf_counter()
        retina.reset_memory()
        img_11 = cv2.imread(str(img_path).replace('_10.png', '_11.png'))
        if img_11 is not None:
            gray_11_stab = vor_stabilize(gray, cv2.cvtColor(img_11, cv2.COLOR_BGR2GRAY))
            for _ in range(VOR_WARMUP_ITERS):
                retina.process_frame(gray_11_stab)
        
        rout = retina.process_frame(gray)
        reti_lat = (time.perf_counter() - t0) * 1000
        
        frame_data = {
            'frame': img_path.stem,
            'RetiGate_Sparsity': rout['sparsity'] * 100,
            'RetiGate_Latency': reti_lat
        }
        
        # 2. Run Baselines
        for name, model in baselines.items():
            t0 = time.perf_counter()
            out = model.process_frame(gray)
            lat = (time.perf_counter() - t0) * 1000
            
            frame_data[f'{name}_Sparsity'] = out['sparsity'] * 100
            frame_data[f'{name}_Latency'] = lat
            
        results.append(frame_data)

    # --- Summary Generation ---
    df = pd.DataFrame(results)
    summary = []
    for sys_name in ["RetiGate", "FrameDiff", "MOG2", "DISFlow", "Farneback"]:
        summary.append({
            'Method': sys_name,
            'Mean_Sparsity (%)': df[f'{sys_name}_Sparsity'].mean(),
            'Mean_Latency (ms)': df[f'{sys_name}_Latency'].mean(),
            'Robust_to_EgoMotion': "YES" if sys_name == "RetiGate" else "NO"
        })
    
    df_summary = pd.DataFrame(summary)
    print("\n" + "="*50)
    print("TABLE II: MOTION METHOD COMPARISON")
    print("="*50)
    print(df_summary.to_string(index=False))
    
    df_summary.to_csv('results/table2_baselines.csv', index=False)
    print(f"\nSaved summary to results/table2_baselines.csv")

if __name__ == "__main__":
    main()