#!/usr/bin/env python3
import cv2 
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Project Imports
from retigate import RetinaCore
from retigate.datasets.kitti import KITTIDataset

# --- HARDWARE CONSTANTS (YOLOv11n Estimates) ---
# YOLOv11n at 640x640 is ~2.8 GFLOPs. 
# We scale this to the KITTI resolution area.
GFLOPS_DENSE_KITTI = 2.8 
WATT_PER_GFLOPS    = 0.5 # Typical mobile NPU/Edge TPU efficiency
GOLDEN_SHAPE       = (1242, 375) # Standardizing resolution for memory alignment

def main():
    ds = KITTIDataset()
    retina = RetinaCore.golden_baseline()
    results = []

    print(f">>> Experiment 08: Computational Power & Energy Analysis (KITTI)")

    # Processing 500 frames for a solid statistical average
    for img_path in tqdm(ds.image_paths[:500], desc="Analyzing Power"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        
        # FIX: Force consistent shape to prevent broadcasting errors
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, GOLDEN_SHAPE)
        
        # 1. Baseline Cost (Dense inference on full area)
        dense_gflops = GFLOPS_DENSE_KITTI
        
        # 2. RetiGate Cost
        # We calculate sparsity specifically for this frame
        rout = retina.process_frame(img_gray)
        
        # Active Fraction: the percentage of the frame the AI actually processes
        active_frac = 1.0 - rout['sparsity']
        
        # RetiGate Cost = (Gated Inference) + (Retina Overhead)
        # Overhead is roughly 0.05 GFLOPs for the CV operations on CPU
        reti_gflops = (active_frac * GFLOPS_DENSE_KITTI) + 0.05
        
        savings = (1 - (reti_gflops / dense_gflops)) * 100
        
        results.append({
            'frame': img_path.name,
            'dense_gflops': dense_gflops,
            'reti_gflops': reti_gflops,
            'energy_savings_pct': savings
        })

    df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("TABLE VIII: ESTIMATED POWER SAVINGS (EDGE NPU)")
    print("="*50)
    print(f"Avg Dense Workload:   {df['dense_gflops'].mean():.3f} GFLOPs")
    print(f"Avg RetiGate Workload: {df['reti_gflops'].mean():.3f} GFLOPs")
    print(f"Total Power Savings:   {df['energy_savings_pct'].mean():.2f}%")
    print("-" * 50)
    
    # Save the audit
    Path('results').mkdir(exist_ok=True)
    df.to_csv('results/table8_power_analysis.csv', index=False)
    print("Power audit saved to results/table8_power_analysis.csv")

if __name__ == "__main__":
    main()