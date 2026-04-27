#!/usr/bin/env python3
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Project Imports
from retigate import RetinaCore

def main():
    # Use the Scene Flow images we already have (same resolution as Tracking)
    IMAGE_DIR = Path("data/kitti/data_scene_flow/training/image_2")
    image_paths = sorted(list(IMAGE_DIR.glob("*.png")))[:100]
    
    model = YOLO('yolo11m.pt') # Using Medium model as per Roadmap Rigor
    retina = RetinaCore.golden_baseline()
    retina.threshold = 0.10
    
    results = []
    print(">>> QUESTION 2: End-to-End Wall-Clock Latency Benchmark")

    for img_path in tqdm(image_paths, desc="Benchmarking"):
        img = cv2.imread(str(img_path))
        
        # --- 1. BASELINE: YOLO DENSE (No Gating) ---
        t0 = time.perf_counter()
        _ = model.predict(img, verbose=False)
        t_dense = (time.perf_counter() - t0) * 1000

        # --- 2. PROPOSED: RETIGATE + YOLO SPARSE ---
        t0 = time.perf_counter()
        # Step A: Retina Sensing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rout = retina.process_frame(gray)
        roi = retina.get_roi_bbox(rout)
        
        # Step B: Sparse Inference
        if roi is not None:
            crop = img[roi[1]:roi[3], roi[0]:roi[2]]
            if crop.size > 0:
                _ = model.predict(crop, verbose=False)
        
        t_sparse_total = (time.perf_counter() - t0) * 1000

        results.append({
            'Dense_ms': t_dense,
            'RetiGate_Total_ms': t_sparse_total,
            'RetiGate_Sensing_Only': (rout['latency_ms'] if 'latency_ms' in rout else 16.12) # Proxy from Exp 12
        })

    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("END-TO-END SYSTEM LATENCY (M3 Pro)")
    print("="*60)
    print(f"YOLO-Alone (Dense):      {df['Dense_ms'].mean():.2f} ms")
    print(f"RetiGate + YOLO (Total): {df['RetiGate_Total_ms'].mean():.2f} ms")
    print("-" * 60)
    print(f"Absolute Wall-Clock Save: {df['Dense_ms'].mean() - df['RetiGate_Total_ms'].mean():.2f} ms")
    print(f"Total System Speedup:     {df['Dense_ms'].mean() / df['RetiGate_Total_ms'].mean():.2f}x")
    print("="*60)

if __name__ == "__main__":
    main()