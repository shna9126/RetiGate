#!/usr/bin/env python3
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Project Imports
from retigate import RetinaCore
from retigate.datasets.kitti import KITTIDataset
from retigate.baselines.sahi_baseline import SAHIBaseline 

def main(n_frames=50): # 50 frames is enough for a deep dive
    ds = KITTIDataset()
    retina = RetinaCore.golden_baseline()
    sahi_bench = SAHIBaseline()
    
    results = []
    print(f">>> EXPERIMENT 05: RETIGATE vs. SAHI (BRUTE FORCE)")

    for img_path in tqdm(ds.image_paths[:n_frames], desc="Dual Evaluation"):
        img_bgr = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # --- 1. RUN SAHI (The "Expensive" Reality) ---
        t_start_sahi = time.perf_counter()
        sahi_boxes, sahi_count = sahi_bench.predict(str(img_path))
        t_sahi = (time.perf_counter() - t_start_sahi) * 1000

        # --- 2. RUN RETIGATE (The "Biological" Logic) ---
        t_start_reti = time.perf_counter()
        retina.reset_memory()
        # Warmup simulating real temporal stream
        for _ in range(5): retina.process_frame(gray)
        rout = retina.process_frame(gray)
        
        # Determine Gated ROI
        roi = retina.get_roi_bbox(rout, pad=20, max_area_frac=0.25)
        t_reti = (time.perf_counter() - t_start_reti) * 1000

        # --- 3. COMPARE ---
        # We calculate "Recall": how many of SAHI's objects were inside RetiGate's ROI?
        captured = 0
        if roi is not None and len(sahi_boxes) > 0:
            for box in sahi_boxes:
                # Center of the object
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                if roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
                    captured += 1
        
        recall = (captured / sahi_count) if sahi_count > 0 else 1.0

        results.append({
            'frame': img_path.name,
            'sahi_latency_ms': t_sahi,
            'retigate_latency_ms': t_reti,
            'sahi_objs': sahi_count,
            'reti_captured': captured,
            'recall': recall,
            'speedup': t_sahi / t_reti
        })

    # Final "Bigger Picture" Report
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("FINAL EXPERIMENT 05 SUMMARY: RETIGATE vs. SAHI")
    print("="*60)
    print(f"Avg SAHI Latency:      {df['sahi_latency_ms'].mean():.2f} ms")
    print(f"Avg RetiGate Latency:  {df['retigate_latency_ms'].mean():.2f} ms")
    print(f"System Speedup:        {df['speedup'].mean():.2f}x")
    print(f"Object Recall (GT):    {df['recall'].mean()*100:.2f}%")
    print("-" * 60)
    
    df.to_csv('results/table5_sahi_showdown.csv', index=False)

if __name__ == "__main__":
    main()