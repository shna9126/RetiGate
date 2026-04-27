#!/usr/bin/env python3
import time
import cv2
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# Project Imports
from retigate import RetinaCore
from retigate.datasets.kitti import KITTIDataset

def run_benchmark(name, frames, processing_fn):
    print(f"\n>>> Starting Benchmark: {name}")
    log_file = f"/tmp/power_{name.replace(' ', '_')}.txt"
    
    # Start powermetrics background process (Roadmap Priority 5)
    pm_process = subprocess.Popen(
        ['sudo', 'powermetrics', '--samplers', 'cpu_power', '--sample-rate', '100', '-o', log_file],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    
    time.sleep(2)
    start_time = time.perf_counter()
    
    for frame in tqdm(frames, desc=f"Profiling {name}"):
        processing_fn(frame)
        
    duration = time.perf_counter() - start_time
    subprocess.run(['sudo', 'killall', 'powermetrics'], check=False)
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            power_vals = [float(l.split(":")[1].split("mW")[0].strip()) 
                         for l in lines if "CPU Power" in l]
        avg_power_mw = np.mean(power_vals) if power_vals else 0.0
    except Exception:
        avg_power_mw = 0.0

    energy_mj_per_frame = ((avg_power_mw / 1000) * duration / len(frames)) * 1000
    
    return {
        "Mode": name,
        "Latency (ms)": (duration / len(frames)) * 1000,
        "Avg Power (mW)": avg_power_mw,
        "Energy (mJ/frame)": energy_mj_per_frame
    }

def main():
    ds = KITTIDataset()
    # SWITCHING TO MEDIUM MODEL FOR SCALABILITY ANALYSIS
    print("Initializing YOLO11m (Medium) for Break-Even Test...")
    model = YOLO('yolo11m.pt') 
    retina = RetinaCore.golden_baseline()
    
    test_frames = []
    for p in ds.image_paths[:50]:
        img = cv2.imread(str(p))
        if img is not None:
            test_frames.append(cv2.resize(img, (1242, 375)))

    def dense_yolo(img):
        model.predict(img, verbose=False, device='cpu')

    def retigate_pipeline(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rout = retina.process_frame(gray)
        roi = retina.get_roi_bbox(rout)
        if roi is not None:
            crop = img[roi[1]:roi[3], roi[0]:roi[2]]
            if crop.size > 0:
                model.predict(crop, verbose=False, device='cpu')

    subprocess.run(['sudo', 'true'], check=True)
    
    results = []
    results.append(run_benchmark("Dense YOLO11m (Baseline)", test_frames, dense_yolo))
    results.append(run_benchmark("RetiGate + YOLO11m", test_frames, retigate_pipeline))

    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("TABLE VIII (SCALABILITY): M3 PRO HARDWARE ENERGY PROFILE (MEDIUM MODEL)")
    print("="*70)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

if __name__ == "__main__":
    main()