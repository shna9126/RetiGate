#!/usr/bin/env python3
import time
import cv2
import queue
import threading
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from retigate import RetinaCore

# Setup
IMAGE_DIR = Path("data/kitti/data_scene_flow/training/image_2")
image_paths = sorted(list(IMAGE_DIR.glob("*.png")))[:50]
model = YOLO('yolo11m.pt')
retina = RetinaCore.golden_baseline()

# Queues for the pipeline
raw_frame_queue = queue.Queue(maxsize=5)
roi_queue = queue.Queue(maxsize=5)
results = []

def retina_thread():
    """Producer: Scans frames and identifies ROIs."""
    for path in image_paths:
        img = cv2.imread(str(path))
        t0 = time.perf_counter()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rout = retina.process_frame(gray)
        roi = retina.get_roi_bbox(rout, frame_shape=img.shape)
        
        latency = (time.perf_counter() - t0) * 1000
        roi_queue.put((img, roi, latency))
    roi_queue.put(None) # Sentinel

def yolo_thread():
    """Consumer: Performs sparse inference on ROIs."""
    while True:
        data = roi_queue.get()
        if data is None: break
        img, roi, retina_lat = data
        
        t0 = time.perf_counter()
        if roi is not None:
            crop = img[roi[1]:roi[3], roi[0]:roi[2]]
            if crop.size > 0:
                _ = model.predict(crop, verbose=False)
        yolo_lat = (time.perf_counter() - t0) * 1000
        
        results.append({
            'Retina_Lat': retina_lat,
            'YOLO_Lat': yolo_lat,
            'System_Effective_Lat': max(retina_lat, yolo_lat)
        })

print(">>> RUNNING ASYNC PIPELINE SIMULATION...")
t_start = time.perf_counter()

t1 = threading.Thread(target=retina_thread)
t2 = threading.Thread(target=yolo_thread)

t1.start()
t2.start()
t1.join()
t2.join()

total_time = (time.perf_counter() - t_start) * 1000
df = pd.DataFrame(results)

print("\n" + "="*60)
print("PIPELINED THROUGHPUT RESULTS (M3 Pro)")
print("="*60)
print(f"Avg Retina Sensing:    {df['Retina_Lat'].mean():.2f} ms")
print(f"Avg Sparse Inference:  {df['YOLO_Lat'].mean():.2f} ms")
print(f"Effective Frame Latency: {df['System_Effective_Lat'].mean():.2f} ms")
print("-" * 60)
print(f"Total Pipeline FPS:    {1000 / df['System_Effective_Lat'].mean():.2f} FPS")
print("="*60)