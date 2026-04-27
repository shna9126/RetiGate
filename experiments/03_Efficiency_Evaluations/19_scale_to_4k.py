#!/usr/bin/env python3
import time
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from retigate import RetinaCore

# Setup
model = YOLO('yolo11m.pt')
retina = RetinaCore.golden_baseline()

# Load a sample frame
img_raw = cv2.imread("data/kitti/data_scene_flow/training/image_2/000000_10.png")
# Upscale to 4K
img_4k = cv2.resize(img_raw, (3840, 2160))

print(">>> SCALE TEST: 4K (3840x2160) DENSE vs SPARSE")

# 1. DENSE 4K INFERENCE
t0 = time.perf_counter()
_ = model.predict(img_4k, verbose=False, imgsz=640) # Standard YOLO scaling
t_dense_4k = (time.perf_counter() - t0) * 1000

# 2. RETIGATE 4K SENSING
# We downsample for the retina (biological sensors don't need 4K to see motion!)
img_small = cv2.resize(img_4k, (1242, 375))
t0 = time.perf_counter()
gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
rout = retina.process_frame(gray)
roi_small = retina.get_roi_bbox(rout,frame_shape=img_small.shape)

# Scale ROI back to 4K coordinates
scale_x = 3840 / 1242
scale_y = 2160 / 375
roi_4k = [
    int(roi_small[0] * scale_x), int(roi_small[1] * scale_y),
    int(roi_small[2] * scale_x), int(roi_small[3] * scale_y)
]
t_sensing_4k = (time.perf_counter() - t0) * 1000

# 3. SPARSE 4K INFERENCE (The Crop)
crop = img_4k[roi_4k[1]:roi_4k[3], roi_4k[0]:roi_4k[2]]
t0 = time.perf_counter()
_ = model.predict(crop, verbose=False, imgsz=640)
t_sparse_4k = (time.perf_counter() - t0) * 1000

print("\n" + "="*60)
print("RESOLUTION SCALING RESULTS (4K Simulation)")
print("="*60)
print(f"Dense 4K Latency:   {t_dense_4k:.2f} ms")
print(f"Sparse 4K Latency:  {t_sparse_4k:.2f} ms")
print(f"Sensing Overhead:   {t_sensing_4k:.2f} ms")
print("-" * 60)
# Note: In Pipelined mode, Effective = Sparse Latency
print(f"Effective 4K Speedup: {t_dense_4k / t_sparse_4k:.2f}x")
print("="*60)