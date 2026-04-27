import time
import cv2
import numpy as np
from ultralytics import YOLO

# Load the exact model used for the mAP audit
model = YOLO('yolo11m.pt')

def measure_inference_ms(model, img, n=50, warmup=10):
    # Warmup phase (gets the NPU/GPU out of idle state)
    for _ in range(warmup):
        model.predict(img, verbose=False)
        
    times = []
    # Measurement phase
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(img, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
        
    return np.mean(times), np.std(times)

# --- 1. Load a real KITTI frame ---
img_path = 'data/kitti/data_tracking_image/image_02/0000/000000.png'
full_frame = cv2.imread(img_path)

if full_frame is None:
    print(f"Error: Could not load image at {img_path}")
else:
    # --- 2. Dense Baseline ---
    mean_dense, std_dense = measure_inference_ms(model, full_frame)

    # --- 3. Sparse ROI Crop ---
    # Simulating a typical tight cluster around a car
    # KITTI cars are roughly 100-200px wide depending on distance
    typical_crop = full_frame[150:250, 400:600] 
    mean_sparse, std_sparse = measure_inference_ms(model, typical_crop)

    print("\n" + "="*50)
    print("EMPIRICAL INFERENCE LATENCY (M3 Pro)")
    print("="*50)
    print(f"Dense (1242x375): {mean_dense:.1f} ± {std_dense:.1f} ms")
    print(f"Sparse (200x100): {mean_sparse:.1f} ± {std_sparse:.1f} ms")
    print("-" * 50)
    print(f"Measured speedup: {mean_dense/mean_sparse:.2f}x")
    print("="*50)