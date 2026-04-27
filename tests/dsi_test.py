import cv2
import numpy as np
import os
from retigate import RetinaCore

# --- 1. SET THE PATH MANUALLY HERE ---
# Check if your folder is '0000' or '0001' and if the file has 6 zeros.
IMG_PATH = 'data/kitti/data_tracking_image/image_02/0000/000000.png'

if not os.path.exists(IMG_PATH):
    print(f"❌ ERROR: File not found at {IMG_PATH}")
    print("Please check your 'data' folder. Is it image_02/0000/000000.png?")
else:
    static_frame = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    
    # --- 2. INITIALIZE ---
    # We use the unified parameters we just locked
    retina = RetinaCore(amacrine_decay=0.1, threshold=0.05, use_vos=True)

    print(f"✅ Image loaded successfully: {IMG_PATH}")
    print(">>> Diagnostic: Testing DSI on a STATIC scene...")

    for i in range(10):
        # We process the SAME frame 10 times.
        # Temporal difference SHOULD be zero after frame 1.
        out = retina.process_frame(static_frame)
        
        if i == 0: 
            print("Frame 0: Initializing memory...")
            continue 
        
        r = np.sum(out['DS_Right'])
        l = np.sum(out['DS_Left'])
        
        # DSI calculation
        dsi = abs(r - l) / (r + l + 1e-5)
        
        print(f"Iter {i} | DS_R: {r:.2f} | DS_L: {l:.2f} | DSI: {dsi:.4f}")