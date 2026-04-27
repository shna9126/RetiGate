#!/usr/bin/env python3
import time
import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

# Project Imports
from retigate import RetinaCore
from retigate.datasets.kitti import KITTIDataset

# --- VOR LOGIC (MODIFIED FOR MOTION MAGNITUDE) ---
_ORB = cv2.ORB_create(500)
_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def vor_stabilize(gray_ref, gray_warp):
    """Returns stabilized image AND the magnitude of displacement."""
    kp1, des1 = _ORB.detectAndCompute(gray_ref, None)
    kp2, des2 = _ORB.detectAndCompute(gray_warp, None)
    
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4: 
        return gray_warp, 0.0
        
    matches = sorted(_MATCHER.match(des1, des2), key=lambda m: m.distance)[:50]
    if len(matches) < 4: return gray_warp, 0.0
    
    pts_ref = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_warp = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, _ = cv2.findHomography(pts_warp, pts_ref, cv2.RANSAC, 5.0)
    
    if H is None: 
        return gray_warp, 0.0
        
    # Calculate Motion Magnitude (L2 norm of translation components)
    mag = np.sqrt(H[0, 2]**2 + H[1, 2]**2)
    
    warped = cv2.warpPerspective(gray_warp, H, (gray_ref.shape[1], gray_ref.shape[0]), 
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped, mag

# --- SYSTEMATIC ABLATION CLASS ---
class RetinaAblator(RetinaCore):
    def __init__(self, variant='baseline'):
        super().__init__(amacrine_decay=0.1, global_weight=1.5, tail_len=15, shift_factor=0.5)
        self.variant = variant

    def process_frame(self, frame):
        img = frame.astype(np.float32) / 255.0

        # 1. Spatial Processing (DoG)
        if self.variant == 'no_dog':
            m_bipolar = img 
        else:
            m_c = cv2.filter2D(img, -1, self.m_center_k, borderType=cv2.BORDER_REFLECT)
            m_s = cv2.filter2D(img, -1, self.m_surround_k, borderType=cv2.BORDER_REFLECT)
            m_bipolar = np.abs(m_c - m_s)

        # 2. Temporal Memory (Amacrine)
        if self.variant == 'no_temporal':
            local_mem = m_bipolar
        else:
            if self.amacrine_state is None: self.amacrine_state = np.zeros_like(m_bipolar)
            self.amacrine_state = (self.amacrine_decay * m_bipolar) + ((1 - self.amacrine_decay) * self.amacrine_state)
            local_mem = self.amacrine_state

        # 3. Global Shunting (Inhibition)
        inhibition = 0 if self.variant == 'no_global' else (np.mean(local_mem) * self.global_weight)
        m_ganglion = np.maximum(0, m_bipolar - inhibition)

        # 4. Directional Selectivity (SAC Tail)
        if self.variant == 'no_sac':
            ds_r, ds_l = m_ganglion, m_ganglion
        else:
            smeared = cv2.blur(local_mem, (self.tail_len, 1))
            ds_r = np.maximum(0, m_ganglion - (np.roll(smeared, -7, axis=1) * 2.5))
            ds_l = np.maximum(0, m_ganglion - (np.roll(smeared, 7, axis=1) * 2.5))

        return {"M_Motion": m_ganglion, "DS_R": ds_r, "DS_L": ds_l}

def main(n_frames=1000):
    ds = KITTIDataset()
    variants = ['baseline', 'no_vor', 'no_dog', 'no_temporal', 'no_global', 'no_sac']
    results = []
    Path('results').mkdir(exist_ok=True)

    print(f">>> Starting 1,000-Frame Audit-Proof Ablation Run...")

    for v in variants:
        model = RetinaAblator(variant=v)
        sparsity_log, dsi_log, mag_log = [], [], []

        for img_path in tqdm(ds.image_paths[:n_frames], desc=f"Ablating: {v}"):
            img_curr = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img_prev = cv2.imread(str(img_path).replace('_10.png', '_11.png'), cv2.IMREAD_GRAYSCALE)
            if img_prev is None: continue

            model.reset_memory()
            
            # Stabilization & Magnitude tracking
            # We run stabilization even in no_vor just to calculate the motion of the frame
            img_stab, mag = vor_stabilize(img_curr, img_prev)
            mag_log.append(mag)

            if v != 'no_vor':
                for _ in range(5): model.process_frame(img_stab)

            out = model.process_frame(img_curr)
            
            # Metrics
            active = out["M_Motion"] > 0.05
            sparsity_log.append(100 * (1.0 - (np.sum(active) / out["M_Motion"].size)))

            r_sig, l_sig = np.sum(out["DS_R"]), np.sum(out["DS_L"])
            dsi = np.abs(r_sig - l_sig) / (r_sig + l_sig + 1e-5)
            dsi_log.append(dsi)

        # Store mean results
        results.append({
            "Variant": v,
            "Sparsity (%)": np.mean(sparsity_log),
            "Motion DSI": np.mean(dsi_log),
            "Avg Motion (px)": np.mean(mag_log)
        })

    df = pd.DataFrame(results)
    
    print("\n" + "="*75)
    print("TABLE V: SYSTEMATIC ABLATION RESULTS (KITTI - 1000 FRAME AUDIT)")
    print("="*75)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    df.to_csv('results/table3_ablation_1000frames.csv', index=False)
    print(f"\nLocked results saved to results/table5_ablation_1000frames.csv")

if __name__ == "__main__":
    main()