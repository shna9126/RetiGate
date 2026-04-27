#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

# Project Imports
from retigate import RetinaCore
from retigate.datasets.kitti import KITTIDataset

class RetinaAblator(RetinaCore):
    def __init__(self, variant='baseline'):
        # Pass variant-specific flags to constructor
        # Note: Added 'no_dog' and 'no_temporal' as valid variants
        super().__init__(
            use_vos=(variant != 'no_vor'),
            use_global_inh=(variant != 'no_global'),
            use_sac_tail=(variant != 'no_sac')
        )
        self.variant = variant
        # If no_temporal, we set alpha to 1.0 (no leaky memory)
        self.amacrine_decay = 0.1 if variant != 'no_temporal' else 1.0

    def process_frame(self, frame):
        # --- Stage 0: VOS ---
        work_gray = self._vos_stabilize(frame) if self.use_vos else frame
        img = work_gray.astype(np.float32) / 255.0

        # --- Stage 1: Spatial (DoG) ---
        if self.variant == 'no_dog':
            m_bipolar = img # No bandpass filtering
        else:
            m_c = cv2.filter2D(img, -1, self.m_center_k, borderType=cv2.BORDER_REFLECT)
            m_s = cv2.filter2D(img, -1, self.m_surround_k, borderType=cv2.BORDER_REFLECT)
            m_bipolar = np.abs(m_c - m_s)

        # --- Stage 2: Temporal (Amacrine) ---
        if self.amacrine_state is None: self.amacrine_state = np.zeros_like(m_bipolar)
        
        if self.variant == 'no_temporal':
            # Signal passes through, but memory is not used for inhibition
            self.amacrine_state = m_bipolar 
            local_mem = np.zeros_like(m_bipolar) # Kill the memory feedback
        else:
            self.amacrine_state = (self.amacrine_decay * m_bipolar + 
                                  (1 - self.amacrine_decay) * self.amacrine_state)
            local_mem = self.amacrine_state

        # --- Stage 3: Inhibition (Ganglion) ---
        # If no_global, we only subtract the local memory, not the mean
        weight = 0 if self.variant == 'no_global' else self.global_weight
        inhibition = local_mem + weight * np.mean(local_mem)
        m_ganglion = np.maximum(0.0, m_bipolar - inhibition)

        # --- Stage 4: SAC Tail ---
        if self.use_sac_tail and self.variant != 'no_sac':
            smeared = cv2.blur(self.amacrine_state, (self.tail_len, 1))
            ds_r = np.maximum(0.0, m_ganglion - np.roll(smeared, -self.shift_amount, axis=1))
            ds_l = np.maximum(0.0, m_ganglion - np.roll(smeared, self.shift_amount, axis=1))
        else:
            ds_r, ds_l = m_ganglion.copy(), m_ganglion.copy()

        active_mask = m_ganglion > self.threshold
        self.prev_gray = work_gray

        return {
            'M_Motion': m_ganglion,
            'DS_Right': ds_r,
            'DS_Left': ds_l,
            'sparsity': 1.0 - (np.sum(active_mask) / active_mask.size)
        }

def main(n_frames=2000): # Increased to 2000 for "Reviewer-Grade" Robustness
    ds = KITTIDataset()
    variants = ['baseline', 'no_vor', 'no_dog', 'no_temporal', 'no_global', 'no_sac']
    results = []
    
    # Ensure results directory exists
    Path('results/04_Ablations').mkdir(parents=True, exist_ok=True)

    print(f">>> Starting Systematic Ablation Audit on {min(n_frames, len(ds))} frames...")

    for v in variants:
        model = RetinaAblator(variant=v)
        sparsity_log, dsi_log = [], []

        # Use slicing to prevent IndexError and enumerate for initialization check
        test_paths = ds.image_paths[:n_frames]

        for i, img_path in enumerate(tqdm(test_paths, desc=f"Ablating: {v}", leave=False)):
            img_curr = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_curr is None: continue

            out = model.process_frame(img_curr)
            
            # Skip initialization frames (first frame or after a shape reset)
            if model.prev_gray is None or i == 0: 
                continue 

            # Handle both possible dict keys
            s_val = out.get('sparsity', 0)
            sparsity_log.append(s_val * 100)

            # Directional Selectivity Index (DSI)
            r_sig, l_sig = np.sum(out.get("DS_Right", 0)), np.sum(out.get("DS_Left", 0))
            dsi = np.abs(r_sig - l_sig) / (r_sig + l_sig + 1e-5)
            dsi_log.append(dsi)

        results.append({
            "Variant": v,
            "Sparsity (%)": np.mean(sparsity_log) if sparsity_log else 0,
            "Motion DSI": np.mean(dsi_log) if dsi_log else 0,
            "Bio-Analogy": v.replace('no_', 'Removed ')
        })

    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print(f"TABLE III: SYSTEMATIC ABLATION RESULTS ({len(test_paths)} FRAMES)")
    print("="*80)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("="*80)
    
    df.to_csv('results/04_Ablations/table3_ablation_final.csv', index=False)
    print(f"Results saved to results/04_Ablations/table3_ablation_final.csv")

if __name__ == "__main__":
    main()