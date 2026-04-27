#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
# Project Imports
from retigate import RetinaCore
from retigate.datasets.kitti import KITTIDataset

def main():
    ds = KITTIDataset()
    # Sweep from 'Flicker-Sensitive' (0.8) to 'High-Persistence' (0.01)
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
    results = []

    print(f">>> Experiment 07: Alpha Sweep (Temporal Persistence vs. Sparsity)")

    for a in alphas:
        sparsity_log = []
        # We modify the amacrine_decay for each run
        retina = RetinaCore.golden_baseline()
        retina.amacrine_decay = a
        
        # 100 frames to see the temporal habituation settle
        for img_path in tqdm(ds.image_paths[:100], desc=f"Alpha {a}", leave=False):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            # We don't reset memory here because we WANT to see the 'tail' build up
            rout = retina.process_frame(img)
            sparsity_log.append(rout['sparsity'] * 100)

        results.append({
            'Alpha': a,
            'Avg_Sparsity': np.mean(sparsity_log)
        })

    df = pd.DataFrame(results)
    print("\n" + "="*45)
    print("TABLE VII: THE TEMPORAL DECAY CHARACTERISTIC")
    print("="*45)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    df.to_csv('results/table7_alpha_data.csv', index=False)
    print(f"\nAlpha data saved to results/table7_alpha_data.csv")

if __name__ == "__main__":
    main()