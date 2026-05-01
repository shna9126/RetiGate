#!/usr/bin/env python3
"""
02_baselines_v2.py
Motion Method Comparison — Table II
Adds RAFT. Fixes timing. Fixes first-frame handling.
"""

import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from retigate import RetinaCore
from retigate.datasets.kitti import KITTIDataset
from retigate.baselines.frame_diff import FrameDiffBaseline
from retigate.baselines.mog2 import MOG2Baseline
from retigate.baselines.disflow import DISFlowBaseline
from retigate.baselines.farneback import FarnebackBaseline
from retigate.baselines.raft_baseline import RAFTBaseline


def main(n_frames=100):
    Path('results/05_Baselines').mkdir(parents=True, exist_ok=True)
    ds = KITTIDataset()

    # ── Instantiate all methods ──────────────────────────────
    retina = RetinaCore.golden_baseline()

    baselines = {
        "FrameDiff": FrameDiffBaseline(threshold=0.05),
        "MOG2":      MOG2Baseline(),
        "DISFlow":   DISFlowBaseline(),
        "Farneback": FarnebackBaseline(),
        "RAFT":      RAFTBaseline(device='cpu', threshold=1.5),
    }

    results = []

    print(f"Running Motion Method Comparison on {n_frames} frames...")
    print("Timing: ONLY process_frame() — no warmup, no reset\n")

    for img_path in tqdm(ds.image_paths[:n_frames]):
        img  = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── RetiGate — time process_frame() only ────────────
        # Warmup is handled by the leaky integrator naturally
        # Do NOT reset between frames — that kills temporal state
        t0       = time.perf_counter()
        rout     = retina.process_frame(gray)
        reti_lat = (time.perf_counter() - t0) * 1000

        frame_data = {
            'frame':              img_path.stem,
            'RetiGate_Sparsity':  rout['sparsity'] * 100,
            'RetiGate_Latency':   reti_lat,
        }

        # ── Baselines — time process_frame() only ───────────
        for name, model in baselines.items():
            t0  = time.perf_counter()
            out = model.process_frame(gray)
            lat = (time.perf_counter() - t0) * 1000

            frame_data[f'{name}_Sparsity'] = out['sparsity'] * 100
            frame_data[f'{name}_Latency']  = lat

        results.append(frame_data)

    # ── Summary ──────────────────────────────────────────────
    df      = pd.DataFrame(results)
    methods = ["RetiGate", "FrameDiff", "MOG2",
               "DISFlow", "Farneback", "RAFT"]

    summary_rows = []
    for m in methods:
        # Skip first frame for baselines
        # (first frame returns sparsity=1.0, no prev_gray)
        sparsity_col = f'{m}_Sparsity'
        latency_col  = f'{m}_Latency'

        # First frame of each baseline returns 100% sparsity
        # (no previous frame). Drop it for fair average.
        sparsity_vals = df[sparsity_col].iloc[1:]  # skip frame 0
        latency_vals  = df[latency_col].iloc[1:]

        summary_rows.append({
            'Method':               m,
            'Mean_Sparsity (%)':    sparsity_vals.mean(),
            'Std_Sparsity (%)':     sparsity_vals.std(),
            'Mean_Latency (ms)':    latency_vals.mean(),
            'Std_Latency (ms)':     latency_vals.std(),
            'Robust_to_EgoMotion':  "YES" if m == "RetiGate" else "NO",
        })

    df_summary = pd.DataFrame(summary_rows)

    print("\n" + "="*70)
    print("TABLE II: MOTION METHOD COMPARISON")
    print("="*70)
    print(df_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    ))

    # Save
    df_summary.to_csv(
        'results/05_Baselines/table2_baselines_v2.csv',
        index=False
    )
    df.to_csv(
        'results/05_Baselines/table2_per_frame.csv',
        index=False
    )

    # ── Diagnostic: check first vs rest ─────────────────────
    print("\nDIAGNOSTIC — Frame 0 vs rest (sparsity):")
    for m in methods:
        col  = f'{m}_Sparsity'
        f0   = df[col].iloc[0]
        rest = df[col].iloc[1:].mean()
        print(f"  {m:<12}: frame0={f0:.1f}%  rest_mean={rest:.1f}%")

    print(f"\nSaved → results/05_Baselines/table2_baselines_v2.csv")


if __name__ == "__main__":
    main(n_frames=100)