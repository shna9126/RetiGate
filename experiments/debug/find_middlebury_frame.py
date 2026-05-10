# Run this to find the best Middlebury frame
# experiments/debug/find_middlebury_frame.py

import cv2
import numpy as np
from pathlib import Path
from retigate import RetinaCore

# Adjust path to your Middlebury location
MIDDLEBURY_ROOT = Path("data/middlebury/other-data")

# Find all image sequences
sequences = []
for ext in ['*.png', '*.jpg', '*.ppm']:
    sequences.extend(list(MIDDLEBURY_ROOT.rglob(ext)))

print(f"Found {len(sequences)} Middlebury images")

retina = RetinaCore.golden_baseline()
retina.threshold = 0.10

results = []

# Group by parent directory (sequence)
from collections import defaultdict
by_seq = defaultdict(list)
for p in sequences:
    by_seq[p.parent].append(p)

for seq_dir, frames in sorted(by_seq.items()):
    frames = sorted(frames)
    if len(frames) < 5:
        continue

    retina.reset_memory()
    areas = []

    for p in frames[:20]:  # first 20 frames
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        rout = retina.process_frame(gray)
        roi  = retina.get_roi_bbox(
            rout, frame_shape=img.shape
        )
        if roi:
            area = (
                (roi[2]-roi[0]) * (roi[3]-roi[1])
                / (W*H) * 100
            )
            areas.append((area, p, roi))

    if areas:
        # Find frame with tightest ROI
        best = min(areas, key=lambda x: x[0])
        area, path, roi = best
        print(f"{seq_dir.name:<25} "
              f"best ROI={area:.0f}%  "
              f"frame={path.name}")
        results.append((area, seq_dir.name, path, roi))

if results:
    results.sort()
    print(f"\nTop 5 tightest Middlebury ROIs:")
    for area, seq, path, roi in results[:5]:
        print(f"  {seq}: {area:.0f}% — {path}")