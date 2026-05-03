# experiments/debug/single_box_area_distribution.py
# How often is single box ROI = 100% of frame?

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from retigate import RetinaCore

DATA_ROOT = Path("data/kitti/data_tracking_image/image_02")

sequences = sorted([
    d.name for d in DATA_ROOT.iterdir() if d.is_dir()
])
test_seqs = sequences[7:]

retina = RetinaCore.golden_baseline()
areas  = []
full_frame_count = 0
total = 0

for seq in test_seqs:
    img_dir   = DATA_ROOT / seq
    if not img_dir.exists(): continue
    img_paths = sorted(list(img_dir.glob("*.png")))

    for img_path in img_paths:
        img  = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        rout = retina.process_frame(gray)
        roi  = retina.get_roi_bbox(
            rout, frame_shape=img.shape
        )

        if roi is None:
            area = 100.0
        else:
            area = (
                (roi[2]-roi[0]) * (roi[3]-roi[1])
                / (W * H) * 100
            )

        areas.append(area)
        total += 1
        if area >= 99.0:
            full_frame_count += 1

areas = np.array(areas)

print("="*55)
print("SINGLE BOX AREA DISTRIBUTION — TEST SPLIT")
print("="*55)
print(f"Total frames:          {total}")
print(f"ROI = 100% (no gain):  "
      f"{full_frame_count} "
      f"({full_frame_count/total*100:.1f}%)")
print(f"ROI > 90%:             "
      f"{(areas>90).sum()} "
      f"({(areas>90).mean()*100:.1f}%)")
print(f"ROI > 75%:             "
      f"{(areas>75).sum()} "
      f"({(areas>75).mean()*100:.1f}%)")
print(f"ROI > 50%:             "
      f"{(areas>50).sum()} "
      f"({(areas>50).mean()*100:.1f}%)")
print(f"ROI <= 50%:            "
      f"{(areas<=50).sum()} "
      f"({(areas<=50).mean()*100:.1f}%)")
print()
print(f"Mean area:             {areas.mean():.1f}%")
print(f"Median area:           {np.median(areas):.1f}%")
print(f"Std area:              {areas.std():.1f}%")
print()
print("KEY QUESTION:")
print(f"Frames where RetiGate provides ANY gain "
      f"(ROI < 100%): "
      f"{(areas<99).sum()} "
      f"({(areas<99).mean()*100:.1f}%)")

# Histogram
fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
ax.hist(areas, bins=50, color='#2980B9',
        edgecolor='white', linewidth=0.5)
ax.axvline(np.median(areas), color='#E74C3C',
           lw=2, linestyle='--',
           label=f'Median={np.median(areas):.1f}%')
ax.axvline(50, color='#27AE60', lw=1.5,
           linestyle=':', label='50% threshold')
ax.set_xlabel('Single Box ROI Area (% of frame)',
              fontsize=11)
ax.set_ylabel('Frame Count', fontsize=11)
ax.set_title(
    'Distribution of Single-Box ROI Area\n'
    'KITTI Tracking Test Split (sequences 0007–0020)',
    fontsize=11, fontweight='bold'
)
ax.legend(fontsize=10)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(
    'experiments/debug/debug_area_distribution.png',
    dpi=150, bbox_inches='tight'
)
print("\nSaved histogram.")