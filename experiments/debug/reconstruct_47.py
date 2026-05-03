# experiments/debug/reconstruct_47.py
# Try every possible way to get 47.9%

import cv2
import numpy as np
from pathlib import Path
from retigate import RetinaCore

DATA_ROOT = Path(
    "data/kitti/data_tracking_image/image_02"
)
sequences = sorted([
    d.name for d in DATA_ROOT.iterdir()
    if d.is_dir()
])

retina     = RetinaCore.golden_baseline()
all_areas  = []
seq_medians = []
seq_means   = []

for seq in sequences:
    img_dir = DATA_ROOT / seq
    if not img_dir.exists(): continue
    seq_areas = []

    for img_path in sorted(img_dir.glob("*.png")):
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
                (roi[2]-roi[0])*(roi[3]-roi[1])
                /(W*H)*100
            )
        seq_areas.append(area)
        all_areas.append(area)

    seq_medians.append(np.median(seq_areas))
    seq_means.append(np.mean(seq_areas))

all_areas   = np.array(all_areas)
active_only = all_areas[all_areas < 99.0]

print("="*60)
print("EVERY POSSIBLE WAY TO COMPUTE ROI AREA")
print("="*60)
print(f"Median of all frames:              "
      f"{np.median(all_areas):.1f}%")
print(f"Mean of all frames:                "
      f"{np.mean(all_areas):.1f}%")
print(f"Median of active frames (<100%):   "
      f"{np.median(active_only):.1f}%")
print(f"Mean of active frames (<100%):     "
      f"{np.mean(active_only):.1f}%")
print(f"Mean of per-seq medians:           "
      f"{np.mean(seq_medians):.1f}%")
print(f"Mean of per-seq means:             "
      f"{np.mean(seq_means):.1f}%")
print(f"Median of per-seq medians:         "
      f"{np.median(seq_medians):.1f}%")
print(f"Median of per-seq means:           "
      f"{np.median(seq_means):.1f}%")
print()
print("Active frame stats:")
print(f"  N active frames:  {len(active_only)}")
print(f"  % of total:       "
      f"{len(active_only)/len(all_areas)*100:.1f}%")
print(f"  Min area:         {active_only.min():.1f}%")
print(f"  Max area:         {active_only.max():.1f}%")
print(f"  Std area:         {active_only.std():.1f}%")