# experiments/debug/find_47_percent.py
# Reproduce the exact 47.9% number from FINAL_NUMBERS.md

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

for split_name, seqs in [
    ("ALL_21",  sequences),
    ("VAL_7",   sequences[:7]),
    ("TEST_14", sequences[7:]),
    ("ONLY_0006_0012_0015_0020",
     ['0006','0012','0015','0020']),
]:
    retina = RetinaCore.golden_baseline()
    areas  = []

    for seq in seqs:
        img_dir = DATA_ROOT / seq
        if not img_dir.exists(): continue
        for img_path in sorted(
            img_dir.glob("*.png")
        ):
            img  = cv2.imread(str(img_path))
            gray = cv2.cvtColor(
                img, cv2.COLOR_BGR2GRAY
            )
            H, W = gray.shape
            rout = retina.process_frame(gray)
            roi  = retina.get_roi_bbox(
                rout, frame_shape=img.shape
            )
            if roi is None:
                areas.append(100.0)
            else:
                areas.append(
                    (roi[2]-roi[0])
                    *(roi[3]-roi[1])
                    /(W*H)*100
                )

    areas = np.array(areas)
    print(f"{split_name:<35}: "
          f"median={np.median(areas):.1f}%  "
          f"mean={np.mean(areas):.1f}%")