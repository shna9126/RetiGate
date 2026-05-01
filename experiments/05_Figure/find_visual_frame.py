# experiments/figures/find_visual_frame.py
# Finds frames where ROI overlaps with a car GT box
# Uses your existing KITTI labels

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from retigate import RetinaCore

DATA_ROOT = Path("data/kitti/data_tracking_image")
IMG_DIR   = DATA_ROOT / "image_02"
LBL_DIR   = DATA_ROOT / "label_02"

def parse_labels(label_path):
    try:
        df = pd.read_csv(label_path, sep=' ', header=None)
        df = df[[0, 2, 6, 7, 8, 9]]
        df.columns = ['frame', 'type', 'x1', 'y1', 'x2', 'y2']
        return df[df['type'].isin(['Car', 'Pedestrian', 'Cyclist'])]
    except:
        return pd.DataFrame()

# Sequences known for clean car visibility
test_seqs = ['0001', '0003', '0006', '0007', '0013']

print("Searching for frame where ROI covers a car...\n")

for seq in test_seqs:
    label_file = LBL_DIR / f"{seq}.txt"
    if not label_file.exists():
        continue

    gt_df     = parse_labels(label_file)
    img_paths = sorted(list((IMG_DIR / seq).glob("*.png")))

    # Warmup retina
    retina = RetinaCore.golden_baseline()
    for p in img_paths[:20]:
        wf = cv2.imread(str(p))
        if wf is not None:
            retina.process_frame(
                cv2.cvtColor(wf, cv2.COLOR_BGR2GRAY)
            )

    for i, img_path in enumerate(img_paths[20:80], start=20):
        img  = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        rout = retina.process_frame(gray)
        roi  = retina.get_roi_bbox(
            rout, frame_shape=img.shape,
            max_area_frac=0.5
        )

        if roi is None:
            continue

        roi_area = (roi[2]-roi[0])*(roi[3]-roi[1])/(W*H)*100
        if roi_area > 70 or roi_area < 15:
            continue

        # Check if any car GT box center is inside ROI
        frame_gt = gt_df[gt_df['frame'] == i]
        car_in_roi = False
        car_info   = []

        for _, gt in frame_gt.iterrows():
            cx = (gt.x1 + gt.x2) / 2
            cy = (gt.y1 + gt.y2) / 2
            if (roi[0] <= cx <= roi[2] and
                    roi[1] <= cy <= roi[3]):
                car_in_roi = True
                car_info.append(
                    f"{gt.type}@({cx:.0f},{cy:.0f})"
                )

        if car_in_roi:
            print(
                f"✅ seq={seq} frame={i:04d} "
                f"roi_area={roi_area:.1f}% "
                f"sparsity={rout['sparsity']*100:.1f}% "
                f"cars={car_info}"
            )
            print(
                f"   IMG_PATH: "
                f"data/kitti/data_tracking_image/"
                f"image_02/{seq}/{img_path.name}"
            )
            print()