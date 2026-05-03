import cv2
import numpy as np
from pathlib import Path
from retigate import RetinaCore

DATA_ROOT = Path("data/kitti/data_tracking_image/image_02")

sequences = sorted([
    d.name for d in DATA_ROOT.iterdir() if d.is_dir()
])

print("="*65)
print(f"{'Seq':<8} {'Split':<6} {'Median%':>8} "
      f"{'Mean%':>7} {'Full%':>8} {'N':>6}")
print("="*65)

for seq in sequences:
    img_dir = DATA_ROOT / seq
    if not img_dir.exists(): continue

    split  = "VAL" if seq < "0007" else "TEST"
    retina = RetinaCore.golden_baseline()
    areas  = []

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
                (roi[2]-roi[0])*(roi[3]-roi[1])
                /(W*H)*100
            )
        areas.append(area)

    areas = np.array(areas)
    full_pct = (areas >= 99.0).mean() * 100

    print(f"{seq:<8} {split:<6} "
          f"{np.median(areas):>7.1f}% "
          f"{areas.mean():>6.1f}% "
          f"{full_pct:>7.1f}% "
          f"{len(areas):>6}")