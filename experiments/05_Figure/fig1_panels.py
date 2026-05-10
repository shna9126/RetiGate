# experiments/figures/fig1_panels.py
# Outputs two clean image files for Canva assembly

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from retigate import RetinaCore

# ── CONFIG ───────────────────────────────────────────────────
SEQ       = "0006"
FRAME_NUM = 44
WARMUP    = 20

DATA_ROOT = Path("data/kitti/data_tracking_image/image_02")
LBL_DIR   = Path("data/kitti/data_tracking_image/label_02")
OUT       = Path("figures/fig1_parts")
OUT.mkdir(parents=True, exist_ok=True)

C_DENSE_CV = (255, 100, 20)   # orange-red
C_GRN_CV   = (0, 220, 60)     # bright green
C_GOLD     = (255, 215, 0)    # yellow
C_BLK      = (0, 0, 0)

# ── LOAD ─────────────────────────────────────────────────────
img_paths = sorted(list((DATA_ROOT / SEQ).glob("*.png")))
img       = cv2.imread(str(img_paths[FRAME_NUM]))
img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W      = img.shape[:2]
print(f"Frame size: {W}×{H}")

# ── WARM UP RETINA ───────────────────────────────────────────
retina = RetinaCore.golden_baseline()
retina.threshold = 0.10
for p in img_paths[max(0, FRAME_NUM - WARMUP):FRAME_NUM]:
    wf = cv2.imread(str(p))
    if wf is not None:
        retina.process_frame(
            cv2.cvtColor(wf, cv2.COLOR_BGR2GRAY)
        )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rout = retina.process_frame(gray)
roi  = retina.get_roi_bbox(rout, frame_shape=img.shape)

# ── YOLO ─────────────────────────────────────────────────────
print("Running YOLO...")
model       = YOLO('yolo11m.pt')
results     = model.predict(img, verbose=False, conf=0.35)[0]
dense_boxes = (results.boxes.xyxy.cpu().numpy()
               if results.boxes is not None
               else np.zeros((0, 4)))
print(f"Detections: {len(dense_boxes)}")

# ── GT BOXES ─────────────────────────────────────────────────
gt_boxes = []
try:
    gt_df    = pd.read_csv(
        LBL_DIR / f"{SEQ}.txt", sep=' ', header=None
    )
    gt_df    = gt_df[[0, 2, 6, 7, 8, 9]]
    gt_df.columns = ['frame','type','x1','y1','x2','y2']
    frame_gt = gt_df[
        (gt_df['frame'] == FRAME_NUM) &
        (gt_df['type'].isin(['Car','Pedestrian','Cyclist']))
    ]
    for _, row in frame_gt.iterrows():
        gt_boxes.append([row.x1, row.y1, row.x2, row.y2])
    print(f"GT boxes: {len(gt_boxes)}")
except Exception as e:
    print(f"GT warning: {e}")

roi_area = 100.0
if roi:
    x1r, y1r, x2r, y2r = roi
    roi_area = (x2r - x1r) * (y2r - y1r) / (W * H) * 100
    print(f"ROI area: {roi_area:.0f}%")

# ── PANEL A: DENSE ────────────────────────────────────────────
panel_a = img_rgb.copy()
for box in dense_boxes:
    # Black outline
    cv2.rectangle(panel_a,
                  (int(box[0])-2, int(box[1])-2),
                  (int(box[2])+2, int(box[3])+2),
                  C_BLK, 6)
    # Orange-red box
    cv2.rectangle(panel_a,
                  (int(box[0]), int(box[1])),
                  (int(box[2]), int(box[3])),
                  C_DENSE_CV, 3)

# Save as PNG — native KITTI resolution
panel_a_bgr = cv2.cvtColor(panel_a, cv2.COLOR_RGB2BGR)
out_a = OUT / "panel_a_dense.png"
cv2.imwrite(str(out_a), panel_a_bgr)
print(f"Saved → {out_a}  ({W}×{H}px)")

# ── PANEL B: SPARSE ───────────────────────────────────────────
if roi:
    # Darkened background, bright ROI window
    panel_b = (img_rgb * 0.40).astype(np.uint8)
    panel_b[y1r:y2r, x1r:x2r] = img_rgb[y1r:y2r, x1r:x2r]

    # Green ROI box — thick with black outline
    cv2.rectangle(panel_b,
                  (x1r-3, y1r-3), (x2r+3, y2r+3),
                  C_BLK, 10)
    cv2.rectangle(panel_b,
                  (x1r, y1r), (x2r, y2r),
                  C_GRN_CV, 6)

    # GT boxes — yellow with black outline
    for box in gt_boxes:
        cv2.rectangle(panel_b,
                      (int(box[0])-2, int(box[1])-2),
                      (int(box[2])+2, int(box[3])+2),
                      C_BLK, 6)
        cv2.rectangle(panel_b,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      C_GOLD, 3)
else:
    panel_b = img_rgb.copy()

panel_b_bgr = cv2.cvtColor(panel_b, cv2.COLOR_RGB2BGR)
out_b = OUT / "panel_b_sparse.png"
cv2.imwrite(str(out_b), panel_b_bgr)
print(f"Saved → {out_b}  ({W}×{H}px)")

# ── PRINT STATS FOR CANVA TEXT BOXES ─────────────────────────
print("\n" + "="*50)
print("STATS TO TYPE INTO CANVA:")
print("="*50)
print(f"\nPanel A label:  "
      f"{len(dense_boxes)} detections  |  31.8 ms  |  2107 mJ")
print(f"Panel B label:  "
      f"{roi_area:.0f}% area  |  17.4 ms  |  1294 mJ  "
      f"|  92.86% mAP retained")
print(f"\nBoth images:    {W} × {H} px  (KITTI native)")
print("Use same width for both panels in Canva.")
print("="*50)