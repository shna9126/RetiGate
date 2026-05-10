# experiments/debug/find_working_frames.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from retigate import RetinaCore

DAVIS_ROOT   = Path("data/davis/DAVIS/JPEGImages/480p")
MIDDLEBURY   = Path("data/middlebury/other-data/Walking")
OUT          = Path("figures")

# ── TEST GOAT FRAMES ────────────────────────────────────────
print("Testing goat frames...")
goat_dir = DAVIS_ROOT / "goat"
frames   = sorted(list(goat_dir.glob("*.jpg")))

fig, axes = plt.subplots(3, 4, figsize=(16, 9),
                          facecolor='white')
axes = axes.flatten()
idx  = 0

for test_frame in [10, 12, 14, 16, 18, 20,
                   22, 24, 26, 28, 30, 32]:
    if test_frame >= len(frames) or idx >= 12:
        break

    retina = RetinaCore.golden_baseline()
    retina.threshold = 0.10

    # Warmup with 8 frames before target
    warmup = 8
    for p in frames[max(0,test_frame-warmup):test_frame]:
        img = cv2.imread(str(p))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retina.process_frame(gray)

    # Target frame
    img  = cv2.imread(str(frames[test_frame]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    rout = retina.process_frame(gray)
    roi  = retina.get_roi_bbox(
        rout, frame_shape=img.shape
    )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if roi:
        x1,y1,x2,y2 = roi
        area = (x2-x1)*(y2-y1)/(W*H)*100
        vis  = (img_rgb*0.40).astype(np.uint8)
        vis[y1:y2,x1:x2] = img_rgb[y1:y2,x1:x2]
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,60),4)
        title = f"F{test_frame} ROI={area:.0f}%"
    else:
        vis   = img_rgb
        title = f"F{test_frame} ROI=100%"

    axes[idx].imshow(vis)
    axes[idx].set_title(title, fontsize=9)
    axes[idx].axis('off')
    idx += 1

plt.suptitle("Goat — warmup=8, frames 10-32",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(str(OUT/"debug_goat_grid.png"),
            dpi=120, bbox_inches='tight')
plt.close()
print("Saved debug_goat_grid.png")

# ── TEST MIDDLEBURY FRAMES ───────────────────────────────────
print("Testing Middlebury frames...")
mb_frames = sorted(list(MIDDLEBURY.glob("frame*.png")))
print(f"Middlebury frames: {[p.name for p in mb_frames]}")

fig, axes = plt.subplots(2, 4, figsize=(16, 6),
                          facecolor='white')
axes = axes.flatten()

for idx, (test_frame, warmup) in enumerate([
    (0, 0), (1, 0), (1, 1), (2, 0),
    (2, 1), (2, 2), (3, 0), (3, 2),
]):
    if test_frame >= len(mb_frames) or idx >= 8:
        break

    retina = RetinaCore.golden_baseline()
    retina.threshold = 0.10

    for p in mb_frames[max(0,test_frame-warmup):
                        test_frame]:
        img = cv2.imread(str(p))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retina.process_frame(gray)

    img  = cv2.imread(str(mb_frames[test_frame]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    rout = retina.process_frame(gray)
    roi  = retina.get_roi_bbox(
        rout, frame_shape=img.shape
    )
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if roi:
        x1,y1,x2,y2 = roi
        area = (x2-x1)*(y2-y1)/(W*H)*100
        vis  = (img_rgb*0.45).astype(np.uint8)
        vis[y1:y2,x1:x2] = img_rgb[y1:y2,x1:x2]
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,60),4)
        title = (f"F={test_frame} W={warmup}\n"
                 f"ROI={area:.0f}%")
    else:
        vis   = img_rgb
        title = f"F={test_frame} W={warmup} 100%"

    axes[idx].imshow(vis)
    axes[idx].set_title(title, fontsize=9)
    axes[idx].axis('off')

plt.suptitle("Middlebury Walking — frame/warmup combos",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(str(OUT/"debug_middlebury_grid.png"),
            dpi=120, bbox_inches='tight')
plt.close()
print("Saved debug_middlebury_grid.png")