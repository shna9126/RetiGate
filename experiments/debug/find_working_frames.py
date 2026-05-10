# Add to find_working_frames.py or run separately

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from retigate import RetinaCore

DAVIS_ROOT = Path("data/davis/DAVIS/JPEGImages/480p")
OUT        = Path("figures")

seq_dir = DAVIS_ROOT / "sheep"
frames  = sorted(list(seq_dir.glob("*.jpg")))
print(f"Sheep: {len(frames)} frames")

fig, axes = plt.subplots(2, 4, figsize=(16, 6),
                          facecolor='white')
axes = axes.flatten()

test_frames = [10, 12, 14, 16, 18, 20, 22, 24]

for idx, tf in enumerate(test_frames):
    if tf >= len(frames): break

    retina = RetinaCore.golden_baseline()
    retina.threshold = 0.10

    for p in frames[max(0, tf-8):tf]:
        img = cv2.imread(str(p))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retina.process_frame(gray)

    img  = cv2.imread(str(frames[tf]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    rout = retina.process_frame(gray)
    roi  = retina.get_roi_bbox(rout, frame_shape=img.shape)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if roi:
        x1,y1,x2,y2 = roi
        area = (x2-x1)*(y2-y1)/(W*H)*100
        vis  = (img_rgb*0.40).astype(np.uint8)
        vis[y1:y2,x1:x2] = img_rgb[y1:y2,x1:x2]
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,60),4)
        title = f"F{tf} ROI={area:.0f}%"
    else:
        vis   = img_rgb
        title = f"F{tf} ROI=100%"

    axes[idx].imshow(vis)
    axes[idx].set_title(title, fontsize=10)
    axes[idx].axis('off')

plt.suptitle("Sheep — warmup=8", fontsize=12,
             fontweight='bold')
plt.tight_layout()
plt.savefig(str(OUT/"debug_sheep_grid.png"),
            dpi=120, bbox_inches='tight')
plt.close()
print("Saved debug_sheep_grid.png")