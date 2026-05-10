# experiments/debug/find_best_davis_frame.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from retigate import RetinaCore

DAVIS_ROOT = Path("data/davis/DAVIS/JPEGImages/480p")

# Only check these candidates
CANDIDATES = {
    'koala':          (2,  'koala'),
    'dog':            (4,  'dog'),
    'pigs':           (4,  'pigs'),
    'goat':           (11, 'goat'),
    'sheep':          (13, 'sheep'),
    'motocross-jump': (3,  'motocross-jump'),
}

fig, axes = plt.subplots(
    len(CANDIDATES), 2,
    figsize=(12, 4 * len(CANDIDATES)),
    facecolor='white'
)

for row_idx, (seq_name, (est_roi, _)) in enumerate(
    CANDIDATES.items()
):
    seq_dir = DAVIS_ROOT / seq_name
    if not seq_dir.exists():
        print(f"Missing: {seq_name}")
        continue

    frames = sorted(list(seq_dir.glob("*.jpg")))

    retina = RetinaCore.golden_baseline()
    retina.threshold = 0.10

    # Warm up
    for p in frames[:10]:
        img = cv2.imread(str(p))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retina.process_frame(gray)

    # Find best frame
    best = None
    best_area = 100.0

    for p in frames[10:50]:
        img = cv2.imread(str(p))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        rout = retina.process_frame(gray)
        roi  = retina.get_roi_bbox(
            rout, frame_shape=img.shape
        )
        if roi:
            area = (
                (roi[2]-roi[0])*(roi[3]-roi[1])
                /(W*H)*100
            )
            if area < best_area:
                best_area = area
                best = (p, roi, img, rout['sparsity']*100)

    if best is None:
        print(f"{seq_name}: no ROI found")
        continue

    p, roi, img, sparsity = best
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W    = img.shape[:2]
    x1,y1,x2,y2 = roi

    # Dense panel
    axes[row_idx][0].imshow(img_rgb)
    axes[row_idx][0].set_title(
        f'{seq_name} — Dense\n{p.name}',
        fontsize=10
    )
    axes[row_idx][0].axis('off')

    # Sparse panel
    sparse = (img_rgb * 0.35).astype(np.uint8)
    sparse[y1:y2, x1:x2] = img_rgb[y1:y2, x1:x2]
    cv2.rectangle(sparse,(x1,y1),(x2,y2),(0,255,60),5)
    cv2.rectangle(sparse,
                  (x1-3,y1-3),(x2+3,y2+3),(0,0,0),8)
    cv2.rectangle(sparse,(x1,y1),(x2,y2),(0,255,60),5)

    axes[row_idx][1].imshow(sparse)
    axes[row_idx][1].set_title(
        f'{seq_name} — Sparse\n'
        f'ROI={best_area:.0f}%  sparsity={sparsity:.1f}%',
        fontsize=10
    )
    axes[row_idx][1].axis('off')

    print(f"{seq_name:<20}: "
          f"ROI={best_area:.0f}%  "
          f"frame={p.name}  "
          f"sparsity={sparsity:.1f}%")

plt.suptitle(
    'DAVIS Candidates — Dense vs Sparse',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    "figures/debug_davis_candidates.png",
    dpi=150, bbox_inches='tight'
)
plt.close()
print("\nSaved → figures/debug_davis_candidates.png")