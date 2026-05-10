import cv2, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from retigate import RetinaCore

MB_ROOT = Path("data/middlebury/other-data")
OUT     = Path("figures")

for seq_name in ['Grove2', 'Grove3', 'Walking',
                 'Urban2', 'Hydrangea']:
    seq_dir = MB_ROOT / seq_name
    frames  = sorted(list(seq_dir.glob("frame*.png")))
    if not frames: continue

    retina = RetinaCore.golden_baseline()
    retina.threshold = 0.10

    best_vis, best_area, best_name = None, 100.0, ""

    for i, p in enumerate(frames):
        img  = cv2.imread(str(p))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        rout = retina.process_frame(gray)
        roi  = retina.get_roi_bbox(
            rout, frame_shape=img.shape
        )
        if roi:
            x1,y1,x2,y2 = roi
            area = (x2-x1)*(y2-y1)/(W*H)*100
            if area < best_area:
                best_area = area
                img_rgb   = cv2.cvtColor(
                    img, cv2.COLOR_BGR2RGB
                )
                vis = (img_rgb*0.40).astype(np.uint8)
                vis[y1:y2,x1:x2] = img_rgb[y1:y2,x1:x2]
                cv2.rectangle(
                    vis,(x1,y1),(x2,y2),(0,255,60),4
                )
                best_vis  = (img_rgb, vis)
                best_name = p.name

    if best_vis:
        fig, axes = plt.subplots(1,2,figsize=(10,3),
                                  facecolor='white')
        axes[0].imshow(best_vis[0])
        axes[0].set_title(f"{seq_name} Dense",
                          fontsize=10)
        axes[0].axis('off')
        axes[1].imshow(best_vis[1])
        axes[1].set_title(
            f"{seq_name} Sparse\n"
            f"ROI={best_area:.0f}%  {best_name}",
            fontsize=10
        )
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(
            str(OUT/f"debug_mb_{seq_name}.png"),
            dpi=120, bbox_inches='tight'
        )
        plt.close()
        print(f"{seq_name}: ROI={best_area:.0f}%  "
              f"{best_name}")