import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from retigate import RetinaCore

DATA_ROOT = Path("data/kitti/data_tracking_image/image_02")
OUT       = Path("figures")

def iou(a, b):
    """IoU between two (x1,y1,x2,y2) boxes."""
    if a is None or b is None: return 0.0
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    a_area = (a[2]-a[0]) * (a[3]-a[1])
    b_area = (b[2]-b[0]) * (b[3]-b[1])
    union  = a_area + b_area - inter
    return inter/union if union > 0 else 0.0

def centroid_displacement(a, b):
    """Pixel displacement of ROI centroid."""
    if a is None or b is None: return None
    ca = ((a[0]+a[2])/2, (a[1]+a[3])/2)
    cb = ((b[0]+b[2])/2, (b[1]+b[3])/2)
    return np.sqrt((ca[0]-cb[0])**2 + (ca[1]-cb[1])**2)

def main():
    test_seqs = ['0007','0008','0009','0010',
                 '0011','0012','0013','0014']

    # Compare different lambda values
    lambdas = [0.05, 0.10, 0.20, 0.50]  # IIR memory
    results = {lam: {'ious': [], 'disps': []}
               for lam in lambdas}

    for lam in lambdas:
        for seq in test_seqs:
            img_dir = DATA_ROOT / seq
            if not img_dir.exists(): continue

            retina = RetinaCore.golden_baseline()
            retina.amacrine_decay = lam  # vary IIR memory

            img_paths = sorted(list(img_dir.glob("*.png")))
            prev_roi  = None

            for img_path in tqdm(
                img_paths, desc=f"λ={lam} seq={seq}",
                leave=False
            ):
                img  = cv2.imread(str(img_path))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rout = retina.process_frame(gray)
                curr_roi = retina.get_roi_bbox(
                    rout, frame_shape=img.shape
                )

                if prev_roi is not None:
                    results[lam]['ious'].append(
                        iou(prev_roi, curr_roi)
                    )
                    d = centroid_displacement(
                        prev_roi, curr_roi
                    )
                    if d is not None:
                        results[lam]['disps'].append(d)

                prev_roi = curr_roi

    # Print summary
    print("\n" + "="*55)
    print("TEMPORAL STABILITY BY λ (IIR DECAY)")
    print("="*55)
    print(f"{'λ':>6} | {'Mean IoU':>10} | "
          f"{'Centroid Δ (px)':>16}")
    print("-"*40)
    for lam in lambdas:
        ious  = results[lam]['ious']
        disps = results[lam]['disps']
        print(f"{lam:>6} | "
              f"{np.mean(ious)*100:>9.1f}% | "
              f"{np.mean(disps):>15.1f}")

    # Figure: temporal IoU distribution for λ=0.10
    fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                              facecolor='white')

    # Left: IoU distribution across lambdas
    ax = axes[0]
    colors = ['#E74C3C','#2980B9','#27AE60','#F39C12']
    for lam, col in zip(lambdas, colors):
        ious = results[lam]['ious']
        ax.hist(ious, bins=30, alpha=0.6,
                color=col, label=f'λ={lam}',
                density=True)
    ax.axvline(0.85, color='black', linestyle='--',
               lw=1.2, label='Stability threshold')
    ax.set_xlabel('Temporal IoU (ROI$_t$ vs ROI$_{t+1}$)',
                  fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('ROI Temporal Stability by IIR λ',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines[['top','right']].set_visible(False)

    # Right: Mean IoU vs lambda
    ax = axes[1]
    mean_ious = [
        np.mean(results[lam]['ious'])*100
        for lam in lambdas
    ]
    mean_disps = [
        np.mean(results[lam]['disps'])
        for lam in lambdas
    ]
    ax2 = ax.twinx()
    ax.plot(lambdas, mean_ious, 'o-',
            color='#2980B9', lw=2, label='Mean IoU (%)')
    ax2.plot(lambdas, mean_disps, 's--',
             color='#E74C3C', lw=2,
             label='Centroid Δ (px)')
    ax.axvline(0.10, color='gray', linestyle=':',
               lw=1.5, label='Chosen λ=0.10')
    ax.set_xlabel('IIR Decay λ', fontsize=10)
    ax.set_ylabel('Mean Temporal IoU (%)',
                  fontsize=10, color='#2980B9')
    ax2.set_ylabel('Centroid Displacement (px)',
                   fontsize=10, color='#E74C3C')
    ax.set_title('λ Effect on Temporal Stability',
                 fontsize=11, fontweight='bold')
    ax.spines[['top']].set_visible(False)
    ax.set_xticks(lambdas)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        p = OUT / f"fig7_temporal_stability.{fmt}"
        plt.savefig(str(p), dpi=300,
                    bbox_inches='tight',
                    facecolor='white')
        print(f"Saved → {p}")
    plt.close()

if __name__ == "__main__":
    main()