# experiments/debug/visualise_cluster_miss.py
# Run this to see exactly what is happening geometrically

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from retigate import RetinaCore
import pandas as pd

DATA_ROOT = Path("data/kitti/data_tracking_image")
IMG_DIR   = DATA_ROOT / "image_02"
LBL_DIR   = DATA_ROOT / "label_02"

def parse_labels(label_path):
    df = pd.read_csv(label_path, sep=' ', header=None)
    df = df[[0, 2, 6, 7, 8, 9]]
    df.columns = ['frame','type','x1','y1','x2','y2']
    return df[df['type'].isin(
        ['Car','Pedestrian','Cyclist']
    )]

def gt_contained_single(gt_box, roi, margin=5):
    if roi is None: return False
    return (gt_box[0] >= roi[0]-margin and
            gt_box[1] >= roi[1]-margin and
            gt_box[2] <= roi[2]+margin and
            gt_box[3] <= roi[3]+margin)

def gt_contained_multi(gt_box, clusters, margin=5):
    for roi in clusters:
        if (gt_box[0] >= roi[0]-margin and
            gt_box[1] >= roi[1]-margin and
            gt_box[2] <= roi[2]+margin and
            gt_box[3] <= roi[3]+margin):
            return True
    return False

def main():
    sequences = sorted([
        d.name for d in IMG_DIR.iterdir() if d.is_dir()
    ])
    test_seqs = sequences[7:]
    
    found = 0
    target = 6  # find 6 failure examples

    fig, axes = plt.subplots(
        target, 3,
        figsize=(15, 4 * target),
        facecolor='white'
    )

    for seq in test_seqs:
        if found >= target: break
        label_file = LBL_DIR / f"{seq}.txt"
        if not label_file.exists(): continue

        retina    = RetinaCore.golden_baseline()
        gt_df     = parse_labels(label_file)
        img_paths = sorted(
            list((IMG_DIR / seq).glob("*.png"))
        )

        for i, img_path in enumerate(img_paths):
            if found >= target: break
            frame_gt = gt_df[gt_df['frame'] == i]
            if frame_gt.empty: continue

            img  = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            H, W = gray.shape

            rout = retina.process_frame(gray)
            roi_single = retina.get_roi_bbox(
                rout, frame_shape=img.shape
            )
            clusters = retina.get_roi_clusters(
                rout, frame_shape=img.shape,
                max_area_frac=0.15, max_clusters=5
            )

            for _, gt in frame_gt.iterrows():
                if found >= target: break
                gt_box = [gt.x1, gt.y1, gt.x2, gt.y2]
                s_hit  = gt_contained_single(
                    gt_box, roi_single
                )
                m_hit  = gt_contained_multi(
                    gt_box, clusters
                )

                if s_hit and not m_hit:
                    active = rout['active_mask']
                    gt_region = active[
                        int(gt.y1):int(gt.y2),
                        int(gt.x1):int(gt.x2)
                    ]
                    if gt_region.sum() == 0:
                        continue  # skip zero-pixel cases

                    row = found
                    img_rgb = cv2.cvtColor(
                        img, cv2.COLOR_BGR2RGB
                    )

                    # --- Col 0: active mask ---
                    ax = axes[row][0]
                    ax.imshow(
                        active, cmap='hot',
                        vmin=0, vmax=1
                    )
                    # GT box on mask
                    rect = patches.Rectangle(
                        (gt.x1, gt.y1),
                        gt.x2-gt.x1, gt.y2-gt.y1,
                        linewidth=2,
                        edgecolor='cyan',
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.set_title(
                        f'Active Mask\n'
                        f'Seq={seq} F={i}\n'
                        f'GT active px: '
                        f'{int(gt_region.sum())}',
                        fontsize=8
                    )
                    ax.axis('off')

                    # --- Col 1: single box ---
                    ax = axes[row][1]
                    ax.imshow(img_rgb)
                    if roi_single:
                        rect = patches.Rectangle(
                            (roi_single[0], roi_single[1]),
                            roi_single[2]-roi_single[0],
                            roi_single[3]-roi_single[1],
                            linewidth=2.5,
                            edgecolor='#00FF00',
                            facecolor='none',
                            label='Single ROI'
                        )
                        ax.add_patch(rect)
                    # GT
                    rect = patches.Rectangle(
                        (gt.x1, gt.y1),
                        gt.x2-gt.x1, gt.y2-gt.y1,
                        linewidth=2,
                        edgecolor='yellow',
                        facecolor='none',
                        label='GT'
                    )
                    ax.add_patch(rect)
                    area_pct = 0
                    if roi_single:
                        area_pct = (
                            (roi_single[2]-roi_single[0])
                            *(roi_single[3]-roi_single[1])
                            /(W*H)*100
                        )
                    ax.set_title(
                        f'Single Box ✓ HITS\n'
                        f'ROI area: {area_pct:.0f}%\n'
                        f'Green=ROI  Yellow=GT',
                        fontsize=8, color='green'
                    )
                    ax.axis('off')

                    # --- Col 2: multi cluster ---
                    ax = axes[row][2]
                    ax.imshow(img_rgb)
                    colors_mc = [
                        '#FF6B6B','#4ECDC4','#45B7D1',
                        '#96CEB4','#FFEAA7'
                    ]
                    for ci, (cx1,cy1,cx2,cy2) in \
                            enumerate(clusters):
                        rect = patches.Rectangle(
                            (cx1, cy1),
                            cx2-cx1, cy2-cy1,
                            linewidth=2,
                            edgecolor=colors_mc[
                                ci % len(colors_mc)
                            ],
                            facecolor='none'
                        )
                        ax.add_patch(rect)
                    # GT
                    rect = patches.Rectangle(
                        (gt.x1, gt.y1),
                        gt.x2-gt.x1, gt.y2-gt.y1,
                        linewidth=2,
                        edgecolor='yellow',
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Print geometry to console
                    print(f"\nSeq={seq} Frame={i}")
                    print(f"GT box:     "
                          f"[{gt.x1:.0f},{gt.y1:.0f},"
                          f"{gt.x2:.0f},{gt.y2:.0f}] "
                          f"size={gt.x2-gt.x1:.0f}x"
                          f"{gt.y2-gt.y1:.0f}")
                    print(f"Single ROI: {roi_single}")
                    print(f"Clusters ({len(clusters)}):")
                    for ci, cl in enumerate(clusters):
                        print(f"  [{ci}] {cl}  "
                              f"size={cl[2]-cl[0]}x"
                              f"{cl[3]-cl[1]}")
                    print(f"Active px in GT box: "
                          f"{int(gt_region.sum())}")

                    ax.set_title(
                        f'Multi-Cluster ✗ MISSES\n'
                        f'{len(clusters)} clusters  '
                        f'Yellow=GT',
                        fontsize=8, color='red'
                    )
                    ax.axis('off')
                    found += 1

    plt.suptitle(
        'Failure Mode: Single box hits, Multi-cluster misses\n'
        'cluster_too_small — geometry analysis',
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    out = Path("experiments/debug/debug_cluster_miss.png")
    plt.savefig(str(out), dpi=150, bbox_inches='tight',
                facecolor='white')
    print(f"\nSaved → {out}")
    plt.close()

if __name__ == "__main__":
    main()