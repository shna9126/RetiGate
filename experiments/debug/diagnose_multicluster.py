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
        df.columns = ['frame','type','x1','y1','x2','y2']
        return df[df['type'].isin(
            ['Car','Pedestrian','Cyclist']
        )]
    except:
        return pd.DataFrame()

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
    sequences  = sorted([
        d.name for d in IMG_DIR.iterdir() if d.is_dir()
    ])
    test_seqs  = sequences[7:]

    # Failure mode counters
    failure_modes = {
        'single_captured_multi_missed': 0,
        'both_missed':                  0,
        'both_captured':                0,
        'single_missed_multi_captured': 0,
    }

    # For missed objects — why?
    miss_reasons = {
        'no_clusters_returned':   0,
        'cluster_too_small':      0,
        'object_between_clusters':0,
        'fallback_triggered':     0,
    }

    total_gt = 0

    for seq in test_seqs[:3]:  # first 3 seqs for speed
        label_file = LBL_DIR / f"{seq}.txt"
        if not label_file.exists(): continue

        retina    = RetinaCore.golden_baseline()
        gt_df     = parse_labels(label_file)
        img_paths = sorted(
            list((IMG_DIR / seq).glob("*.png"))
        )[:100]  # first 100 frames only

        for i, img_path in enumerate(img_paths):
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
                rout,
                frame_shape=img.shape,
                max_area_frac=0.15,
                max_clusters=5
            )

            for _, gt in frame_gt.iterrows():
                total_gt += 1
                gt_box = [gt.x1, gt.y1, gt.x2, gt.y2]

                s_hit = gt_contained_single(
                    gt_box, roi_single
                )
                m_hit = gt_contained_multi(
                    gt_box, clusters
                )

                if s_hit and not m_hit:
                    failure_modes[
                        'single_captured_multi_missed'
                    ] += 1

                    # WHY was it missed?
                    if len(clusters) == 0:
                        miss_reasons[
                            'no_clusters_returned'
                        ] += 1
                    else:
                        # Check if gt_box is between clusters
                        # i.e. active mask has no pixel there
                        active = rout['active_mask']
                        gt_region = active[
                            int(gt.y1):int(gt.y2),
                            int(gt.x1):int(gt.x2)
                        ]
                        if gt_region.sum() == 0:
                            miss_reasons[
                                'object_between_clusters'
                            ] += 1
                            print(
                                f"\nSEQ {seq} Frame {i}: "
                                f"GT box [{gt.x1:.0f},"
                                f"{gt.y1:.0f},"
                                f"{gt.x2:.0f},"
                                f"{gt.y2:.0f}] "
                                f"has ZERO active pixels"
                            )
                            print(
                                f"  Clusters: {clusters}"
                            )
                            print(
                                f"  Single ROI: {roi_single}"
                            )
                        else:
                            miss_reasons[
                                'cluster_too_small'
                            ] += 1
                            print(
                                f"\nSEQ {seq} Frame {i}: "
                                f"GT box [{gt.x1:.0f},"
                                f"{gt.y1:.0f},"
                                f"{gt.x2:.0f},"
                                f"{gt.y2:.0f}] "
                                f"HAS active pixels but "
                                f"not in any cluster"
                            )
                            print(
                                f"  Active pixels in GT: "
                                f"{gt_region.sum()}"
                            )
                            print(
                                f"  Clusters: {clusters}"
                            )
                            print(
                                f"  Single ROI: {roi_single}"
                            )

                elif s_hit and m_hit:
                    failure_modes['both_captured'] += 1
                elif not s_hit and not m_hit:
                    failure_modes['both_missed'] += 1
                else:
                    failure_modes[
                        'single_missed_multi_captured'
                    ] += 1

    print("\n" + "="*60)
    print("FAILURE MODE ANALYSIS")
    print("="*60)
    print(f"Total GT objects evaluated: {total_gt}")
    print()
    for k, v in failure_modes.items():
        pct = v/total_gt*100
        print(f"  {k:<40}: {v:5d} ({pct:.1f}%)")
    print()
    print("MISS REASONS (single hit, multi missed):")
    missed_total = failure_modes[
        'single_captured_multi_missed'
    ]
    for k, v in miss_reasons.items():
        if missed_total > 0:
            pct = v/missed_total*100
            print(f"  {k:<40}: {v:5d} ({pct:.1f}%)")

if __name__ == "__main__":
    main()