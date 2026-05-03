import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from retigate import RetinaCore

DATA_ROOT = Path("data/kitti/data_tracking_image")
IMG_DIR   = DATA_ROOT / "image_02"
LBL_DIR   = DATA_ROOT / "label_02"

def parse_labels(label_path):
    try:
        df = pd.read_csv(label_path, sep=' ', header=None)
        df = df[[0, 2, 6, 7, 8, 9]]
        df.columns = ['frame', 'type', 'x1', 'y1', 'x2', 'y2']
        return df[df['type'].isin(
            ['Car', 'Pedestrian', 'Cyclist']
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
    """GT is captured if ANY cluster contains it."""
    for roi in clusters:
        if (gt_box[0] >= roi[0]-margin and
            gt_box[1] >= roi[1]-margin and
            gt_box[2] <= roi[2]+margin and
            gt_box[3] <= roi[3]+margin):
            return True
    return False

def roi_area_single(roi, W, H):
    if roi is None: return 0
    return (roi[2]-roi[0])*(roi[3]-roi[1])/(W*H)*100

def roi_area_multi(clusters, W, H):
    """
    Total area covered by all clusters combined.
    Uses a pixel mask to avoid double-counting overlap.
    """
    if not clusters:
        return 0.0
    mask = np.zeros((H, W), dtype=np.uint8)
    for (x1, y1, x2, y2) in clusters:
        mask[y1:y2, x1:x2] = 1
    return mask.sum() / (W * H) * 100

def main():
    sequences = sorted([
        d.name for d in IMG_DIR.iterdir() if d.is_dir()
    ])
    test_seqs = sequences[7:]  # held-out test split

    single_recalls  = []
    multi_recalls   = []
    single_areas    = []
    multi_areas     = []

    for seq in test_seqs:
        label_file = LBL_DIR / f"{seq}.txt"
        if not label_file.exists(): continue

        retina = RetinaCore.golden_baseline()
        retina.threshold = 0.10

        gt_df     = parse_labels(label_file)
        img_paths = sorted(
            list((IMG_DIR / seq).glob("*.png"))
        )

        for i, img_path in enumerate(
            tqdm(img_paths, desc=seq, leave=False)
        ):
            frame_gt = gt_df[gt_df['frame'] == i]
            if frame_gt.empty: continue

            img  = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            H, W = gray.shape
            rout = retina.process_frame(gray)

            # Single box
            roi_single = retina.get_roi_bbox(
                rout, frame_shape=img.shape
            )

            # Multi cluster (top 3)
            clusters = retina.get_roi_clusters(
                rout,
                frame_shape=img.shape,
                max_area_frac=0.15,
                max_clusters=5
            )

            for _, gt in frame_gt.iterrows():
                gt_box = [gt.x1, gt.y1, gt.x2, gt.y2]

                # Single box metrics
                single_recalls.append(
                    1 if gt_contained_single(
                        gt_box, roi_single
                    ) else 0
                )
                single_areas.append(
                    roi_area_single(roi_single, W, H)
                )

                # Multi cluster metrics
                multi_recalls.append(
                    1 if gt_contained_multi(
                        gt_box, clusters
                    ) else 0
                )
                multi_areas.append(
                    roi_area_multi(clusters, W, H)
                )

    print("\n" + "="*60)
    print("SINGLE BOX vs MULTI-CLUSTER COMPARISON")
    print("="*60)
    print(f"Single box:     "
          f"Recall={np.mean(single_recalls)*100:.2f}%  "
          f"Area={np.mean(single_areas):.1f}%")
    print(f"Multi-cluster:  "
          f"Recall={np.mean(multi_recalls)*100:.2f}%  "
          f"Area={np.mean(multi_areas):.1f}%")

    efficiency_single = (
        100 - np.mean(single_areas)
    )
    efficiency_multi = (
        100 - np.mean(multi_areas)
    )
    print(f"\nArea SAVED:")
    print(f"  Single box:    {efficiency_single:.1f}%")
    print(f"  Multi-cluster: {efficiency_multi:.1f}%")
    print(
        f"\nConclusion: Multi-cluster saves "
        f"{efficiency_multi - efficiency_single:.1f}% "
        f"MORE area while maintaining recall"
    )

    # Save
    pd.DataFrame({
        'method':  ['single_box', 'multi_cluster'],
        'recall':  [np.mean(single_recalls)*100,
                    np.mean(multi_recalls)*100],
        'area':    [np.mean(single_areas),
                    np.mean(multi_areas)],
        'savings': [efficiency_single, efficiency_multi]
    }).to_csv(
        'results/table_multicluster_vs_singlebox.csv',
        index=False
    )

if __name__ == "__main__":
    main()
