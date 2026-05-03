import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from retigate import RetinaCore

DATA_ROOT = Path("data/kitti/data_tracking_image")
IMG_DIR   = DATA_ROOT / "image_02"
LBL_DIR   = DATA_ROOT / "label_02"

# Key insight: fallback threshold controls
# safety vs efficiency trade-off explicitly
FALLBACK_THRESHOLDS = [0.50, 0.60, 0.70, 0.80,
                       0.90, 1.00]
# 0.50 = fallback if ROI > 50% (aggressive safety)
# 1.00 = never fallback (current behaviour)

def parse_labels(label_path):
    try:
        df = pd.read_csv(label_path,
                         sep=' ', header=None)
        df = df[[0, 2, 6, 7, 8, 9]]
        df.columns = [
            'frame','type','x1','y1','x2','y2'
        ]
        return df[df['type'].isin(
            ['Car','Pedestrian','Cyclist']
        )]
    except:
        return pd.DataFrame()

def gt_contained(gt_box, roi, margin=5):
    if roi is None: return False
    return (gt_box[0] >= roi[0]-margin and
            gt_box[1] >= roi[1]-margin and
            gt_box[2] <= roi[2]+margin and
            gt_box[3] <= roi[3]+margin)

def main():
    sequences = sorted([
        d.name for d in IMG_DIR.iterdir()
        if d.is_dir()
    ])
    test_seqs = sequences[7:]
    results   = []

    for thresh in FALLBACK_THRESHOLDS:
        recalls      = []
        areas        = []
        fallback_pct = []

        for seq in test_seqs:
            label_file = LBL_DIR / f"{seq}.txt"
            if not label_file.exists(): continue

            retina    = RetinaCore.golden_baseline()
            gt_df     = parse_labels(label_file)
            img_paths = sorted(
                list((IMG_DIR / seq).glob("*.png"))
            )

            for i, img_path in enumerate(
                tqdm(img_paths,
                     desc=f"thresh={thresh:.2f} "
                          f"seq={seq}",
                     leave=False)
            ):
                frame_gt = gt_df[gt_df['frame'] == i]
                img  = cv2.imread(str(img_path))
                gray = cv2.cvtColor(
                    img, cv2.COLOR_BGR2GRAY
                )
                H, W = gray.shape
                rout = retina.process_frame(gray)
                roi  = retina.get_roi_bbox(
                    rout, frame_shape=img.shape
                )

                # Apply fallback
                fell_back = False
                if roi is not None:
                    area = (
                        (roi[2]-roi[0])
                        *(roi[3]-roi[1])
                        /(W*H)
                    )
                    if area > thresh:
                        effective_roi = (0, 0, W, H)
                        fell_back = True
                    else:
                        effective_roi = roi
                else:
                    effective_roi = (0, 0, W, H)
                    fell_back = True

                fallback_pct.append(
                    1 if fell_back else 0
                )
                eff_area = (
                    (effective_roi[2]-effective_roi[0])
                    *(effective_roi[3]-effective_roi[1])
                    /(W*H)*100
                )
                areas.append(eff_area)

                for _, gt in frame_gt.iterrows():
                    gt_box = [
                        gt.x1, gt.y1, gt.x2, gt.y2
                    ]
                    recalls.append(
                        1 if gt_contained(
                            gt_box, effective_roi
                        ) else 0
                    )

        fb_rate    = np.mean(fallback_pct)*100
        mean_area  = np.mean(areas)
        recall     = np.mean(recalls)*100
        # Effective speedup estimate
        # Active frames get 1.61×, fallback gets 1.0×
        active_frac = 1 - fb_rate/100
        eff_speedup = (
            active_frac * 1.61
            + (1-active_frac) * 1.0
        )

        results.append({
            'Fallback threshold': thresh,
            'Recall (%)':         recall,
            'Mean area (%)':      mean_area,
            'Fallback rate (%)':  fb_rate,
            'Area saved (%)':     100 - mean_area,
            'Eff. speedup (×)':   eff_speedup,
        })

    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("FALLBACK ANALYSIS — KITTI TEST SPLIT")
    print("="*70)
    print(df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    ))
    df.to_csv(
        'results/table_fallback_analysis.csv',
        index=False
    )

    # Key operating points
    print("\n\nKEY OPERATING POINTS FOR PAPER:")
    print("-"*50)
    for _, row in df.iterrows():
        print(
            f"thresh={row['Fallback threshold']:.2f}: "
            f"recall={row['Recall (%)']:.1f}%  "
            f"speedup={row['Eff. speedup (×)']:.2f}×  "
            f"fallback={row['Fallback rate (%)']:.1f}%"
        )

if __name__ == "__main__":
    main()