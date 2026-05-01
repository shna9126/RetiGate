# experiments/figures/fig4.py
# Qualitative results grid — 3 scenes × 2 columns

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from retigate import RetinaCore

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── SCENE CONFIGURATIONS ─────────────────────────────────────
# Pick 2 KITTI sequences + 1 DAVIS sequence
# Using sequences confirmed to have cars in ROI

KITTI_ROOT = Path("data/kitti/data_tracking_image/image_02")
LBL_DIR    = Path("data/kitti/data_tracking_image/label_02")
DAVIS_ROOT = Path("data/davis/DAVIS/JPEGImages/480p")
DAVIS_ANNO = Path("data/davis/DAVIS/Annotations/480p")

scenes = [
    {
        'type':   'kitti',
        'seq':    '0004',
        'frame':  50,        # confirmed good — 41% ROI
        'warmup': 20,
        'label':  'KITTI Tracking — Urban',
    },
    {
        'type':   'kitti',
        'seq':    '0003',    # was '0001' — change this
        'frame':  60,        # try 40, 60, 80 until ROI < 70%
        'warmup': 20,
        'label':  'KITTI Tracking — Suburban',
    },
    {
        'type':   'davis',
        'seq':    'car-roundabout',
        'frame':  20,
        'warmup': 15,
        'label':  'DAVIS 2017 — Zero-Shot Transfer',
    },
]

# ── HELPERS ──────────────────────────────────────────────────
def load_kitti_gt(seq, frame_num):
    boxes = []
    try:
        gt_df = pd.read_csv(
            LBL_DIR / f"{seq}.txt",
            sep=' ', header=None
        )
        gt_df = gt_df[[0,2,6,7,8,9]]
        gt_df.columns = ['frame','type','x1','y1','x2','y2']
        frame_gt = gt_df[
            (gt_df['frame'] == frame_num) &
            (gt_df['type'].isin(
                ['Car','Pedestrian','Cyclist']
            ))
        ]
        for _, row in frame_gt.iterrows():
            boxes.append(
                [row.x1, row.y1, row.x2, row.y2]
            )
    except:
        pass
    return boxes


def load_davis_gt(seq, frame_num):
    """Get GT boxes from DAVIS mask."""
    img_paths = sorted(
        list((DAVIS_ROOT / seq).glob("*.jpg"))
    )
    if frame_num >= len(img_paths):
        return [], None
    img_path  = img_paths[frame_num]
    mask_path = (DAVIS_ANNO / seq /
                 img_path.name.replace('.jpg', '.png'))
    boxes = []
    if mask_path.exists():
        mask = cv2.imread(
            str(mask_path), cv2.IMREAD_GRAYSCALE
        )
        for uid in np.unique(mask):
            if uid == 0:
                continue
            ys, xs = np.where(mask == uid)
            if len(xs) > 0:
                boxes.append([
                    int(np.min(xs)), int(np.min(ys)),
                    int(np.max(xs)), int(np.max(ys))
                ])
    return boxes, str(img_path)


def process_scene(scene, model):
    """Returns (dense_img_rgb, sparse_img_rgb, meta_dict)"""

    # Load image
    if scene['type'] == 'kitti':
        img_paths = sorted(
            list((KITTI_ROOT / scene['seq']).glob("*.png"))
        )
        img_path  = img_paths[scene['frame']]
        img       = cv2.imread(str(img_path))
        gt_boxes  = load_kitti_gt(
            scene['seq'], scene['frame']
        )

    else:  # davis
        img_paths = sorted(
            list((DAVIS_ROOT / scene['seq']).glob("*.jpg"))
        )
        img_path  = img_paths[scene['frame']]
        img       = cv2.imread(str(img_path))
        gt_boxes, _ = load_davis_gt(
            scene['seq'], scene['frame']
        )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W    = img.shape[:2]

    # Warmup retina
    retina = RetinaCore.golden_baseline()
    for p in img_paths[
        max(0, scene['frame']-scene['warmup']):
        scene['frame']
    ]:
        wf = cv2.imread(str(p))
        if wf is not None:
            retina.process_frame(
                cv2.cvtColor(wf, cv2.COLOR_BGR2GRAY)
            )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rout = retina.process_frame(gray)
    roi  = retina.get_roi_bbox(
        rout, frame_shape=img.shape,
        max_area_frac=0.6
    )

    # YOLO on full frame
    yolo_results = model.predict(
        img, verbose=False, conf=0.35
    )[0]
    det_boxes = yolo_results.boxes.xyxy.cpu().numpy() \
                if yolo_results.boxes \
                else np.zeros((0,4))

    # Dense visualization
    dense = img_rgb.copy()
    for box in det_boxes:
        cv2.rectangle(
            dense,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (220, 50, 50), 2
        )

    # Sparse visualization
    if roi:
        x1,y1,x2,y2 = roi
        roi_area = (x2-x1)*(y2-y1)/(W*H)*100
        sparse                = (img_rgb * 0.22).astype(np.uint8)
        sparse[y1:y2, x1:x2] = img_rgb[y1:y2, x1:x2]
        cv2.rectangle(sparse, (x1,y1), (x2,y2),
                      (0, 230, 0), 3)
    else:
        sparse   = img_rgb.copy()
        roi_area = 100.0
        x1,y1,x2,y2 = 0,0,W,H

    # GT boxes in yellow on sparse panel
    for box in gt_boxes:
        cv2.rectangle(
            sparse,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (255, 215, 0), 2
        )

    meta = {
        'n_det':    len(det_boxes),
        'n_gt':     len(gt_boxes),
        'roi_area': roi_area,
        'sparsity': rout['sparsity'] * 100,
    }

    return dense, sparse, meta


# ── MAIN ─────────────────────────────────────────────────────
print("Loading YOLO...")
model = YOLO('yolo11m.pt')

print("Processing scenes...")
scene_data = []
for scene in scenes:
    print(f"  {scene['label']}...")
    dense, sparse, meta = process_scene(scene, model)
    scene_data.append((scene, dense, sparse, meta))

# ── FIGURE: 3 rows × 2 cols ──────────────────────────────────
fig, axes = plt.subplots(
    3, 2,
    figsize=(12, 7.5),    # slightly shorter
    facecolor='white',
    gridspec_kw={
        'wspace': 0.04,
        'hspace': 0.08    # was 0.18 — much tighter
    }
)

LEFT_TITLE  = 'Dense Inference (full frame)'
RIGHT_TITLE = 'RetiGate (sparse ROI)'

for row, (scene, dense, sparse, meta) in \
        enumerate(scene_data):

    ax_l = axes[row, 0]
    ax_r = axes[row, 1]

    # Left — dense
    ax_l.imshow(dense)
    ax_l.axis('off')

    # Right — sparse
    ax_r.imshow(sparse)
    ax_r.axis('off')

    # Row label on left edge
    ax_l.text(
        -0.02, 0.5,
        scene['label'],
        transform=ax_l.transAxes,
        fontsize=8.5, fontweight='bold',
        color='#2C3E50', va='center',
        ha='right', rotation=90
    )

    # Stats overlay — bottom strip on each image
    # Left panel: n detections
    ax_l.text(
        0.5, 0.02,
        f'{meta["n_det"]} detections',
        transform=ax_l.transAxes,
        ha='center', va='bottom',
        fontsize=8.5, color='white',
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.25',
            facecolor='#C0392B',
            alpha=0.82,
            edgecolor='none'
        )
    )

    # Right panel: roi area + sparsity
    ax_r.text(
        0.5, 0.02,
        f'ROI {meta["roi_area"]:.0f}%  ·  '
        f'{meta["sparsity"]:.1f}% sparsity',
        transform=ax_r.transAxes,
        ha='center', va='bottom',
        fontsize=8.5, color='white',
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.25',
            facecolor='#1E8449',
            alpha=0.82,
            edgecolor='none'
        )
    )

    # DAVIS badge on last row
    if scene['type'] == 'davis':
        ax_r.text(
            0.98, 0.97,
            'Zero-shot\ntransfer',
            transform=ax_r.transAxes,
            ha='right', va='top',
            fontsize=7.5, color='white',
            fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='#8E44AD',
                alpha=0.88,
                edgecolor='none'
            )
        )

# Column titles at very top
axes[0, 0].set_title(
    LEFT_TITLE,
    fontsize=11, fontweight='bold',
    color='#C0392B', pad=6
)
axes[0, 1].set_title(
    RIGHT_TITLE,
    fontsize=11, fontweight='bold',
    color='#1E8449', pad=6
)

# Legend below last row
legend_elements = [
    mpatches.Patch(
        facecolor='#C0392B', edgecolor='none',
        label='YOLO detection'
    ),
    mpatches.Patch(
        facecolor='#00E050', edgecolor='none',
        label='RetiGate ROI'
    ),
    mpatches.Patch(
        facecolor='#FFD700', edgecolor='none',
        label='Ground truth annotation'
    ),
    mpatches.Patch(
        facecolor='#8E44AD', edgecolor='none',
        label='DAVIS (no retuning)'
    ),
]
fig.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=4,
    fontsize=9,
    framealpha=0.9,
    bbox_to_anchor=(0.5, -0.03),
    edgecolor='lightgray'
)

fig.suptitle(
    'Qualitative Results: KITTI Tracking and DAVIS 2017',
    fontsize=13, fontweight='bold', y=1.01
)

# ── SAVE ─────────────────────────────────────────────────────
for fmt in ['pdf', 'png']:
    p = OUT / f"fig4_qualitative.{fmt}"
    plt.savefig(
        str(p), dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    print(f"Saved → {p}")

plt.close()
print("Figure 4 done.")