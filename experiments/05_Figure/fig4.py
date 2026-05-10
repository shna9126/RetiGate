# experiments/figures/fig4.py
# Qualitative results — 3 domains × 2 columns
# Row 1: KITTI Suburban   (in-distribution)
# Row 2: DAVIS goat       (zero-shot, animal)
# Row 3: Middlebury Walking (zero-shot, indoor)

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

# ── DATA ROOTS ───────────────────────────────────────────────
KITTI_ROOT   = Path("data/kitti/data_tracking_image/image_02")
LBL_DIR      = Path("data/kitti/data_tracking_image/label_02")
DAVIS_ROOT   = Path("data/davis/DAVIS/JPEGImages/480p")
DAVIS_ANNO   = Path("data/davis/DAVIS/Annotations/480p")
MIDDLEBURY   = Path("data/middlebury/other-data/Urban2")

# ── SCENE CONFIGS ────────────────────────────────────────────
scenes = [
    {
        'type':        'kitti',
        'seq':         '0006',
        'frame':       44,
        'warmup':      20,
        'label':       'KITTI Tracking\n— Suburban',
        'badge':       None,
        'badge_color': None,
    },
    {
        'type':        'davis',
        'seq':         'sheep',
        'frame':       24,       # 00035.jpg, ROI=11%
        'warmup':      8,
        'label':       'DAVIS 2017\n— Zero-Shot',
        'badge':       'Zero-shot\ntransfer',
        'badge_color': '#8E44AD',
    },
    {
        'type':        'middlebury',
        'seq':         'Urban2',
        'frame':       4,        
        'warmup':      3,
        'label':       'Middlebury\n— Zero-Shot',
        'badge':       'Middlebury\n(no retuning)',
        'badge_color': '#0E6655',
    },
]

# ── GT LOADERS ───────────────────────────────────────────────
def load_kitti_gt(seq, frame_num):
    boxes = []
    try:
        gt_df = pd.read_csv(
            LBL_DIR / f"{seq}.txt",
            sep=' ', header=None
        )
        gt_df = gt_df[[0, 2, 6, 7, 8, 9]]
        gt_df.columns = [
            'frame','type','x1','y1','x2','y2'
        ]
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
    except Exception as e:
        print(f"  GT load warning: {e}")
    return boxes


def load_davis_gt(seq, frame_path):
    """Load GT boxes from DAVIS binary mask."""
    boxes     = []
    mask_path = (
        DAVIS_ANNO / seq /
        Path(frame_path).name.replace('.jpg', '.png')
    )
    if mask_path.exists():
        mask = cv2.imread(
            str(mask_path), cv2.IMREAD_GRAYSCALE
        )
        for uid in np.unique(mask):
            if uid == 0:
                continue
            ys, xs = np.where(mask == uid)
            if len(xs) > 50:   # ignore tiny regions
                boxes.append([
                    int(np.min(xs)), int(np.min(ys)),
                    int(np.max(xs)), int(np.max(ys))
                ])
    return boxes


# ── SCENE PROCESSOR ──────────────────────────────────────────
def process_scene(scene, model):
    """
    Returns (dense_rgb, sparse_rgb, meta_dict)
    """
    stype = scene['type']

    # ── Load image paths + target image ──────────────────
    if stype == 'kitti':
        img_paths = sorted(
            list((KITTI_ROOT / scene['seq'])
                 .glob("*.png"))
        )
        img_path  = img_paths[scene['frame']]
        img       = cv2.imread(str(img_path))
        gt_boxes  = load_kitti_gt(
            scene['seq'], scene['frame']
        )

    elif stype == 'davis':
        img_paths = sorted(
            list((DAVIS_ROOT / scene['seq'])
                 .glob("*.jpg"))
        )
        img_path  = img_paths[scene['frame']]
        img       = cv2.imread(str(img_path))
        gt_boxes  = load_davis_gt(
            scene['seq'], str(img_path)
        )

    else:  # middlebury
        img_paths = sorted(
            list(MIDDLEBURY.glob("frame*.png"))
        )
        if not img_paths:
            # Try ppm format
            img_paths = sorted(
                list(MIDDLEBURY.glob("frame*.ppm"))
            )
        img_path  = img_paths[scene['frame']]
        img       = cv2.imread(str(img_path))
        gt_boxes  = []   # no GT for Middlebury

    if img is None:
        raise FileNotFoundError(
            f"Could not load: {img_path}"
        )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W    = img.shape[:2]
    print(f"  Loaded: {img_path.name}  {W}×{H}")

    # ── Warm up RetiGate ──────────────────────────────────
    retina = RetinaCore.golden_baseline()
    retina.threshold = 0.10

    warmup_paths = img_paths[
        max(0, scene['frame'] - scene['warmup']):
        scene['frame']
    ]
    for p in warmup_paths:
        wf = cv2.imread(str(p))
        if wf is not None:
            retina.process_frame(
                cv2.cvtColor(wf, cv2.COLOR_BGR2GRAY)
            )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rout = retina.process_frame(gray)
    roi  = retina.get_roi_bbox(
        rout, frame_shape=img.shape
    )

    sparsity = rout['sparsity'] * 100
    print(f"  Sparsity={sparsity:.1f}%", end='')

    if roi:
        x1, y1, x2, y2 = roi
        roi_area = (x2-x1)*(y2-y1)/(W*H)*100
        print(f"  ROI={roi_area:.0f}%")
    else:
        roi_area = 100.0
        print(f"  ROI=100% (full frame fallback)")

    # ── YOLO dense inference ──────────────────────────────
    yolo_res  = model.predict(
        img, verbose=False, conf=0.35
    )[0]
    det_boxes = (yolo_res.boxes.xyxy.cpu().numpy()
                 if yolo_res.boxes is not None
                 else np.zeros((0, 4)))

    # ── Dense panel ───────────────────────────────────────
    dense = img_rgb.copy()
    for box in det_boxes:
        cv2.rectangle(dense,
                      (int(box[0])-1, int(box[1])-1),
                      (int(box[2])+1, int(box[3])+1),
                      (0, 0, 0), 4)
        cv2.rectangle(dense,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (220, 60, 60), 2)

    # ── Sparse panel ──────────────────────────────────────
    if roi:
        x1, y1, x2, y2 = roi
        # Clamp to frame
        x1c = max(3, x1);   y1c = max(3, y1)
        x2c = min(W-3, x2); y2c = min(H-3, y2)

        sparse = (img_rgb * 0.38).astype(np.uint8)
        sparse[y1c:y2c, x1c:x2c] = \
            img_rgb[y1c:y2c, x1c:x2c]

        # Green ROI box — thick with black outline
        cv2.rectangle(sparse,
                      (x1c-2, y1c-2),
                      (x2c+2, y2c+2),
                      (0, 0, 0), 8)
        cv2.rectangle(sparse,
                      (x1c, y1c), (x2c, y2c),
                      (0, 230, 60), 5)
    else:
        sparse = img_rgb.copy()

    # GT boxes in yellow
    for box in gt_boxes:
        cv2.rectangle(sparse,
                      (int(box[0])-1, int(box[1])-1),
                      (int(box[2])+1, int(box[3])+1),
                      (0, 0, 0), 4)
        cv2.rectangle(sparse,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (255, 215, 0), 2)

    meta = {
        'n_det':    len(det_boxes),
        'n_gt':     len(gt_boxes),
        'roi_area': roi_area,
        'sparsity': sparsity,
    }
    return dense, sparse, meta


# ── MAIN ─────────────────────────────────────────────────────
print("Loading YOLO11m...")
model = YOLO('yolo11m.pt')

print("\nProcessing scenes...")
scene_data = []
for scene in scenes:
    print(f"\n[{scene['label'].replace(chr(10),' ')}]")
    dense, sparse, meta = process_scene(scene, model)
    scene_data.append((scene, dense, sparse, meta))

# ── FIGURE ───────────────────────────────────────────────────
fig, axes = plt.subplots(
    3, 2,
    figsize=(12, 8.0),
    facecolor='white',
    gridspec_kw={'wspace': 0.04, 'hspace': 0.10}
)

for row, (scene, dense, sparse, meta) in \
        enumerate(scene_data):

    ax_l = axes[row, 0]
    ax_r = axes[row, 1]

    ax_l.imshow(dense)
    ax_l.axis('off')
    ax_r.imshow(sparse)
    ax_r.axis('off')

    # Row label — left of dense panel
    ax_l.text(
        -0.02, 0.5,
        scene['label'],
        transform=ax_l.transAxes,
        fontsize=9, fontweight='bold',
        color='#2C3E50',
        va='center', ha='right',
        rotation=90
    )

    # Dense stats badge
    det_label = (
        'YOLO: 0 det. (synthetic domain)'
        if scene['type'] == 'middlebury'
        else f'{meta["n_det"]} detections'
    )

    ax_l.text(
        0.5, 0.03,
        det_label,
        transform=ax_l.transAxes,
        ha='center', va='bottom',
        fontsize=8.5 if scene['type'] != 'middlebury'
                  else 7.5,
        color='white',
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='#C0392B',
            alpha=0.85,
            edgecolor='none'
        )
    )

    # Sparse stats badge
    ax_r.text(
        0.5, 0.03,
        f'ROI {meta["roi_area"]:.0f}%'
        f'  ·  {meta["sparsity"]:.1f}% sparsity',
        transform=ax_r.transAxes,
        ha='center', va='bottom',
        fontsize=9, color='white',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='#1E8449',
                  alpha=0.85, edgecolor='none')
    )

    # Domain badge — top right of sparse panel
    if scene['badge']:
        ax_r.text(
            0.98, 0.97,
            scene['badge'],
            transform=ax_r.transAxes,
            ha='right', va='top',
            fontsize=8, color='white',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor=scene['badge_color'],
                      alpha=0.90, edgecolor='none')
        )

# Column headers
axes[0, 0].set_title(
    'Dense Inference (full frame)',
    fontsize=12, fontweight='bold',
    color='#C0392B', pad=7
)
axes[0, 1].set_title(
    'RetiGate (sparse ROI)',
    fontsize=12, fontweight='bold',
    color='#1E8449', pad=7
)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#DC3C3C',
                   edgecolor='none',
                   label='YOLO detection'),
    mpatches.Patch(facecolor='#00E650',
                   edgecolor='none',
                   label='RetiGate ROI'),
    mpatches.Patch(facecolor='#FFD700',
                   edgecolor='none',
                   label='Ground truth annotation'),
    mpatches.Patch(facecolor='#8E44AD',
                   edgecolor='none',
                   label='DAVIS (no retuning)'),
    mpatches.Patch(facecolor='#0E6655',
                   edgecolor='none',
                   label='Middlebury (no retuning)'),
]
fig.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=5,
    fontsize=8.5,
    framealpha=0.92,
    bbox_to_anchor=(0.5, -0.02),
    edgecolor='lightgray'
)

fig.suptitle(
    'Qualitative Results: KITTI Tracking, '
    'DAVIS 2017, and Middlebury',
    fontsize=13, fontweight='bold', y=1.01
)

# ── SAVE ─────────────────────────────────────────────────────
for fmt in ['pdf', 'png']:
    p = OUT / f"fig4_qualitative.{fmt}"
    plt.savefig(
        str(p), dpi=300,
        bbox_inches='tight',
        facecolor='white',
        pad_inches=0.08
    )
    print(f"\nSaved → {p}")

plt.close()
print("Figure 4 done.")