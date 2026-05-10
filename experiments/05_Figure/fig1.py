# experiments/figures/fig1.py — conference final

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from retigate import RetinaCore

# ── CONFIG ───────────────────────────────────────────────────
SEQ       = "0006"
FRAME_NUM = 44
WARMUP    = 20

DATA_ROOT = Path("data/kitti/data_tracking_image/image_02")
LBL_DIR   = Path("data/kitti/data_tracking_image/label_02")
OUT       = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

C_RED      = '#C0392B'
C_GREEN    = '#1A7A3A'
C_GOLD     = (255, 215, 0)
C_GRN_CV   = (0, 220, 60)
C_BLK      = (0, 0, 0)
C_DENSE_CV = (255, 100, 20)   # orange-red — distinct from scene

# ── LOAD + PROCESS ───────────────────────────────────────────
img_paths = sorted(list((DATA_ROOT / SEQ).glob("*.png")))
img       = cv2.imread(str(img_paths[FRAME_NUM]))
img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W      = img.shape[:2]

retina = RetinaCore.golden_baseline()
retina.threshold = 0.10
for p in img_paths[max(0, FRAME_NUM - WARMUP):FRAME_NUM]:
    wf = cv2.imread(str(p))
    if wf is not None:
        retina.process_frame(
            cv2.cvtColor(wf, cv2.COLOR_BGR2GRAY)
        )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rout = retina.process_frame(gray)
roi  = retina.get_roi_bbox(rout, frame_shape=img.shape)

print("Running YOLO...")
model       = YOLO('yolo11m.pt')
results     = model.predict(img, verbose=False, conf=0.35)[0]
dense_boxes = (results.boxes.xyxy.cpu().numpy()
               if results.boxes is not None
               else np.zeros((0, 4)))

gt_boxes = []
try:
    gt_df    = pd.read_csv(
        LBL_DIR / f"{SEQ}.txt", sep=' ', header=None
    )
    gt_df    = gt_df[[0, 2, 6, 7, 8, 9]]
    gt_df.columns = ['frame','type','x1','y1','x2','y2']
    frame_gt = gt_df[
        (gt_df['frame'] == FRAME_NUM) &
        (gt_df['type'].isin(['Car','Pedestrian','Cyclist']))
    ]
    for _, row in frame_gt.iterrows():
        gt_boxes.append([row.x1, row.y1, row.x2, row.y2])
except Exception as e:
    print(f"GT warning: {e}")

roi_area = 100.0
if roi:
    x1r, y1r, x2r, y2r = roi
    roi_area = (x2r - x1r) * (y2r - y1r) / (W * H) * 100

print(f"ROI: {roi_area:.0f}%  Detections: {len(dense_boxes)}")

# ── BUILD DENSE PANEL ─────────────────────────────────────────
dense_vis = img_rgb.copy()
for box in dense_boxes:
    cv2.rectangle(dense_vis,
                  (int(box[0])-1, int(box[1])-1),
                  (int(box[2])+1, int(box[3])+1),
                  C_BLK, 5)
    cv2.rectangle(dense_vis,
                  (int(box[0]), int(box[1])),
                  (int(box[2]), int(box[3])),
                  C_DENSE_CV, 3)

# ── BUILD SPARSE PANEL ───────────────────────────────────────
if roi:
    roi_vis = (img_rgb * 0.42).astype(np.uint8)
    roi_vis[y1r:y2r, x1r:x2r] = img_rgb[y1r:y2r, x1r:x2r]
    cv2.rectangle(roi_vis,
                  (x1r-3, y1r-3), (x2r+3, y2r+3),
                  C_BLK, 10)
    cv2.rectangle(roi_vis,
                  (x1r, y1r), (x2r, y2r),
                  C_GRN_CV, 6)
    for box in gt_boxes:
        cv2.rectangle(roi_vis,
                      (int(box[0])-1, int(box[1])-1),
                      (int(box[2])+1, int(box[3])+1),
                      C_BLK, 5)
        cv2.rectangle(roi_vis,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      C_GOLD, 3)
else:
    roi_vis = img_rgb.copy()

# ── FIGURE ────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5.8), facecolor='white', dpi=150)

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    width_ratios=[10, 10, 5],
    height_ratios=[10, 1],
    wspace=0.18,          # FIX 3: wider gap → no label bleed
    hspace=0.03,          # FIX 1: tight gap → caption close
    left=0.01,
    right=0.99,
    top=0.90,
    bottom=0.09           # FIX 1: lift bottom so caption visible
)

ax_a  = fig.add_subplot(gs[0, 0])
ax_b  = fig.add_subplot(gs[0, 1])
ax_ca = fig.add_subplot(gs[1, 0])
ax_cb = fig.add_subplot(gs[1, 1])

gs_bars = gridspec.GridSpecFromSubplotSpec(
    2, 1,
    subplot_spec=gs[0, 2],
    hspace=0.80
)
ax_lat  = fig.add_subplot(gs_bars[0])
ax_enrg = fig.add_subplot(gs_bars[1])

ax_dummy = fig.add_subplot(gs[1, 2])
ax_dummy.axis('off')

# ── PANEL A ──────────────────────────────────────────────────
ax_a.imshow(dense_vis)
ax_a.set_title(
    '(a) Dense Inference — Full Frame',
    fontsize=12.5, fontweight='bold',
    pad=5, color='#2C3E50'
)
ax_a.axis('off')
ax_a.text(
    0.012, 0.976, 'A',
    transform=ax_a.transAxes,
    fontsize=14, fontweight='bold',
    color='white', va='top',
    bbox=dict(boxstyle='round,pad=0.25',
              facecolor='#C0392B',
              edgecolor='none', alpha=0.93)
)

# ── PANEL A CAPTION ───────────────────────────────────────────
ax_ca.axis('off')
ax_ca.text(
    0.5, 0.5,
    f'{len(dense_boxes)} detections  ·  '
    f'31.8 ms  ·  2107 mJ',
    transform=ax_ca.transAxes,
    ha='center', va='center',
    fontsize=11, color='#C0392B',
    fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.35',
              facecolor='#FDEDEC',
              edgecolor='#C0392B',
              linewidth=1.3, alpha=0.92)
)

# ── PANEL B ──────────────────────────────────────────────────
ax_b.imshow(roi_vis)
ax_b.set_title(
    f'(b) RetiGate — Sparse ROI '
    f'({roi_area:.0f}% of frame)',
    fontsize=12.5, fontweight='bold',
    pad=5, color='#2C3E50'
)
ax_b.axis('off')
ax_b.text(
    0.012, 0.976, 'B',
    transform=ax_b.transAxes,
    fontsize=14, fontweight='bold',
    color='white', va='top',
    bbox=dict(boxstyle='round,pad=0.25',
              facecolor='#1A7A3A',
              edgecolor='none', alpha=0.93)
)

legend_elements = [
    mpatches.Patch(facecolor='#00DC3C',
                   edgecolor='black', linewidth=0.8,
                   label='RetiGate ROI'),
    mpatches.Patch(facecolor='#FFD700',
                   edgecolor='black', linewidth=0.8,
                   label='GT car'),
]
ax_b.legend(
    handles=legend_elements,
    loc='lower left',
    fontsize=9.5,
    framealpha=0.90,
    facecolor='white',
    edgecolor='#AAAAAA',
    borderpad=0.5,
    handlelength=1.3
)

# ── PANEL B CAPTION ───────────────────────────────────────────
ax_cb.axis('off')
ax_cb.text(
    0.5, 0.5,
    f'{roi_area:.0f}% area  ·  17.4 ms  ·  '
    f'−38.6% energy  ·  92.86% mAP',
    transform=ax_cb.transAxes,
    ha='center', va='center',
    fontsize=11, color='#1A7A3A',
    fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.35',
              facecolor='#EAFAF1',
              edgecolor='#1A7A3A',
              linewidth=1.3, alpha=0.92)
)

# ── LATENCY BAR ───────────────────────────────────────────────
methods = ['Dense', 'RetiGate']

# FIX 3: hide y-tick labels — values inside bars
ax_lat.barh(
    methods, [31.8, 17.4],
    color=[C_RED, C_GREEN],
    edgecolor='white', linewidth=0.5, height=0.52
)
ax_lat.set_xlim(0, 50)
ax_lat.set_xlabel('ms', fontsize=10.5, labelpad=2)
ax_lat.set_title('Latency', fontsize=12,
                 fontweight='bold', pad=5)
ax_lat.set_yticks([])       # FIX 3: no y-labels
ax_lat.spines[['top','right','left']].set_visible(False)
ax_lat.grid(True, alpha=0.18, axis='x',
            linestyle='--', linewidth=0.7)
ax_lat.set_axisbelow(True)
ax_lat.tick_params(axis='x', labelsize=9.5)

for y, (label, val) in enumerate(
    zip(methods, [31.8, 17.4])
):
    # Value inside bar
    ax_lat.text(
        val / 2, y,
        f'{val} ms',
        ha='center', va='center',
        fontsize=10, color='white',
        fontweight='bold'
    )
    # Method label LEFT of bar, inside axes
    ax_lat.text(
        1.0, y,
        label,
        ha='left', va='center',
        fontsize=9.5, color='#2C3E50',
        fontweight='bold'
    )

ax_lat.text(
    43.5, 0.5,
    '1.83×',
    ha='center', va='center',
    fontsize=11.5, color=C_GREEN,
    fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3',
              facecolor='#EAFAF1',
              edgecolor=C_GREEN, linewidth=1.5)
)

# ── ENERGY BAR ────────────────────────────────────────────────
ax_enrg.barh(
    methods, [2107, 1294],
    color=[C_RED, C_GREEN],
    edgecolor='white', linewidth=0.5, height=0.52
)
ax_enrg.set_xlim(0, 2900)
ax_enrg.set_xlabel('mJ / frame', fontsize=10.5, labelpad=2)
ax_enrg.set_title('Energy', fontsize=12,
                  fontweight='bold', pad=5)
ax_enrg.set_yticks([])      # FIX 3: no y-labels
ax_enrg.spines[['top','right','left']].set_visible(False)
ax_enrg.grid(True, alpha=0.18, axis='x',
             linestyle='--', linewidth=0.7)
ax_enrg.set_axisbelow(True)
ax_enrg.tick_params(axis='x', labelsize=9.5)

for y, (label, val) in enumerate(
    zip(methods, [2107, 1294])
):
    ax_enrg.text(
        val / 2, y,
        f'{val}',
        ha='center', va='center',
        fontsize=10, color='white',
        fontweight='bold'
    )
    ax_enrg.text(
        1.0, y,
        label,
        ha='left', va='center',
        fontsize=9.5, color='#2C3E50',
        fontweight='bold'
    )

ax_enrg.text(
    2580, 0.5,
    '−38.6%',
    ha='center', va='center',
    fontsize=11.5, color=C_GREEN,
    fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3',
              facecolor='#EAFAF1',
              edgecolor=C_GREEN, linewidth=1.5)
)

# ── TITLE ─────────────────────────────────────────────────────
fig.suptitle(
    'RetiGate: Bio-Inspired Motion Gating Reduces '
    'Compute While Preserving Detection Accuracy',
    fontsize=15,            # FIX 2: larger title
    fontweight='bold',
    y=0.998,
    color='#1C1C1C'
)

# ── SAVE ─────────────────────────────────────────────────────
for fmt in ['pdf', 'png']:
    p = OUT / f"fig1_teaser.{fmt}"
    plt.savefig(
        str(p), dpi=300,
        bbox_inches='tight',
        facecolor='white',
        pad_inches=0.12     # FIX 2: prevents title clipping
    )
    print(f"Saved → {p}")

plt.close()
print("Figure 1 done.")