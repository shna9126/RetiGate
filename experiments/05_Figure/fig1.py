# experiments/figures/fig1.py — tight layout redesign

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

# ── LOAD + PROCESS ───────────────────────────────────────────
img_paths   = sorted(list((DATA_ROOT / SEQ).glob("*.png")))
img         = cv2.imread(str(img_paths[FRAME_NUM]))
img_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W        = img.shape[:2]

retina = RetinaCore.golden_baseline()
for p in img_paths[max(0, FRAME_NUM-WARMUP):FRAME_NUM]:
    wf = cv2.imread(str(p))
    if wf is not None:
        retina.process_frame(
            cv2.cvtColor(wf, cv2.COLOR_BGR2GRAY)
        )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rout = retina.process_frame(gray)
roi  = retina.get_roi_bbox(
    rout, frame_shape=img.shape, max_area_frac=0.6
)

print("Running YOLO...")
model       = YOLO('yolo11m.pt')
results     = model.predict(img, verbose=False, conf=0.35)[0]
dense_boxes = results.boxes.xyxy.cpu().numpy() \
              if results.boxes else np.zeros((0,4))

gt_boxes = []
try:
    gt_df = pd.read_csv(
        LBL_DIR / f"{SEQ}.txt", sep=' ', header=None
    )
    gt_df = gt_df[[0,2,6,7,8,9]]
    gt_df.columns = ['frame','type','x1','y1','x2','y2']
    frame_gt = gt_df[
        (gt_df['frame'] == FRAME_NUM) &
        (gt_df['type'].isin(['Car','Pedestrian','Cyclist']))
    ]
    for _, row in frame_gt.iterrows():
        gt_boxes.append([row.x1, row.y1, row.x2, row.y2])
except:
    pass

# ── BUILD VISUALIZATIONS ─────────────────────────────────────
# Dense panel
dense_vis = img_rgb.copy()
for box in dense_boxes:
    cv2.rectangle(dense_vis,
                  (int(box[0]),int(box[1])),
                  (int(box[2]),int(box[3])),
                  (220,50,50), 2)

# Sparse panel
if roi:
    x1,y1,x2,y2 = roi
    roi_area = (x2-x1)*(y2-y1)/(W*H)*100
    roi_vis                = (img_rgb * 0.22).astype(np.uint8)
    roi_vis[y1:y2, x1:x2] = img_rgb[y1:y2, x1:x2]
    cv2.rectangle(roi_vis, (x1,y1), (x2,y2), (0,230,0), 3)
    for box in gt_boxes:
        cv2.rectangle(roi_vis,
                      (int(box[0]),int(box[1])),
                      (int(box[2]),int(box[3])),
                      (255,215,0), 2)
else:
    roi_vis  = img_rgb.copy()
    roi_area = 100.0

# Add caption bars directly onto images
def add_caption_bar(img_array, text, color_bgr):
    """
    Draws a semi-transparent bar at the BOTTOM of the image
    with white text. Avoids overlapping main content.
    Works on RGB arrays.
    """
    out  = img_array.copy()
    bar_h = max(28, int(img_array.shape[0] * 0.10))
    overlay = out[-bar_h:, :].copy()
    overlay[:] = color_bgr
    out[-bar_h:, :] = cv2.addWeighted(
        out[-bar_h:, :], 0.25,
        overlay, 0.75, 0
    )
    # Draw text
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = bar_h / 40
    thickness  = 1
    text_size  = cv2.getTextSize(
        text, font, font_scale, thickness
    )[0]
    text_x = (img_array.shape[1] - text_size[0]) // 2
    text_y = img_array.shape[0] - bar_h//2 + text_size[1]//2
    cv2.putText(out, text, (text_x, text_y),
                font, font_scale,
                (255,255,255), thickness,
                cv2.LINE_AA)
    return out

dense_vis = add_caption_bar(
    dense_vis,
    f'{len(dense_boxes)} detections  |  31.8 ms  |  2107 mJ',
    (200, 50, 50)
)
roi_vis = add_caption_bar(
    roi_vis,
    f'{roi_area:.0f}% area  |  17.4 ms  |  1294 mJ  |  92.86% mAP kept',
    (20, 160, 60)
)

# ── FIGURE: 1 row, 3 cols ────────────────────────────────────
# Panels A and B are equal width images
# Panel C is narrower bar charts
fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 4.2),
    facecolor='white',
    gridspec_kw={
        'width_ratios': [5, 5, 3],
        'wspace': 0.06
    }
)

# ── PANEL A ──────────────────────────────────────────────────
axes[0].imshow(dense_vis)
axes[0].set_title(
    '(a) Dense Inference — Full Frame',
    fontsize=11, fontweight='bold',
    pad=5, color='#2C3E50'
)
axes[0].axis('off')
axes[0].text(
    0.015, 0.975, 'A',
    transform=axes[0].transAxes,
    fontsize=12, fontweight='bold',
    color='white', va='top',
    bbox=dict(boxstyle='round,pad=0.2',
              facecolor='#C0392B', alpha=0.9)
)

# ── PANEL B ──────────────────────────────────────────────────
axes[1].imshow(roi_vis)
axes[1].set_title(
    f'(b) RetiGate — Sparse ROI ({roi_area:.0f}% of frame)',
    fontsize=11, fontweight='bold',
    pad=5, color='#2C3E50'
)
axes[1].axis('off')
axes[1].text(
    0.015, 0.975, 'B',
    transform=axes[1].transAxes,
    fontsize=12, fontweight='bold',
    color='white', va='top',
    bbox=dict(boxstyle='round,pad=0.2',
              facecolor='#1E8449', alpha=0.9)
)

# Legend inside image — top right
legend_elements = [
    mpatches.Patch(facecolor='#00E050',
                   edgecolor='none',
                   label='RetiGate ROI'),
    mpatches.Patch(facecolor='#FFD700',
                   edgecolor='none',
                   label='GT car'),
]
axes[1].legend(
    handles=legend_elements,
    loc='upper right',
    fontsize=8,
    framealpha=0.82,
    borderpad=0.3,
    handlelength=1.0
)

# ── PANEL C: Bar charts ──────────────────────────────────────
ax_c = axes[2]
ax_c.axis('off')

# Use GridSpecFromSubplotSpec for two stacked bars
from matplotlib.gridspec import GridSpecFromSubplotSpec
gs_inner = GridSpecFromSubplotSpec(
    2, 1,
    subplot_spec=fig.add_gridspec(
        1, 3,
        width_ratios=[5,5,3],
        wspace=0.06
    )[2],
    hspace=0.55
)

ax_lat  = fig.add_subplot(gs_inner[0])
ax_enrg = fig.add_subplot(gs_inner[1])

methods = ['Dense', 'RetiGate']
C_RED   = '#C0392B'
C_GRN   = '#1E8449'

# --- Latency ---
lats    = [31.8, 17.4]
lcolors = [C_RED, C_GRN]

b1 = ax_lat.barh(
    methods, lats,
    color=lcolors, edgecolor='none', height=0.42
)
ax_lat.set_xlim(0, 42)
ax_lat.set_xlabel('ms', fontsize=9)
ax_lat.set_title('Latency', fontsize=10,
                 fontweight='bold', pad=4)
ax_lat.tick_params(labelsize=8.5)
ax_lat.spines[['top','right','left']].set_visible(False)
ax_lat.grid(True, alpha=0.2, axis='x')

for bar, val in zip(b1, lats):
    # value inside bar
    ax_lat.text(
        val/2,
        bar.get_y() + bar.get_height()/2,
        f'{val} ms',
        ha='center', va='center',
        fontsize=9, color='white',
        fontweight='bold'
    )

# Speedup badge
ax_lat.text(
    36, 0.5,
    '1.83×',
    ha='center', va='center',
    fontsize=10, color=C_GRN,
    fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.25',
              facecolor='#EAFAF1',
              edgecolor=C_GRN,
              linewidth=1.2)
)

# --- Energy ---
energies = [2107, 1294]
ecolors  = [C_RED, C_GRN]

b2 = ax_enrg.barh(
    methods, energies,
    color=ecolors, edgecolor='none', height=0.42
)
ax_enrg.set_xlim(0, 2700)
ax_enrg.set_xlabel('mJ / frame', fontsize=9)
ax_enrg.set_title('Energy', fontsize=10,
                  fontweight='bold', pad=4)
ax_enrg.tick_params(labelsize=8.5)
ax_enrg.spines[['top','right','left']].set_visible(False)
ax_enrg.grid(True, alpha=0.2, axis='x')

for bar, val in zip(b2, energies):
    ax_enrg.text(
        val/2,
        bar.get_y() + bar.get_height()/2,
        f'{val}',
        ha='center', va='center',
        fontsize=9, color='white',
        fontweight='bold'
    )

# Saving badge
ax_enrg.text(
    2350, 0.5,
    '−38.6%',
    ha='center', va='center',
    fontsize=10, color=C_GRN,
    fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.25',
              facecolor='#EAFAF1',
              edgecolor=C_GRN,
              linewidth=1.2)
)

# ── TITLE ────────────────────────────────────────────────────
fig.suptitle(
    'RetiGate: Bio-Inspired Motion Gating Reduces '
    'Compute While Preserving Detection Accuracy',
    fontsize=12, fontweight='bold', y=1.03
)

# ── SAVE ─────────────────────────────────────────────────────
for fmt in ['pdf', 'png']:
    p = OUT / f"fig1_teaser.{fmt}"
    plt.savefig(str(p), dpi=300,
                bbox_inches='tight',
                facecolor='white')
    print(f"Saved → {p}")

plt.close()
print("Figure 1 done.")