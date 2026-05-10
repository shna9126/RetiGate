# experiments/figures/fig2.py
# Fixed version — 5 equal panels, correct layout

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from pathlib import Path
from retigate import RetinaCore

# ── CONFIG ───────────────────────────────────────────────────
# Use the frame confirmed by diagnose_pipeline.py
# Adjust sequence and frame number to your best frame
SEQ        = "0006"   # highest recall sequence
FRAME_NUM  = 44       # mid-sequence, temporal state warm
WARMUP     = 50

DATA_ROOT  = Path("data/kitti/data_tracking_image/image_02")
OUT        = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── LOAD FRAME ───────────────────────────────────────────────
img_paths = sorted(list((DATA_ROOT / SEQ).glob("*.png")))

if FRAME_NUM >= len(img_paths):
    raise ValueError(
        f"Frame {FRAME_NUM} not in seq {SEQ} "
        f"(only {len(img_paths)} frames)"
    )

target_path = img_paths[FRAME_NUM]
img         = cv2.imread(str(target_path))
gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W        = gray.shape
print(f"Frame: {target_path.name}  {W}×{H}")

# ── WARMUP RETINA ────────────────────────────────────────────
retina       = RetinaCore.golden_baseline()
warmup_start = max(0, FRAME_NUM - WARMUP)
for p in img_paths[warmup_start:FRAME_NUM]:
    wf = cv2.imread(str(p))
    if wf is not None:
        retina.process_frame(
            cv2.cvtColor(wf, cv2.COLOR_BGR2GRAY)
        )
print(f"Warmed up on {FRAME_NUM - warmup_start} frames")

# ── PROCESS TARGET FRAME ─────────────────────────────────────
rout = retina.process_frame(gray)
# Diagnostic — understand what data we have
print(f"\nPipeline output keys: {list(rout.keys())}")
print(f"M_Motion range: "
      f"[{rout['M_Motion'].min():.6f}, "
      f"{rout['M_Motion'].max():.6f}]")
print(f"Active pixels: "
      f"{(rout['M_Motion'] > 0).sum()} "
      f"/ {H*W} "
      f"({(rout['M_Motion']>0).mean()*100:.2f}%)")
print(f"Sparsity: {rout['sparsity']*100:.2f}%")
print(f"Amacrine range: "
      f"[{retina.amacrine_state.min():.6f}, "
      f"{retina.amacrine_state.max():.6f}]")

# Compute DoG manually for the figure
img_f = gray.astype(np.float32) / 255.0
m_c   = cv2.filter2D(img_f, -1, retina.m_center_k,
                      borderType=cv2.BORDER_REFLECT)
m_s   = cv2.filter2D(img_f, -1, retina.m_surround_k,
                      borderType=cv2.BORDER_REFLECT)
dog   = np.abs(m_c - m_s)

# ── BUILD ROI VISUALIZATION ──────────────────────────────────
roi     = retina.get_roi_bbox(rout, frame_shape=img.shape)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if roi:
    x1, y1, x2, y2 = roi
    # Clamp to frame
    x1c = max(2, x1)
    y1c = max(2, y1)
    x2c = min(W-6, x2)
    y2c = min(H-6, y2)
    
    roi_area = (x2c-x1c)*(y2c-y1c) / (W*H) * 100
    print(f"ROI: {roi}  ({roi_area:.1f}% of frame)")
    print(f"Sparsity: {rout['sparsity']*100:.1f}%")

    # Dim outside ROI
    roi_vis                        = (img_rgb * 0.40).astype(np.uint8)
    roi_vis[y1c:y2c, x1c:x2c]     = img_rgb[y1c:y2c, x1c:x2c]
    # Thick green box with black outline
    cv2.rectangle(roi_vis, (x1c-2, y1c-2),
                  (x2c+2, y2c+2), (0,0,0), 8)
    cv2.rectangle(roi_vis, (x1c, y1c),
                  (x2c, y2c), (0,255,60), 5)
    roi_label = f"ROI  ({roi_area:.0f}% area)"
else:
    roi_vis   = img_rgb
    roi_label = "ROI  (full frame)"
    roi_area  = 100.0

# ── LOAD AND DRAW GT BOXES ──────────────────────────────────
# Load GT boxes for this frame
LBL_DIR = Path("data/kitti/data_tracking_image/label_02")
label_file = LBL_DIR / "0006.txt"

try:
    gt_df = pd.read_csv(label_file, sep=' ', header=None)
    gt_df = gt_df[[0, 2, 6, 7, 8, 9]]
    gt_df.columns = ['frame', 'type', 'x1', 'y1', 'x2', 'y2']
    gt_df = gt_df[
        (gt_df['frame'] == FRAME_NUM) &
        (gt_df['type'].isin(['Car', 'Pedestrian', 'Cyclist']))
    ]

    # Draw GT boxes on roi_vis in yellow
    for _, gt in gt_df.iterrows():
        cv2.rectangle(
            roi_vis,
            (int(gt.x1), int(gt.y1)),
            (int(gt.x2), int(gt.y2)),
            (255, 220, 0),   # yellow
            2
        )
except:
    pass  # skip if labels not available

# ── AFTER roi is computed: Add ROI debug ────────────────────
print(f"ROI: {roi}")
print(f"Sparsity: {rout['sparsity']*100:.1f}%")
if roi:
    area_pct = (roi[2]-roi[0])*(roi[3]-roi[1])/(W*H)*100
    print(f"ROI area: {area_pct:.1f}%")
else:
    print("WARNING: roi is None — trying max_area_frac=0.5")
    roi = retina.get_roi_bbox(
        rout,
        frame_shape=img.shape,
        max_area_frac=0.5   # more permissive
    )
    print(f"Retry ROI: {roi}")

# ── NORMALIZE HELPER ─────────────────────────────────────────
def norm(x):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

# ── AMACRINE: Log normalization ──────────────────────────────
amacrine_raw     = retina.amacrine_state.copy()
p98              = np.percentile(amacrine_raw, 98)
amacrine_clipped = np.clip(amacrine_raw, 0, p98)
# Sqrt compression — reduces bright peaks more than DoG
amacrine_sqrt    = np.sqrt(norm(amacrine_clipped))
amacrine_blurred = cv2.GaussianBlur(
    amacrine_sqrt.astype(np.float32), (11, 11), 3.0
)
amacrine_display = norm(amacrine_blurred)

# ── GANGLION: Clip then gamma ────────────────────────────────
# Also replace ganglion processing:
ganglion_raw = rout['M_Motion'].copy()
active_vals  = ganglion_raw[ganglion_raw > 0]

if len(active_vals) > 0:
    # Use lower threshold — show all active signal
    # just clip extreme outliers
    p99 = np.percentile(active_vals, 99)
    ganglion_clipped = np.clip(ganglion_raw, 0, p99)
    # Gamma boost so faint pixels become visible
    ganglion_norm = norm(ganglion_clipped)
    # Only show pixels above 10% of max — removes noise halo
    ganglion_norm[ganglion_norm < 0.10] = 0
    ganglion_vis  = np.power(ganglion_norm, 0.50)
else:
    ganglion_vis = norm(ganglion_raw)
    print("WARNING: No active pixels in ganglion output")
    print(f"  M_Motion max: {ganglion_raw.max():.6f}")
    print(f"  M_Motion mean: {ganglion_raw.mean():.6f}")


# ── STAGE DEFINITIONS (5 stages) ────────────────────────────
stages = [
    {
        'data':   gray,
        'cmap':   'gray',
        'title':  'Raw Frame',
        'bio':    'Photoreceptors',
        'math':   r'$I_t \in [0,1]^{H \times W}$',
        'param':  None,
        'color':  False,
    },
    {
        'data':   norm(dog),
        'cmap':   'hot',
        'title':  'DoG Filter',
        'bio':    'Bipolar Cells',
        'math':   r'$|G_c \ast I - G_s \ast I|$',
        'param':  r'$\sigma_c=1.5,\ \sigma_s=4.0$',
        'color':  False,
    },
    {
        'data':   amacrine_display,
        'cmap':   'inferno',
        'title':  'Amacrine State',
        'bio':    'Amacrine Cells',
        'math':   r'$\lambda M + (1{-}\lambda)\hat{M}$',
        'param':  r'$\lambda=0.10,\ \omega=1.5$',
        'color':  False,
    },
    {
        'data':   ganglion_vis,
        'cmap':   'hot',
        'title':  'Ganglion Output',
        'bio':    'Ganglion Cells',
        'math':   r'$\max(0,\ M_{bip} - \mathrm{Inhib})$',
        'param':  r'$\tau=0.10$',
        'color':  False,
    },
    {
        'data':   roi_vis,
        'cmap':   None,
        'title':  'Gated ROI',
        'bio':    'Foveal Gating',
        'math':   r'$\mathrm{bbox}(\mathcal{A}_{ganglion})$',
        'param':  f'$\\mathbf{{{roi_area:.0f}\\%}}$ area retained',
        'color':  True,
    },
]

N = len(stages)  # 5

# ── GRID LAYOUT ──────────────────────────────────────────────
# Columns: img arr img arr img arr img arr img
# Indices:  0   1   2   3   4   5   6   7   8
# Widths:   4  0.4  4  0.4  4  0.4  4  0.4  4

n_img_cols   = N                       # 5
n_arrow_cols = N - 1                   # 4
total_cols   = n_img_cols + n_arrow_cols  # 9

col_widths = []
for i in range(N):
    col_widths.append(4)
    if i < N - 1:
        col_widths.append(0.4)

fig = plt.figure(figsize=(22, 7.5), facecolor='white')
gs  = GridSpec(
    4, total_cols,
    figure=fig,
    height_ratios=[3.5, 0.55, 0.55, 0.55],
    width_ratios=col_widths,
    hspace=0.06,
    wspace=0.03
)

# Image panel column indices (0, 2, 4, 6, 8)
img_cols   = list(range(0, total_cols, 2))
# Arrow column indices (1, 3, 5, 7)
arrow_cols = list(range(1, total_cols, 2))

# ── COLORS ───────────────────────────────────────────────────
C_TITLE = '#2C3E50'
C_BIO   = '#1A5276'
C_MATH  = '#1E8449'
C_PARAM = '#6C3483'
C_ARROW = '#C0392B'


def make_label(ax, text, bg_color, fontsize=9):
    """Draw a colored rounded box with centered white text."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(FancyBboxPatch(
        (0.03, 0.10), 0.94, 0.80,
        boxstyle="round,pad=0.04",
        facecolor=bg_color,
        edgecolor='none',
        zorder=2
    ))
    ax.text(
        0.5, 0.5, text,
        ha='center', va='center',
        fontsize=fontsize,
        color='white',
        fontweight='bold',
        zorder=3
    )


# ── DRAW THUMBNAILS (ROW 0) ──────────────────────────────────
for i, (col, stage) in enumerate(zip(img_cols, stages)):
    ax = fig.add_subplot(gs[0, col])

    if stage['color']:
        # Color image (ROI panel) — must use imshow directly
        ax.imshow(stage['data'], aspect='auto',
                  interpolation='bilinear')
    else:
        ax.imshow(stage['data'], cmap=stage['cmap'],
                  aspect='auto', interpolation='bilinear')

    # Title badge
    ax.set_title(
        stage['title'],
        fontsize=10, fontweight='bold',
        color='white', pad=5,
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor=C_TITLE,
            alpha=0.92,
            edgecolor='none'
        )
    )
    ax.axis('off')

    # Ganglion inset annotation (stage index 3)
    if i == 3:
        # Find centroid of active pixels
        active_mask = ganglion_vis > 0.1
        if active_mask.any():
            ys, xs = np.where(active_mask)
            # These are already axis fractions (0-1)
            # because imshow maps pixel coords to axes coords
            cy = 1.0 - (ys.mean() / H)  # flip y for matplotlib
            cx = xs.mean() / W
            print(f"Active pixel centroid: cx={cx:.2f} cy={cy:.2f}")
        else:
            # Fallback — point to centre-bottom of panel
            cx, cy = 0.5, 0.5
            print("WARNING: no active pixels above 0.1 threshold")
            print("Lowering threshold for annotation...")
            # Try lower threshold
            active_mask2 = ganglion_vis > 0.01
            if active_mask2.any():
                ys2, xs2 = np.where(active_mask2)
                cy = 1.0 - (ys2.mean() / H)
                cx = xs2.mean() / W

            # Place text in bottom-right, arrow points UP to pixels
        text_x = 0.72
        text_y = 0.22

        # Ensure text doesn't overlap with arrow target
        if cy < 0.4:   # active pixels in upper half
            text_y = 0.75  # put text in lower half

        ax.annotate(
            f'active pixels\n({(active_mask.sum()/(H*W)*100):.2f}%)',
            xy=(cx, cy),
            xytext=(text_x, text_y),
            xycoords='axes fraction',
            textcoords='axes fraction',
            fontsize=8,
            color='white',
            fontweight='bold',
            ha='center',
            arrowprops=dict(
                arrowstyle='->',
                color='white',
                lw=1.5,
                connectionstyle='arc3,rad=0.2'
            ),
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='#C0392B',
                alpha=0.90,
                edgecolor='none'
            )
        )

    # Green border on ROI panel
    if stage['color']:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#27AE60')
            spine.set_linewidth(3)

# ── DRAW ARROWS (ROW 0, ODD COLUMNS) ────────────────────────
for col in arrow_cols:
    ax = fig.add_subplot(gs[0, col])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.annotate(
        '',
        xy=(0.90, 0.50),
        xytext=(0.10, 0.50),
        arrowprops=dict(
            arrowstyle='->',
            color=C_ARROW,
            lw=2.5,
            mutation_scale=22,
        )
    )

# ── DRAW LABEL ROWS (ROWS 1–3) ───────────────────────────────
for i, (col, stage) in enumerate(zip(img_cols, stages)):

    # Row 1 — Biological analogue
    ax_bio = fig.add_subplot(gs[1, col])
    make_label(ax_bio, stage['bio'], C_BIO, fontsize=9)

    # Empty arrow cells in label rows
    if i < N - 1:
        for row in [1, 2, 3]:
            fig.add_subplot(gs[row, arrow_cols[i]]).axis('off')

    # Row 2 — Mathematical operation
    ax_math = fig.add_subplot(gs[2, col])
    make_label(ax_math, stage['math'], C_MATH, fontsize=8.5)

    # Row 3 — Tunable parameter (or empty)
    ax_param = fig.add_subplot(gs[3, col])
    if stage['param']:
        make_label(ax_param, stage['param'], C_PARAM,
                   fontsize=8)
    else:
        ax_param.axis('off')

# ── LEGEND ───────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=C_BIO,
                   label='Biological analogue',
                   edgecolor='none'),
    mpatches.Patch(facecolor=C_MATH,
                   label='Mathematical operation',
                   edgecolor='none'),
    mpatches.Patch(facecolor=C_PARAM,
                   label='Tunable parameter',
                   edgecolor='none'),
]
fig.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=3,
    fontsize=10,
    framealpha=0.9,
    bbox_to_anchor=(0.5, -0.01),
    edgecolor='lightgray'
)

# ── TITLE ────────────────────────────────────────────────────
fig.suptitle(
    'RetiGate Pipeline: Bio-Inspired Motion Saliency Pre-Filter',
    fontsize=13,
    fontweight='bold',
    y=1.005
)

# ── SAVE ─────────────────────────────────────────────────────
for fmt in ['pdf', 'png']:
    save_path = OUT / f"fig2_pipeline.{fmt}"
    plt.savefig(
        str(save_path),
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    print(f"Saved → {save_path}")

plt.close()
print("\nDone. Check fig2_pipeline.png")