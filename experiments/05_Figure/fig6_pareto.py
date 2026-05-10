# experiments/figures/fig6_pareto.py — final version

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── DATA ─────────────────────────────────────────────────────
# RetiGate — two honest operating points
retigate_points = [
    {
        'label':   'RetiGate DAVIS\n(100% active, ours)',
        'speedup': 1.83,    # conditional = effective on DAVIS
        'recall':  93.9,
        'marker':  '*',
        'color':   '#E74C3C',
        'size':    320,
    },
    {
        'label':   'RetiGate KITTI\n(eff. all frames, ours)',
        'speedup': 1.37,    # effective across all frames
        'recall':  95.44,
        'marker':  '*',
        'color':   '#C0392B',
        'size':    220,
    },
]

# Learned baselines — published numbers
# NOT directly comparable (different datasets/detectors)
# Shown only for context
learned_points = [
    {
        'label':   'AdaFocus V2\n[Wang 2022]',
        'speedup': 2.0,
        'recall':  99.0,
        'marker':  's',
        'color':   '#2980B9',
        'size':    130,
    },
    {
        'label':   'Glance & Focus\n[Wang 2020]',
        'speedup': 1.8,
        'recall':  98.5,
        'marker':  '^',
        'color':   '#1A5276',
        'size':    130,
    },
]

# Classical baselines
classical_points = [
    {
        'label':   'MOG2 + YOLO',
        'speedup': 1.3,
        'recall':  85.0,
        'marker':  'D',
        'color':   '#95A5A6',
        'size':    90,
    },
    {
        'label':   'Dense YOLO11m\n(baseline)',
        'speedup': 1.0,
        'recall':  100.0,
        'marker':  'o',
        'color':   '#2C3E50',
        'size':    110,
    },
]

# ── FIGURE ───────────────────────────────────────────────────
fig, ax = plt.subplots(
    figsize=(9, 6), facecolor='white'
)

# High-efficiency zone shading
ax.axvspan(1.5, 2.3, alpha=0.06,
           color='#27AE60', zorder=0)
ax.text(
    1.88, 103.0,       # move to TOP of zone
    'High-efficiency\nzone',
    ha='center', va='top',
    fontsize=8.5,
    color='#1E8449', style='italic',
    fontweight='bold'
)

# Training-free boundary line
ax.axvline(x=1.5, color='#BDC3C7',
           linewidth=1.0, linestyle='--',
           alpha=0.5, zorder=1)

# ── PLOT POINTS ──────────────────────────────────────────────
all_groups = [
    (retigate_points,  True),
    (learned_points,   False),
    (classical_points, None),
]

for group, is_free in all_groups:
    for m in group:
        ec = ('#E74C3C' if is_free
              else '#2980B9' if is_free is False
              else '#95A5A6')
        ax.scatter(
            m['speedup'], m['recall'],
            marker=m['marker'],
            color=m['color'],
            s=m['size'],
            zorder=5,
            edgecolors='white',
            linewidth=1.2,
        )

# ── ANNOTATIONS ──────────────────────────────────────────────
# RetiGate DAVIS point
ax.annotate(
    'RetiGate DAVIS\n(ours, training-free)',
    xy=(1.83, 93.9),
    xytext=(2.0, 91.0),
    fontsize=8.5, color='#E74C3C',
    fontweight='bold', ha='center',
    bbox=dict(boxstyle='round,pad=0.25',
              facecolor='#FDEDEC',
              edgecolor='#E74C3C',
              alpha=0.92, linewidth=1.0),
    arrowprops=dict(arrowstyle='->',
                    color='#E74C3C', lw=1.2)
)

# RetiGate KITTI effective point
ax.annotate(
    'RetiGate KITTI\n(eff. all frames)',
    xy=(1.37, 95.44),
    xytext=(0.72, 93.5),
    fontsize=8.5, color='#C0392B',
    fontweight='bold', ha='center',
    bbox=dict(boxstyle='round,pad=0.25',
              facecolor='#FDEDEC',
              edgecolor='#C0392B',
              alpha=0.92, linewidth=1.0),
    arrowprops=dict(arrowstyle='->',
                    color='#C0392B', lw=1.2)
)

# Label learned methods directly
ax.annotate(
    'AdaFocus V2\n[Wang 2022]',
    xy=(2.0, 99.0),
    xytext=(2.1, 96.8),
    fontsize=8, color='#2980B9',
    ha='center',
    arrowprops=dict(arrowstyle='-',
                    color='#BDC3C7', lw=0.8)
)

ax.annotate(
    'Glance & Focus\n[Wang 2020]',
    xy=(1.8, 98.5),
    xytext=(1.55, 101.0),
    fontsize=8, color='#1A5276',
    ha='center',
    arrowprops=dict(arrowstyle='-',
                    color='#BDC3C7', lw=0.8)
)

ax.annotate(
    'MOG2 + YOLO',
    xy=(1.3, 85.0),
    xytext=(1.05, 84.0),
    fontsize=8, color='#7F8C8D',
    ha='center',
    arrowprops=dict(arrowstyle='-',
                    color='#BDC3C7', lw=0.8)
)

ax.annotate(
    'Dense YOLO11m\n(no gating)',
    xy=(1.0, 100.0),
    xytext=(0.72, 101.5),
    fontsize=8, color='#2C3E50',
    ha='center',
    arrowprops=dict(arrowstyle='-',
                    color='#BDC3C7', lw=0.8)
)

# ── DISCLAIMER NOTE ──────────────────────────────────────────
ax.text(
    0.98, 0.02,
    '† Learned methods use different datasets\n'
    '  and detectors; not directly comparable.',
    transform=ax.transAxes,
    fontsize=7.5, color='#7F8C8D',
    ha='right', va='bottom', style='italic'
)

# ── LEGEND — shape guide only ────────────────────────────────
legend_handles = [
    plt.scatter([], [], marker='*', color='#E74C3C',
                s=200, label='RetiGate (ours, training-free)'),
    plt.scatter([], [], marker='s', color='#2980B9',
                s=90,  label='Learned adaptive method †'),
    plt.scatter([], [], marker='D', color='#95A5A6',
                s=70,  label='Classical baseline'),
    plt.scatter([], [], marker='o', color='#2C3E50',
                s=80,  label='Dense baseline (1×)'),
]
ax.legend(
    handles=legend_handles,
    fontsize=8.5,
    loc='lower right',
    framealpha=0.92,
    edgecolor='lightgray',
    borderpad=0.6
)

# ── AXES ─────────────────────────────────────────────────────
ax.set_xlim(0.55, 2.25)
ax.set_ylim(82, 103.5)
ax.set_xlabel(
    'Inference Speedup (×)\n'
    '[RetiGate: conditional on active frames]',
    fontsize=11, labelpad=5
)
ax.set_ylabel(
    'Object Recall / mAP Retention (%)',
    fontsize=11, labelpad=5
)
ax.set_title(
    'Accuracy–Efficiency Trade-off\n'
    'RetiGate vs Learned Adaptive Methods',
    fontsize=12, fontweight='bold', pad=10
)
ax.grid(True, alpha=0.18, linestyle='--',
        linewidth=0.7)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(labelsize=9.5)

# ── SAVE ─────────────────────────────────────────────────────
plt.tight_layout(pad=1.5)
for fmt in ['pdf', 'png']:
    p = OUT / f"fig6_pareto.{fmt}"
    plt.savefig(str(p), dpi=300,
                bbox_inches='tight',
                facecolor='white')
    print(f"Saved → {p}")
plt.close()
print("Fig 6 done.")