# experiments/figures/fig5.py — final production version

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── DATA ─────────────────────────────────────────────────────
davis_sparsity = [92.9, 98.6, 99.8]
davis_recall   = [70.2, 93.3, 93.9]

kitti_sparsity = 98.25
kitti_recall   = 95.68

variants = ['Baseline', 'w/o\nTemporal', 'w/o\nGlobal',
            'w/o\nSAC',   'w/o\nDoG',    'w/o\nVOR']
sparsity = [98.25, 82.80, 94.66, 98.25, 99.20, 98.20]

# ── FIGURE ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(13, 5.0),
    facecolor='white',
    gridspec_kw={'wspace': 0.32}
)

# ════════════════════════════════════════════════════════════
# LEFT: PARETO CURVE
# ════════════════════════════════════════════════════════════

# Success zone shading
ax1.axvspan(
    98.6, 99.8,
    alpha=0.18, color='#27AE60',
    zorder=1, label='Success Zone'
)
# Success zone label inside shading
ax1.text(
    99.2, 101.5,
    'Success\nZone',
    ha='center', va='top',
    fontsize=7.5, color='#1E8449',
    fontweight='bold', alpha=0.8
)

# Vertical dotted line at KITTI sparsity
ax1.axvline(
    x=kitti_sparsity,
    color='#1A5276', linewidth=1.0,
    linestyle=':', alpha=0.45, zorder=2
)

# DAVIS Pareto curve — 3 points only
ax1.plot(
    davis_sparsity, davis_recall,
    'o-', color='#2980B9',
    linewidth=2.5, markersize=9,
    markerfacecolor='white',
    markeredgewidth=2.2,
    zorder=5,
    label='DAVIS 2017 (90 seq)'
)

# KITTI triangle
ax1.plot(
    kitti_sparsity, kitti_recall,
    '^', color='#1A5276',
    markersize=13, zorder=6,
    label='KITTI Tracking (21 seq)'
)

# ── FIX 3: KITTI label — dark blue, clear of success zone
ax1.annotate(
    'KITTI\n95.7%',
    xy=(kitti_sparsity, kitti_recall),
    xytext=(97.2, 98.8),
    fontsize=8.5,
    color='#1A5276',
    fontweight='bold',
    ha='center',
    bbox=dict(boxstyle='round,pad=0.25',
              facecolor='white',
              edgecolor='#1A5276',
              linewidth=1.0,
              alpha=0.92),
    arrowprops=dict(arrowstyle='-',
                    color='#1A5276',
                    lw=0.8)
)

# ── FIX 2: τ labels on ALL three points ──────────────────────
# τ=0.01 — below-right
ax1.annotate(
    'τ=0.01',
    xy=(davis_sparsity[0], davis_recall[0]),
    xytext=(93.6, 66.5),
    fontsize=9, color='#2C3E50',
    ha='center',
    arrowprops=dict(arrowstyle='-',
                    color='#BDC3C7', lw=0.9)
)

# τ=0.05 — above-left, away from KITTI label
ax1.annotate(
    'τ=0.05',
    xy=(davis_sparsity[1], davis_recall[1]),
    xytext=(97.0, 96.5),
    fontsize=9, color='#2C3E50',
    ha='center',
    arrowprops=dict(arrowstyle='-',
                    color='#BDC3C7', lw=0.9)
)

# τ=0.10★ — below point, blue and bold
ax1.annotate(
    'τ=0.10 ★',
    xy=(davis_sparsity[2], davis_recall[2]),
    xytext=(99.5, 87.5),
    fontsize=9.5, color='#2980B9',
    fontweight='bold', ha='center',
    arrowprops=dict(arrowstyle='-',
                    color='#2980B9', lw=1.2)
)

# τ=0.20 collapse annotation — red, off-axis
ax1.annotate(
    'τ=0.20: recall\ncollapses to 22.6%\n(off-axis)',
    xy=(davis_sparsity[2], davis_recall[2]),
    xytext=(97.4, 71.5),
    fontsize=8,
    color='#E74C3C',
    ha='center',
    bbox=dict(boxstyle='round,pad=0.3',
              facecolor='#FDEDEC',
              edgecolor='#E74C3C',
              alpha=0.92),
    arrowprops=dict(
        arrowstyle='->',
        color='#E74C3C',
        lw=1.5,
        connectionstyle='arc3,rad=0.25'
    )
)

# Best balance info box — bottom-left
ax1.text(
    92.15, 61.5,
    'Best Balance (τ=0.10):\n'
    'Sparsity: 99.8%\n'
    'Recall:     93.9%',
    fontsize=8.5, color='#2980B9',
    va='bottom', zorder=10,
    bbox=dict(boxstyle='round,pad=0.4',
              facecolor='#EBF5FB',
              edgecolor='#2980B9',
              alpha=0.95)
)

ax1.set_xlim(92, 100.5)
ax1.set_ylim(60, 104)
ax1.set_xlabel('Background Sparsity (%)',
               fontsize=11, labelpad=4)
ax1.set_ylabel('Object Recall (IoG %)',
               fontsize=11, labelpad=4)
ax1.set_title('Sparsity–Recall Pareto Frontier',
              fontsize=12, fontweight='bold', pad=10)
ax1.legend(fontsize=9, loc='upper left',
           framealpha=0.92, edgecolor='lightgray')
ax1.grid(True, alpha=0.22, linestyle='--',
         linewidth=0.7)
ax1.tick_params(labelsize=9.5)

# ════════════════════════════════════════════════════════════
# RIGHT: ABLATION STUDY
# ════════════════════════════════════════════════════════════
bar_colors = [
    '#27AE60',  # Baseline
    '#E74C3C',  # w/o Temporal
    '#E67E22',  # w/o Global
    '#F39C12',  # w/o SAC
    '#8E44AD',  # w/o DoG
    '#95A5A6',  # w/o VOR
]

bars = ax2.bar(
    range(len(variants)), sparsity,
    color=bar_colors,
    edgecolor='white',
    linewidth=0.8,
    width=0.62,
    zorder=3
)

# Baseline dashed line
ax2.axhline(
    y=98.25, color='#27AE60',
    linestyle='--', linewidth=1.5,
    alpha=0.65, zorder=2,
    label='Baseline (98.25%)'
)

# Drop delta labels — inside bars, white text
drops = {1: '−15.5%', 2: '−3.6%'}
for idx, label in drops.items():
    mid_y = 75 + (sparsity[idx] - 75) / 2
    ax2.text(
        idx, mid_y,
        label,
        ha='center', va='center',
        fontsize=9, color='white',
        fontweight='bold', zorder=4
    )

# ── FIX 1: DSI annotation INSIDE the SAC bar ─────────────────
# Value label on top (clear of everything)
ax2.text(
    3,
    sparsity[3] + 0.35,
    f'{sparsity[3]:.1f}%',
    ha='center', va='bottom',
    fontsize=8, color='#2C3E50',
    zorder=5
)
# DSI label inside bar (mid-height of SAC bar)
sac_mid = 75 + (sparsity[3] - 75) / 2
ax2.text(
    3, sac_mid,
    'DSI=0\n(dir. lost)',
    ha='center', va='center',
    fontsize=7.5, color='white',
    fontweight='bold', zorder=4,
    bbox=dict(boxstyle='round,pad=0.2',
              facecolor='#E67E22',
              alpha=0.75,
              edgecolor='none')
)

# Value labels on all OTHER bars (not SAC — handled above)
for i, (bar, val) in enumerate(zip(bars, sparsity)):
    if i == 3:
        continue   # SAC handled separately above
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.35,
        f'{val:.1f}%',
        ha='center', va='bottom',
        fontsize=8, color='#2C3E50',
        zorder=5
    )

ax2.set_xticks(range(len(variants)))
ax2.set_xticklabels(variants, fontsize=9.5)
ax2.set_ylabel('Mean Sparsity (%)',
               fontsize=11, labelpad=4)
ax2.set_title('Ablation Study',
              fontsize=12, fontweight='bold', pad=10)
ax2.set_ylim(74, 103)
ax2.grid(True, alpha=0.22, axis='y',
         linestyle='--', linewidth=0.7, zorder=1)
ax2.legend(fontsize=9, loc='lower right',
           framealpha=0.92, edgecolor='lightgray')
ax2.tick_params(axis='y', labelsize=9.5)
ax2.spines[['top', 'right']].set_visible(False)
ax1.spines[['top', 'right']].set_visible(False)

# ── SAVE ─────────────────────────────────────────────────────
plt.tight_layout(pad=1.8)

for fmt in ['pdf', 'png']:
    p = OUT / f"fig5_pareto_ablation.{fmt}"
    plt.savefig(
        str(p), dpi=300,
        bbox_inches='tight',
        facecolor='white',
        pad_inches=0.08
    )
    print(f"Saved → {p}")

plt.close()
print("Figure 5 done.")