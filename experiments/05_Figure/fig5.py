# experiments/figures/fig5.py
# Full replacement with all 4 visual adjustments

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── DATA ─────────────────────────────────────────────────────
davis_tau      = [0.01, 0.05, 0.10, 0.20]
davis_sparsity = [92.9, 98.6, 99.8, 100.0]
davis_recall   = [70.2, 93.3, 93.9, 22.6]

kitti_sparsity = 98.25
kitti_recall   = 95.68

variants = ['Baseline', 'w/o\nTemporal', 'w/o\nGlobal',
            'w/o\nSAC',   'w/o\nDoG',    'w/o\nVOR']
sparsity = [98.25, 82.80, 94.66, 98.25, 99.20, 98.20]

# ── FIGURE ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(12, 4.5), facecolor='white'
)

# ── LEFT: PARETO CURVE — clean version ──────────────────────

# FIX 4: Bolder Success Zone
ax1.axvspan(
    98.6, 99.8,
    alpha=0.15, color='#27AE60',
    zorder=1, label='Success Zone'
)

# FIX 3: Vertical dotted line at KITTI x-position
ax1.axvline(
    x=kitti_sparsity,
    color='#27AE60', linewidth=1.2,
    linestyle=':', alpha=0.5, zorder=2
)

# DAVIS curve — only τ=0.01, 0.05, 0.10 on plot
ax1.plot(
    davis_sparsity[:3], davis_recall[:3],
    'o-', color='#2980B9', linewidth=2.5,
    markersize=9, zorder=5,
    label='DAVIS 2017 (90 seq)'
)

# KITTI triangle
ax1.plot(
    kitti_sparsity, kitti_recall,
    '^', color='#27AE60', markersize=12,
    zorder=6, label='KITTI Tracking (21 seq)'
)

# τ=0.20 collapse — shown as arrow pointing DOWN-RIGHT
# from last visible point, with annotation
# instead of a line going off-plot
ax1.annotate(
    'τ=0.20: recall\ncollapses to 22.6%\n(off-axis)',
    xy=(davis_sparsity[2], davis_recall[2]),
    xytext=(97.8, 70),
    fontsize=8,
    color='#E74C3C',
    bbox=dict(boxstyle='round,pad=0.3',
              facecolor='#FDEDEC',
              edgecolor='#E74C3C',
              alpha=0.9),
    arrowprops=dict(
        arrowstyle='->',
        color='#E74C3C',
        lw=1.5,
        connectionstyle='arc3,rad=0.3'
    )
)

# ── POINT LABELS — no collisions ────────────────────────────

# τ=0.01 — below and right
ax1.annotate(
    'τ=0.01',
    xy=(davis_sparsity[0], davis_recall[0]),
    xytext=(93.3, 67),
    fontsize=9, color='#2C3E50',
    arrowprops=dict(arrowstyle='-',
                    color='#BDC3C7', lw=1.0)
)

# τ=0.05 — above the point, shifted left to avoid KITTI
ax1.annotate(
    'τ=0.05',
    xy=(davis_sparsity[1], davis_recall[1]),
    xytext=(96.5, 96),
    fontsize=9, color='#2C3E50',
    arrowprops=dict(arrowstyle='-',
                    color='#BDC3C7', lw=1.0)
)

# τ=0.10★ — below point
ax1.annotate(
    'τ=0.10 ★',
    xy=(davis_sparsity[2], davis_recall[2]),
    xytext=(99.2, 88),
    fontsize=9.5, color='#2980B9',
    fontweight='bold',
    arrowprops=dict(arrowstyle='-',
                    color='#2980B9', lw=1.2)
)

# KITTI label — right of triangle
ax1.annotate(
    'KITTI\n95.7%',
    xy=(kitti_sparsity, kitti_recall),
    xytext=(98.35, 98.5),
    fontsize=8, color='#27AE60',
    fontweight='bold'
)

# Golden box — bottom-left empty space
ax1.text(
    92.15, 62,
    'Best Balance (τ=0.10):\nSparsity: 99.8%\nRecall: 93.9%',
    fontsize=8.5, color='#2980B9',
    bbox=dict(boxstyle='round,pad=0.4',
              facecolor='#EBF5FB',
              edgecolor='#2980B9',
              alpha=0.95),
    va='bottom', zorder=10
)

# ── AXIS SETTINGS ────────────────────────────────────────────
ax1.set_xlim(92, 100.5)
ax1.set_ylim(60, 103)
ax1.set_xlabel('Background Sparsity (%)', fontsize=11)
ax1.set_ylabel('Object Recall (IoG %)', fontsize=11)
ax1.set_title('Sparsity–Recall Pareto Frontier',
              fontsize=12, fontweight='bold', pad=10)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.25, linestyle='--')


# ════════════════════════════════════════════════════════════
# RIGHT: ABLATION — unchanged from previous version
# ════════════════════════════════════════════════════════════
bar_colors = [
    '#27AE60',  # Baseline
    '#E74C3C',  # no_temporal
    '#E67E22',  # no_global
    '#F39C12',  # no_sac
    '#8E44AD',  # no_dog
    '#95A5A6',  # no_vor
]

bars = ax2.bar(
    range(len(variants)), sparsity,
    color=bar_colors,
    edgecolor='white',
    linewidth=0.8,
    width=0.65,
    zorder=3
)

ax2.axhline(
    y=98.25, color='#27AE60',
    linestyle='--', linewidth=1.5,
    alpha=0.6, zorder=2,
    label='Baseline (98.25%)'
)

# Drop annotations inside bars
drops = {1: '−15.5%', 2: '−3.6%'}
for idx, label in drops.items():
    ax2.annotate(
        label,
        xy=(idx, sparsity[idx]),
        xytext=(idx, sparsity[idx] - 3.5),
        ha='center', fontsize=8.5,
        color='white', fontweight='bold'
    )

# DSI=0 annotation for no_sac
ax2.annotate(
    'DSI=0\n(dir. lost)',
    xy=(3, sparsity[3]),
    xytext=(3, sparsity[3] + 1.0),
    ha='center', fontsize=7.5,
    color='#E67E22'
)

ax2.set_xticks(range(len(variants)))
ax2.set_xticklabels(variants, fontsize=9)
ax2.set_ylabel('Mean Sparsity (%)', fontsize=11)
ax2.set_title('Ablation Study',
              fontsize=12, fontweight='bold', pad=10)
ax2.set_ylim(74, 102.5)
ax2.grid(True, alpha=0.25, axis='y',
         linestyle='--', zorder=1)
ax2.legend(fontsize=9, loc='lower right', framealpha=0.9)

# Value labels on bars
for bar, val in zip(bars, sparsity):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.3,
        f'{val:.1f}%',
        ha='center', va='bottom',
        fontsize=8, color='#2C3E50'
    )

# ── SAVE ─────────────────────────────────────────────────────
plt.tight_layout(pad=2.0)

for fmt in ['pdf', 'png']:
    p = OUT / f"fig5_pareto_ablation.{fmt}"
    plt.savefig(str(p), dpi=300,
                bbox_inches='tight',
                facecolor='white')
    print(f"Saved → {p}")

plt.close()
print("Figure 5 done.")