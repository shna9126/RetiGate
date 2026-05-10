# experiments/figures/fig9_latency_breakdown.py
# Standalone version — no retigate imports needed
# Uses locked numbers from FINAL_NUMBERS.md

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── LOCKED NUMBERS (Mac M3 Pro, n=500) ───────────────────────
stages = {
    'VOS\n(ORB+RANSAC)':       7.93,   # 35.7%
    'Spatial Filter\n(DoG)':   11.88,  # 53.4%
    'Temporal\nIntegration':   0.27,   # 1.2%
    'Global\nInhibition':      0.53,   # 2.4%
    'SAC Directional\nTail':   0.91,   # 4.1%
    'ROI\nExtraction':         0.73,   # 3.3%
}
total_sensing = sum(stages.values())   # = 22.25ms

colors = [
    '#E74C3C',   # VOS — red (largest, most notable)
    '#2980B9',   # DoG — blue (largest)
    '#27AE60',   # Temporal
    '#F39C12',   # Global
    '#8E44AD',   # SAC
    '#16A085',   # ROI
]

# ── FIGURE ───────────────────────────────────────────────────
fig, (ax_pie, ax_bar) = plt.subplots(
    1, 2,
    figsize=(13, 5.5),
    facecolor='white',
    gridspec_kw={'wspace': 0.35}
)

# ── PIE CHART — fix label overlap ────────────────────────────
vals   = list(stages.values())
labels = list(stages.keys())

# Consolidate slices < 3% into a visible label
# All slices here are visible — use explode for DoG
explode = [0.02] * len(vals)
explode[1] = 0.06   # pull out DoG (largest)
explode[0] = 0.04   # pull out VOS (second largest)

wedges, texts, autotexts = ax_pie.pie(
    vals,
    labels=None,        # no inline labels — use legend
    colors=colors,
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.78,
    explode=explode,
    wedgeprops=dict(linewidth=1.5, edgecolor='white')
)

# Style percentage text
for at in autotexts:
    at.set_fontsize(8.5)
    at.set_fontweight('bold')
    at.set_color('white')

# Legend outside pie — no overlap possible
ax_pie.legend(
    wedges, labels,
    title='Pipeline Stage',
    title_fontsize=9,
    loc='center left',
    bbox_to_anchor=(-0.35, 0.5),
    fontsize=8.5,
    framealpha=0.92,
    edgecolor='lightgray'
)

ax_pie.set_title(
    'RetiGate Sensing Overhead\nby Pipeline Stage',
    fontsize=12, fontweight='bold', pad=10
)

# Total annotation in centre
ax_pie.text(
    0, 0,
    f'Total\n{total_sensing:.1f} ms',
    ha='center', va='center',
    fontsize=10, fontweight='bold',
    color='#2C3E50'
)

# ── BAR CHART — sensing vs inference only ────────────────────
# FIX: Remove "Sparse Total" — it adds M3 + T4 (invalid)
# Show only: RetiGate sensing | YOLO dense inference
# Add note that they run in parallel

bar_labels = [
    'RetiGate Sensing\n(Apple M3 Pro, CPU)',
    'YOLO Dense\nInference\n(NVIDIA T4, GPU)',
]
bar_vals   = [total_sensing, 31.76]
bar_colors = ['#27AE60', '#E74C3C']

bars = ax_bar.bar(
    bar_labels, bar_vals,
    color=bar_colors,
    width=0.42,
    edgecolor='white',
    linewidth=0.8,
    zorder=3
)

# Value labels on bars
for bar, val in zip(bars, bar_vals):
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.4,
        f'{val:.2f} ms',
        ha='center', va='bottom',
        fontsize=11, fontweight='bold',
        color='#2C3E50'
    )

# Parallel operation annotation
ax_bar.annotate(
    '',
    xy=(1, max(bar_vals) * 0.5),
    xytext=(0, max(bar_vals) * 0.5),
    arrowprops=dict(
        arrowstyle='<->',
        color='#7F8C8D',
        lw=1.5
    )
)
ax_bar.text(
    0.5, max(bar_vals) * 0.52,
    'Run in\nparallel',
    ha='center', va='bottom',
    fontsize=8.5, color='#7F8C8D',
    style='italic'
)

# Effective latency annotation
effective = max(total_sensing, 31.76)
ax_bar.axhline(
    y=effective,
    color='#2C3E50', linewidth=1.2,
    linestyle='--', alpha=0.4, zorder=2
)
ax_bar.text(
    1.28, effective + 0.3,
    f'Pipeline latency\n= max(sensing, inference)\n'
    f'= {effective:.2f} ms',
    ha='center', va='bottom',
    fontsize=8, color='#2C3E50',
    bbox=dict(boxstyle='round,pad=0.3',
              facecolor='#F8F9FA',
              edgecolor='#BDC3C7',
              alpha=0.92)
)

ax_bar.set_ylabel('Latency (ms)', fontsize=11,
                  labelpad=4)
ax_bar.set_title(
    'Sensing vs Inference Latency\n'
    '(separate hardware, pipelined)',
    fontsize=12, fontweight='bold', pad=10
)
ax_bar.set_ylim(0, 42)
ax_bar.grid(True, alpha=0.18, axis='y',
            linestyle='--', linewidth=0.7,
            zorder=1)
ax_bar.spines[['top', 'right']].set_visible(False)
ax_bar.tick_params(labelsize=9.5)

# Hardware note at bottom
ax_bar.text(
    0.5, -0.22,
    'Sensing: Apple M3 Pro CPU (n=500 frames)\n'
    'Inference: NVIDIA Tesla T4 GPU (CUDA events, n=200)\n'
    'Hardware is separate — latencies are NOT additive',
    transform=ax_bar.transAxes,
    ha='center', va='top',
    fontsize=7.5, color='#7F8C8D',
    style='italic'
)

# ── SAVE ─────────────────────────────────────────────────────
plt.tight_layout(pad=1.5)
plt.subplots_adjust(bottom=0.18)

for fmt in ['pdf', 'png']:
    p = OUT / f"fig9_latency_breakdown.{fmt}"
    plt.savefig(
        str(p), dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    print(f"Saved → {p}")

plt.close()
print("Fig 9 done.")