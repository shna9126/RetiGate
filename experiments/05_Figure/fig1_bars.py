# experiments/figures/fig1_bars.py
# Outputs clean bar chart PNG for Canva assembly
# Transparent background so it composites cleanly

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

OUT = Path("figures/fig1_parts")
OUT.mkdir(parents=True, exist_ok=True)

C_RED   = '#C0392B'
C_GREEN = '#1A7A3A'

fig = plt.figure(
    figsize=(4.5, 6.0),
    facecolor='white',
    dpi=300
)

gs = gridspec.GridSpec(
    2, 1,
    figure=fig,
    hspace=0.70,
    top=0.93,
    bottom=0.10,
    left=0.02,
    right=0.82
)

ax_lat  = fig.add_subplot(gs[0])
ax_enrg = fig.add_subplot(gs[1])

# ── LATENCY ──────────────────────────────────────────────────
lats    = [31.8, 17.4]
labels  = ['Dense', 'RetiGate']
colors  = [C_RED, C_GREEN]

bars = ax_lat.barh(
    labels, lats,
    color=colors,
    edgecolor='none',
    height=0.55
)
ax_lat.set_xlim(0, 48)
ax_lat.set_xlabel('ms', fontsize=13, labelpad=4)
ax_lat.set_title('Latency', fontsize=15,
                 fontweight='bold', pad=8)
ax_lat.tick_params(axis='x', labelsize=11)
ax_lat.tick_params(axis='y', labelsize=12)
ax_lat.spines[['top','right','left']].set_visible(False)
ax_lat.grid(True, alpha=0.2, axis='x',
            linestyle='--', linewidth=0.8)
ax_lat.set_axisbelow(True)

# Value inside each bar
for bar, val in zip(bars, lats):
    ax_lat.text(
        val / 2,
        bar.get_y() + bar.get_height() / 2,
        f'{val} ms',
        ha='center', va='center',
        fontsize=12, color='white',
        fontweight='bold'
    )

# Speedup badge — right of bars
ax_lat.annotate(
    '1.83×',
    xy=(31.8, 1),
    xytext=(40, 0.5),
    fontsize=13, color=C_GREEN,
    fontweight='bold', va='center',
    bbox=dict(boxstyle='round,pad=0.35',
              facecolor='#EAFAF1',
              edgecolor=C_GREEN,
              linewidth=1.8)
)

# ── ENERGY ───────────────────────────────────────────────────
energies = [2107, 1294]

bars2 = ax_enrg.barh(
    labels, energies,
    color=colors,
    edgecolor='none',
    height=0.55
)
ax_enrg.set_xlim(0, 2800)
ax_enrg.set_xlabel('mJ / frame', fontsize=13, labelpad=4)
ax_enrg.set_title('Energy', fontsize=15,
                  fontweight='bold', pad=8)
ax_enrg.tick_params(axis='x', labelsize=11)
ax_enrg.tick_params(axis='y', labelsize=12)
ax_enrg.spines[['top','right','left']].set_visible(False)
ax_enrg.grid(True, alpha=0.2, axis='x',
             linestyle='--', linewidth=0.8)
ax_enrg.set_axisbelow(True)

for bar, val in zip(bars2, energies):
    ax_enrg.text(
        val / 2,
        bar.get_y() + bar.get_height() / 2,
        f'{val}',
        ha='center', va='center',
        fontsize=12, color='white',
        fontweight='bold'
    )

ax_enrg.annotate(
    '−38.6%',
    xy=(2107, 1),
    xytext=(2400, 0.5),
    fontsize=13, color=C_GREEN,
    fontweight='bold', va='center',
    bbox=dict(boxstyle='round,pad=0.35',
              facecolor='#EAFAF1',
              edgecolor=C_GREEN,
              linewidth=1.8)
)

# ── SAVE ─────────────────────────────────────────────────────
out = OUT / "bars_latency_energy.png"
plt.savefig(
    str(out), dpi=300,
    bbox_inches='tight',
    facecolor='white',
    pad_inches=0.08
)
plt.close()
print(f"Saved → {out}")