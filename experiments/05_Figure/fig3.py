# experiments/figures/fig3.py — final definitive version

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    'input':    '#2471A3',
    'spatial':  '#154360',
    'temporal': '#6C3483',
    'output':   '#1E8449',
    'sac':      '#7E5109',
    'final':    '#C0392B',
}

# ── FIGURE — single axes, draw everything manually ───────────
fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# ── PANEL BACKGROUNDS ────────────────────────────────────────
# Left panel: x=0.1 to 6.8
# Right panel: x=8.8 to 15.9
ax.add_patch(FancyBboxPatch(
    (0.1, 0.3), 6.7, 8.4,
    boxstyle="round,pad=0.1",
    facecolor='#EBF5FB', edgecolor='#AED6F1',
    linewidth=2, zorder=0
))
ax.add_patch(FancyBboxPatch(
    (8.8, 0.3), 7.0, 8.4,
    boxstyle="round,pad=0.1",
    facecolor='#EAFAF1', edgecolor='#A9DFBF',
    linewidth=2, zorder=0
))

# Panel titles
ax.text(3.45, 8.45, 'Biological Retina',
        ha='center', fontsize=13, fontweight='bold',
        color='#1A5276')
ax.text(12.3, 8.45, 'RetiGate Computational Model',
        ha='center', fontsize=13, fontweight='bold',
        color='#1E8449')

# ── ROW Y POSITIONS (bottom of each box) ────────────────────
# 5 rows, evenly spaced
row_bottoms = [6.8, 5.4, 4.0, 2.6, 0.65]
row_h       = 1.1

# Left panel box: x=0.3, width=5.2
# Right panel box: x=9.0, width=5.8
LX, LW = 0.35, 5.2    # left box x, width
RX, RW = 9.05, 5.5    # right box x, width
LC      = 2.95         # left center x
RC      = 11.80        # right center x

# ── DRAW FUNCTIONS ───────────────────────────────────────────
def draw_box(x, y, w, h, color, title, sub=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor='white',
        linewidth=2, zorder=3
    ))
    cx = x + w/2
    if sub:
        ax.text(cx, y + h*0.65, title,
                ha='center', va='center',
                fontsize=10.5, fontweight='bold',
                color='white', zorder=4)
        ax.text(cx, y + h*0.28, sub,
                ha='center', va='center',
                fontsize=8, color='#D6EAF8',
                zorder=4, wrap=True)
    else:
        ax.text(cx, y + h*0.5, title,
                ha='center', va='center',
                fontsize=10.5, fontweight='bold',
                color='white', zorder=4)


def draw_connector(y_top_bottom, y_bot_top):
    """Thin line connecting two boxes on same side."""
    pass  # intentionally empty — clean look without arrows


def draw_note(x, y_center, text, right=False):
    """Small italic note badge."""
    ha = 'right' if right else 'left'
    ax.text(
        x, y_center, text,
        ha=ha, va='center',
        fontsize=7.5, color='#2C3E50',
        style='italic',
        bbox=dict(boxstyle='round,pad=0.2',
                  facecolor='white',
                  edgecolor='#BDC3C7',
                  alpha=0.85),
        zorder=6
    )


def draw_param(x, y_center, text, color):
    """Parameter badge — placed to the RIGHT of right panel."""
    ax.text(
        x, y_center, text,
        ha='left', va='center',
        fontsize=7.5, color=color,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25',
                  facecolor='white',
                  edgecolor=color,
                  linewidth=1.3,
                  alpha=0.95),
        zorder=6
    )


def draw_arrow_between(x, y_from, y_to):
    """Downward arrow between boxes."""
    ax.annotate(
        '',
        xy=(x, y_to + row_h + 0.02),
        xytext=(x, y_from - 0.02),
        arrowprops=dict(
            arrowstyle='-|>',
            color='#85929E',
            lw=1.8,
            mutation_scale=14
        ),
        zorder=2
    )


def draw_sac(x, y_center, sub):
    """SAC mini-box branching right from main box."""
    sx = x + 0.15
    sy = y_center - 0.45
    sw, sh = 1.1, 0.9
    ax.add_patch(FancyBboxPatch(
        (sx, sy), sw, sh,
        boxstyle="round,pad=0.06",
        facecolor=COLORS['sac'],
        edgecolor='white',
        linewidth=1.5, zorder=3
    ))
    ax.text(sx + sw/2, sy + sh*0.62, 'SAC',
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color='white', zorder=4)
    ax.text(sx + sw/2, sy + sh*0.25, sub,
            ha='center', va='center',
            fontsize=7, color='#F0D9B5', zorder=4)
    # Arrow from main box edge to SAC
    ax.annotate(
        '',
        xy=(sx, sy + sh/2),
        xytext=(sx - 0.18, sy + sh/2),
        arrowprops=dict(
            arrowstyle='-|>',
            color=COLORS['sac'],
            lw=1.3, mutation_scale=11
        ),
        zorder=2
    )


# ── SOURCE LABELS ────────────────────────────────────────────
ax.text(LC, 8.1, '☀  Incoming Light',
        ha='center', fontsize=9.5,
        color='#D4AC0D', fontweight='bold')
ax.text(RC, 8.1, '🎬  Video Frame  Iₜ',
        ha='center', fontsize=9.5,
        color='#2C3E50', fontweight='bold')

# ── BIO ROWS ─────────────────────────────────────────────────
bio = [
    (COLORS['input'],    'Photoreceptors',
     'Rods & cones — spatial sampling',
     'Spatial\nsampling'),
    (COLORS['spatial'],  'Bipolar Cells',
     'Center-surround receptive fields',
     'DoG\nfiltering'),
    (COLORS['temporal'], 'Amacrine Cells',
     'Lateral inhibition + temporal memory',
     'Leaky\nintegration'),
    (COLORS['output'],   'Ganglion Cells',
     'Sparse spikes → motion saliency',
     'Global\ninhibition'),
    (COLORS['final'],    'Optic Nerve',
     'Signal to visual cortex',
     None),
]

for i, (color, title, sub, note) in enumerate(bio):
    yb = row_bottoms[i]
    draw_box(LX, yb, LW, row_h, color, title, sub)
    if note:
        draw_note(LX - 0.08, yb + row_h/2, note)
    # Downward arrow to next box
    if i < len(bio) - 1:
        draw_arrow_between(LC,
                           row_bottoms[i],
                           row_bottoms[i+1])

# SAC on bio side
draw_sac(LX + LW, row_bottoms[2] + row_h/2, 'Dir.\nselect.')

# ── RETIGATE ROWS ────────────────────────────────────────────
reti = [
    (COLORS['input'],   'VOS Stage',
     'Ego-motion cancellation (ORB homography)',
     'use_vos\n=True',    '#2471A3'),
    (COLORS['spatial'], 'DoG Filter',
     '| G_c * I  −  G_s * I |',
     'σ_c=1.5\nσ_s=4.0',  '#154360'),
    (COLORS['temporal'],'Leaky Integration',
     'M̂_t = λ·M_t + (1−λ)·M̂_{t−1}',
     'λ=0.10\nω=1.5',     '#6C3483'),
    (COLORS['output'],  'Global Inhibition',
     'M_out = max(0, M_t − M̂_t − ω·M̄)',
     'τ = 0.10',           '#1E8449'),
    (COLORS['final'],   'ROI  →  YOLO',
     'bbox(active > τ)  →  sparse inference',
     '−38.6%\nenergy',    '#C0392B'),
]

# Param badges placed at x=15.7 (right of right panel)
PARAM_X = 15.72

for i, (color, title, sub, param, pc) in enumerate(reti):
    yb = row_bottoms[i]
    draw_box(RX, yb, RW, row_h, color, title, sub)
    draw_param(PARAM_X, yb + row_h/2, param, color=pc)
    # Downward arrow to next box
    if i < len(reti) - 1:
        draw_arrow_between(RC,
                           row_bottoms[i],
                           row_bottoms[i+1])

# SAC on retigate side
draw_sac(RX + RW, row_bottoms[2] + row_h/2, 'Shift &\nsubtract')

# ── MIDDLE CONNECTORS ────────────────────────────────────────
# Horizontal dashed lines connecting matching rows
# with color-coded dots showing the analogy
'''
MID_X = 7.9
for i, yb in enumerate(row_bottoms):
    y_center = yb + row_h / 2
    # Horizontal connector line
    ax.plot([LX + LW + 0.05, RX - 0.08],
            [y_center, y_center],
            color='#E67E22', lw=1.2,
            linestyle='dashed', alpha=0.5,
            zorder=1)
    # Center label
    ax.text(MID_X, y_center, '≡',
            ha='center', va='center',
            fontsize=14, color='#E67E22',
            fontweight='bold', zorder=3,
            alpha=0.7)

# Middle title
ax.text(MID_X, 8.45, 'Maps to',
        ha='center', fontsize=9,
        color='#E67E22', fontweight='bold')
'''
# ── LEGEND ───────────────────────────────────────────────────
patches = [
    mpatches.Patch(facecolor=COLORS['input'],
                   label='Input processing'),
    mpatches.Patch(facecolor=COLORS['spatial'],
                   label='Spatial filtering (DoG)'),
    mpatches.Patch(facecolor=COLORS['temporal'],
                   label='Temporal integration'),
    mpatches.Patch(facecolor=COLORS['output'],
                   label='Inhibition / output'),
    mpatches.Patch(facecolor=COLORS['sac'],
                   label='Directional selectivity'),
    mpatches.Patch(facecolor=COLORS['final'],
                   label='Final output'),
]
fig.legend(
    handles=patches,
    loc='lower center', ncol=6,
    fontsize=8.5, framealpha=0.9,
    bbox_to_anchor=(0.5, 0.0),
    edgecolor='lightgray'
)

fig.suptitle(
    'RetiGate: Computational Model Inspired by '
    'Retinal Ganglion Cell Circuitry',
    fontsize=14, fontweight='bold', y=0.99
)

for fmt in ['pdf', 'png']:
    p = OUT / f"fig3_bio_analogy.{fmt}"
    plt.savefig(str(p), dpi=300,
                bbox_inches='tight',
                facecolor='white')
    print(f"Saved → {p}")

plt.close()
print("Done.")