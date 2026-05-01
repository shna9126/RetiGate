import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Locked Data Points
davis_sparsity = [92.9, 98.6, 99.8]
davis_recall = [70.2, 93.3, 93.9]
kitti_pt = [98.25, 95.7]

fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
ax.set_facecolor('#fdfdfd')

# 1. Target Region (Success Zone) - Subtle shading
ax.axvspan(98.2, 100.5, color='#e8f5e9', alpha=0.8, zorder=1, label='Target Region')

# 2. Main Pareto Frontier - Emphasizing the "Knee"
ax.plot(davis_sparsity, davis_recall, marker='o', color='#1f77b4', linewidth=2.5, 
        markersize=10, markerfacecolor='white', markeredgewidth=2, zorder=5, 
        label='DAVIS 2017 (90 seq)')

# 3. KITTI Benchmark
ax.plot(kitti_pt[0], kitti_pt[1], '^', color='#2ca02c', markersize=12, zorder=6, 
        label='KITTI Tracking (21 seq)')

# 4. Clean Annotations - Using 'offset points' to avoid collisions
# Tau = 0.01
ax.annotate(r'$\tau=0.01$', xy=(92.9, 70.2), xytext=(10, -15), textcoords='offset points', 
             fontsize=10, color='#444', fontweight='bold',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='#999'))

# Tau = 0.05 (The Elbow Start)
ax.annotate(r'$\tau=0.05$', xy=(98.6, 93.3), xytext=(-40, 15), textcoords='offset points', 
             fontsize=10, color='#444', fontweight='bold',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-.2', color='#999'))

# Tau = 0.10 (Golden Constant)
ax.annotate(r'$\tau=0.10$ ★', xy=(99.8, 93.9), xytext=(-35, -25), textcoords='offset points', 
             fontsize=11, color='#1f77b4', fontweight='black',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='#1f77b4', lw=1.5))

# 5. System Collapse - Graceful Curve
ax.annotate('System Collapse\n' + r'($\tau=0.20$, Recall 22.6%)', 
             xy=(99.9, 93.5), xytext=(98.5, 75),
             fontsize=9, color='#d32f2f', fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffebee', edgecolor='#d32f2f', alpha=0.8),
             arrowprops=dict(arrowstyle='->,head_width=0.4,head_length=0.7', 
                             connectionstyle='arc3,rad=0.35', color='#d32f2f', lw=2))

# 6. Stylized Info Box
info_box = ('BEST BALANCE ($\\tau=0.10$)\n'
            'Sparsity: 99.8%\n'
            'Recall: 93.9%')
ax.text(92.5, 62, info_box, fontsize=10, color='#0d47a1', fontweight='bold',
        bbox=dict(boxstyle='round4,pad=0.8', facecolor='#e3f2fd', edgecolor='#1f77b4', linewidth=1.2))

# 7. Final Polish
ax.set_xlim(92, 100.5)
ax.set_ylim(60, 105)
ax.set_xlabel('Background Sparsity (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Object Recall (IoG %)', fontsize=12, fontweight='bold')
ax.set_title('Sparsity–Recall Pareto Frontier', fontsize=14, fontweight='black', pad=15)
ax.grid(True, alpha=0.2, linestyle='--')
ax.legend(loc='upper left', frameon=True, facecolor='white', fontsize=9)

plt.tight_layout()
plt.show()