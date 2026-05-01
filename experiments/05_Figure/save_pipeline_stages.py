# experiments/figures/save_pipeline_stages.py
# Run this FIRST — generates the thumbnail images
# that Figure 2 will be built around

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from retigate import RetinaCore

# --- CONFIG ---
# Pick a clean frame with a visible car
# Sequence 0000, frame 10 is usually good
IMG_PATH = Path(
    "data/kitti/data_tracking_image/image_02/0005/000100.png"
)
OUT_DIR = Path("figures/assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- LOAD AND WARMUP ---
retina = RetinaCore.golden_baseline()
img    = cv2.imread(str(IMG_PATH))

if img is None:
    raise FileNotFoundError(f"Cannot load {IMG_PATH}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape
print(f"Frame: {W}×{H}")

# Warmup retina with 8 frames (temporal state)
warmup_dir = IMG_PATH.parent
warmup_paths = sorted(list(warmup_dir.glob("*.png")))[:8]
for p in warmup_paths:
    wf = cv2.imread(str(p))
    if wf is not None:
        wg = cv2.cvtColor(wf, cv2.COLOR_BGR2GRAY)
        retina.process_frame(wg)

print(f"Warmed up on {len(warmup_paths)} frames")

# --- PROCESS TARGET FRAME ---
rout = retina.process_frame(gray)

# --- EXTRACT STAGES ---
# Stage 0: Raw input
stage_input = gray.copy()

# Stage 1: DoG (bipolar) — recompute manually
img_f  = gray.astype(np.float32) / 255.0
m_c    = cv2.filter2D(img_f, -1, retina.m_center_k,
                       borderType=cv2.BORDER_REFLECT)
m_s    = cv2.filter2D(img_f, -1, retina.m_surround_k,
                       borderType=cv2.BORDER_REFLECT)
stage_dog = np.abs(m_c - m_s)

# Stage 2: Amacrine (temporal integration)
stage_amacrine = retina.amacrine_state.copy()

# Stage 3: Ganglion (after inhibition)
stage_ganglion = rout['M_Motion'].copy()

# Stage 4: Active mask (binary)
stage_mask = rout['active_mask'].astype(np.float32)

# Stage 5: ROI on original image
roi    = retina.get_roi_bbox(rout, frame_shape=img.shape)
stage_roi = img.copy()
if roi:
    cv2.rectangle(stage_roi,
                  (roi[0], roi[1]), (roi[2], roi[3]),
                  (0, 255, 0), 3)
    # Dim everything outside ROI
    overlay = stage_roi.copy()
    mask_roi = np.zeros(img.shape[:2], dtype=np.uint8)
    mask_roi[roi[1]:roi[3], roi[0]:roi[2]] = 255
    stage_roi[mask_roi == 0] = (
        stage_roi[mask_roi == 0] * 0.35
    ).astype(np.uint8)

# --- NORMALIZE FOR DISPLAY ---
def norm(x):
    """Normalize array to [0, 1] for display."""
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

# --- SAVE INDIVIDUAL STAGE IMAGES ---
stages = {
    '0_input':     (stage_input,             'gray'),
    '1_dog':       (norm(stage_dog),         'hot'),
    '2_amacrine':  (norm(stage_amacrine),    'hot'),
    '3_ganglion':  (norm(stage_ganglion),    'hot'),
    '4_mask':      (stage_mask,              'gray'),
}

for name, (data, cmap) in stages.items():
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.imshow(data, cmap=cmap, aspect='auto')
    ax.axis('off')
    plt.tight_layout(pad=0)
    save_path = OUT_DIR / f"stage_{name}.png"
    plt.savefig(str(save_path), dpi=200,
                bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved {save_path}")

# Save ROI image separately (color)
cv2.imwrite(
    str(OUT_DIR / "stage_5_roi.png"),
    stage_roi
)
print(f"Saved {OUT_DIR}/stage_5_roi.png")

# --- SAVE COMBINED STRIP for reference ---
fig, axes = plt.subplots(1, 6, figsize=(18, 2.5))
titles = [
    'Input\n(Gray)',
    'DoG Filter\n(Bipolar)',
    'Amacrine\n(Temporal)',
    'Ganglion\n(Inhibited)',
    'Active\nMask',
    'ROI\n(Gated)'
]
cmaps  = ['gray', 'hot', 'hot', 'hot', 'gray', None]
images = [
    stage_input,
    norm(stage_dog),
    norm(stage_amacrine),
    norm(stage_ganglion),
    stage_mask,
    cv2.cvtColor(stage_roi, cv2.COLOR_BGR2RGB)
]

for ax, title, data, cmap in zip(
    axes, titles, images, cmaps
):
    if cmap is None:
        ax.imshow(data)
    else:
        ax.imshow(data, cmap=cmap)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.axis('off')

plt.suptitle(
    'RetiGate Pipeline — Intermediate Stages',
    fontsize=11, y=1.02
)
plt.tight_layout()
strip_path = OUT_DIR / "pipeline_strip.png"
plt.savefig(str(strip_path), dpi=200,
            bbox_inches='tight')
plt.close()
print(f"\nSaved combined strip: {strip_path}")
print("Run fig2.py next to build the full figure.")