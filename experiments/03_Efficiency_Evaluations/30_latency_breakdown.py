# experiments/03_Efficiency_Evaluations/30_latency_breakdown.py
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from retigate import RetinaCore

# Paths matching your environment
DATA_ROOT = Path("data/kitti/data_tracking_image/image_02")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)
N_FRAMES = 500
WARMUP = 50

def main():
    # Using sequence 0007 as a representative urban sequence
    img_paths = sorted(list((DATA_ROOT / "0007").glob("*.png")))[:N_FRAMES + WARMUP]

    if not img_paths:
        print(f"Error: No images found in {DATA_ROOT / '0007'}")
        return

    stages = {
        'VOS (ORB+RANSAC)': [],
        'Spatial Filter (DoG)': [],
        'Temporal Integration': [],
        'Global Inhibition': [],
        'SAC Directional Tail': [],
        'ROI Extraction': [],
    }
    
    retina = RetinaCore.golden_baseline()

    print(f"Benchmarking RetiGate pipeline on {len(img_paths)} frames...")

    for i, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Stage 0: VOS Stabilization
        t_start = time.perf_counter()
        work_gray = retina._vos_stabilize(gray) if retina.use_vos else gray
        t_vos = (time.perf_counter() - t_start) * 1000
        
        # Prepare for processing
        img_float = work_gray.astype(np.float32) / 255.0

        # Stage 1: Spatial DoG filtering
        t_start = time.perf_counter()
        m_c = cv2.filter2D(img_float, -1, retina.m_center_k, borderType=cv2.BORDER_REFLECT)
        m_s = cv2.filter2D(img_float, -1, retina.m_surround_k, borderType=cv2.BORDER_REFLECT)
        m_bipolar = np.abs(m_c - m_s)
        t_dog = (time.perf_counter() - t_start) * 1000

        # Stage 2: Temporal leaky integration
        t_start = time.perf_counter()
        if retina.amacrine_state is None: 
            retina.amacrine_state = np.zeros_like(m_bipolar)
        retina.amacrine_state = (retina.amacrine_decay * m_bipolar + 
                                (1 - retina.amacrine_decay) * retina.amacrine_state)
        t_iir = (time.perf_counter() - t_start) * 1000

        # Stage 3: Global inhibition
        t_start = time.perf_counter()
        if retina.use_global_inh:
            inhibition = (retina.amacrine_state + retina.global_weight * np.mean(retina.amacrine_state))
        else:
            inhibition = retina.amacrine_state
        m_ganglion = np.maximum(0.0, m_bipolar - inhibition)
        t_ginh = (time.perf_counter() - t_start) * 1000

        # Stage 4: SAC directional tail
        t_start = time.perf_counter()
        if retina.use_sac_tail:
            smeared = cv2.blur(retina.amacrine_state, (retina.tail_len, 1))
            _ = np.maximum(0.0, m_ganglion - np.roll(smeared, -retina.shift_amount, axis=1))
        t_sac = (time.perf_counter() - t_start) * 1000

        # Stage 5: ROI Extraction
        # Create a mock output dict to feed into get_roi_bbox
        active_mask = m_ganglion > retina.threshold
        mock_out = {'active_mask': active_mask}
        t_start = time.perf_counter()
        _ = retina.get_roi_bbox(mock_out, frame_shape=img.shape)
        t_roi = (time.perf_counter() - t_start) * 1000
        
        # Cleanup for next frame
        retina.prev_gray = work_gray

        # Record metrics after warmup
        if i >= WARMUP:
            stages['VOS (ORB+RANSAC)'].append(t_vos)
            stages['Spatial Filter (DoG)'].append(t_dog)
            stages['Temporal Integration'].append(t_iir)
            stages['Global Inhibition'].append(t_ginh)
            stages['SAC Directional Tail'].append(t_sac)
            stages['ROI Extraction'].append(t_roi)

    # Compute means
    means = [np.mean(v) for v in stages.values()]
    labels = [k.replace(' ', '\n') for k in stages.keys()]
    total_overhead = sum(means)

    # Print Summary
    print("\n" + "="*60)
    print("RETIGATE SENSING OVERHEAD BREAKDOWN (Mac-MPS)")
    print("="*60)
    for label, mean in zip(stages.keys(), means):
        pct = (mean / total_overhead) * 100
        print(f"  {label:<25}: {mean:6.3f} ms ({pct:4.1f}%)")
    print("-" * 60)
    print(f"  Total Sensing Overhead   : {total_overhead:6.3f} ms")
    print(f"  YOLO11m Dense Reference  : 32.260 ms")
    print(f"  Net Estimated Saving     : {32.26 - total_overhead:6.3f} ms")
    print("="*60)

    # Visualization
    colors = ['#E74C3C','#2980B9','#27AE60','#F39C12','#8E44AD','#16A085']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

    # Pie Chart
    axes[0].pie(means, labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=90, pctdistance=0.85, textprops={'fontsize': 8})
    axes[0].set_title('RetiGate Compute Distribution', fontweight='bold', fontsize=12)

    # Bar Chart
    categories = ['RetiGate\nOverhead', 'YOLO Dense\n(T4 Ref)', 'Sparse\nTotal (Est)']
    # Sparse Total = Overhead + (~20ms sparse inference estimate)
    vals = [total_overhead, 32.26, total_overhead + 20.13]
    bars = axes[1].bar(categories, vals, color=['#27AE60', '#E74C3C', '#2980B9'], width=0.6)
    
    axes[1].set_ylabel('Latency (ms)', fontsize=11)
    axes[1].set_title('Compute Budget Comparison', fontweight='bold', fontsize=12)
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.2f}ms', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT / "fig9_latency_breakdown.png", dpi=300)
    print(f"\nFinal breakdown figure saved to {OUT / 'fig9_latency_breakdown.png'}")
    plt.show()

if __name__ == "__main__":
    main()