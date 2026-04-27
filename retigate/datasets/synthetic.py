import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from retigate.core.retina import RetinaCore

def run_synthetic_rigor(world_image_path, speed=2, total_frames=100):
    img = cv2.imread(world_image_path, cv2.IMREAD_GRAYSCALE)
    # We resize larger to give more "runway" for the pan
    img = cv2.resize(img, (2000, 480)) 
    
    retina = RetinaCore.golden_baseline()
    stats = []
    
    retina.reset_memory() 
    h, w = 480, 640

    for i in range(1, total_frames):
        x_start = i * speed
        x_prev = (i-1) * speed
        
        # Boundary Check: Stop if we hit the edge of the world
        if x_start + w > img.shape[1]:
            break
            
        curr = img[0:h, x_start:x_start+w]
        prev = img[0:h, x_prev:x_prev+w]
        
        # Priming/Habituation
        retina.process_frame(prev) 
        rout = retina.process_frame(curr)
        stats.append(rout['sparsity'] * 100)
        
    return np.mean(stats)

def populate_synthetic(image_path, output_dir='data/synthetic/pan_benchmark'):
    """Generate synthetic pan sequence from a high-res world image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    # Use high-quality resize to give more "runway" for the pan
    img = cv2.resize(img, (2500, 480), interpolation=cv2.INTER_LANCZOS4) 
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Generating Synthetic sequence in {output_dir}...")
    for i in tqdm(range(0, 100)):
        x = i * 5  # Standardized 5px speed
        crop = img[0:480, x:x+640]
        cv2.imwrite(f"{output_dir}/frame_{i:03d}.png", crop)
    print(f"Synthetic benchmark complete: {output_dir}")

def main():
    # Generate synthetic pan benchmark from KITTI world image
    world_img = "data/synthetic/world_tokyo.png"
    if os.path.exists(world_img):
        populate_synthetic(world_img, output_dir='data/synthetic/pan_benchmark')
    else:
        print(f"Warning: World image not found at {world_img}")
    
    world_img = "data/kitti/data_scene_flow/training/image_2/000000_10.png"
    speeds = [1, 2, 5, 10]
    results = []

    print(">>> GENERATING SYNTHETIC STRESS-TEST RESULTS")
    for s in speeds:
        avg_sparse = run_synthetic_rigor(world_img, speed=s)
        print(f"  Speed: {s:>2} px/frame | Sparsity: {avg_sparse:.2f}%")
        results.append({'Speed_px': s, 'Ideal_Sparsity': avg_sparse})

    df = pd.DataFrame(results)
    df.to_csv('results/table_synthetic_limit.csv', index=False)

if __name__ == "__main__":
    main()