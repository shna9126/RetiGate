#!/usr/bin/env python3
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Project Imports
from retigate import RetinaCore

# --- CONFIGURATION (Matches your new structure) ---
DATA_ROOT = Path("data/kitti/data_tracking_image") 
IMG_DIR = DATA_ROOT / "image_02"
LBL_DIR = DATA_ROOT / "label_02"

def parse_tracking_labels(label_path):
    """
    KITTI Tracking Format:
    0: Frame ID
    1: Track ID
    2: Object Type (Car, Pedestrian, etc.)
    6-9: BBox (left, top, right, bottom)
    """
    try:
        df = pd.read_csv(label_path, sep=' ', header=None)
        # We only need: Frame, Type, BBox coordinates
        df = df[[0, 2, 6, 7, 8, 9]]
        df.columns = ['frame', 'type', 'x1', 'y1', 'x2', 'y2']
        # Filter for the 'Big 3' moving objects in KITTI
        return df[df['type'].isin(['Car', 'Pedestrian', 'Cyclist'])]
    except Exception as e:
        print(f"Error parsing {label_path}: {e}")
        return pd.DataFrame()

def main():
    model = YOLO('yolo11m.pt')
    retina = RetinaCore.golden_baseline()
    retina.threshold = 0.10  # Our "Golden Constant"
    
    overall_results = []
    
    # Audit all sequences from 0000 to 0020
    sequences = sorted([d.name for d in IMG_DIR.iterdir() if d.is_dir()])
    
    print(f">>> STARTING ACTUAL FIDELITY AUDIT: {len(sequences)} Sequences")
    
    for seq in sequences:
        label_file = LBL_DIR / f"{seq}.txt"
        if not label_file.exists(): continue
        
        gt_df = parse_tracking_labels(label_file)
        img_paths = sorted(list((IMG_DIR / seq).glob("*.png")))
        
        # --- THE RIGOROUS FIX ---
        # Re-instantiate the Retina for every sequence. 
        # This clears ALL internal memory buffers (Amacrine, Bipolar, etc.)
        # and allows the Retina to adapt to the new resolution of the current drive.
        retina = RetinaCore.golden_baseline()
        retina.threshold = 0.10
        
        seq_hits = 0
        seq_total_gt = 0
        
        for i, img_path in enumerate(tqdm(img_paths, desc=f"Seq {seq}", leave=False)):
            frame_gt = gt_df[gt_df['frame'] == i]
            if frame_gt.empty: continue
            
            # 1. Biological Filtering
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rout = retina.process_frame(gray)
            roi = retina.get_roi_bbox(rout, frame_shape=img.shape)
            
            # 2. Fidelity Check (Recall)
            # Did our Retina ROI successfully "capture" the ground truth objects?
            for _, gt in frame_gt.iterrows():
                seq_total_gt += 1
                if roi:
                    # Check if Ground Truth box is contained within Retina ROI
                    # We add a tiny 5px padding for edge cases
                    if (gt.x1 >= roi[0]-5 and gt.y1 >= roi[1]-5 and 
                        gt.x2 <= roi[2]+5 and gt.y2 <= roi[3]+5):
                        seq_hits += 1
        
        if seq_total_gt > 0:
            recall = (seq_hits / seq_total_gt) * 100
            print(f"Sequence {seq}: Recall @ 0.10 tau = {recall:.2f}%")
            overall_results.append(recall)

    print("\n" + "="*50)
    print("FINAL SCIENTIFIC PROOF: KITTI TRACKING FIDELITY")
    print("="*50)
    print(f"Mean Recall across all sequences: {np.mean(overall_results):.2f}%")
    print(f"Standard Deviation: {np.std(overall_results):.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()