#!/usr/bin/env python3
import os
from pathlib import Path

class KITTIDataset:
    """
    Standardized KITTI Tracking Dataset Loader.
    Defaults to the 15GB Tracking dataset as per 2026 Peer Review requirements.
    """
    def __init__(self, root=None):
        # Default path if none provided
        if root is None:
            self.root = Path("data/kitti/data_tracking_image/image_02")
        else:
            self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(f"KITTI Root not found at {self.root}. Check your 'data' folder structure!")

        # Recursive glob finds images in all sequence subfolders (0000, 0001... 0020)
        # We sort them to ensure temporal continuity for the Leaky Integrator
        self.image_paths = sorted(list(self.root.glob("**/*.png")))

        if len(self.image_paths) == 0:
            print(f"ERROR: Found 0 images in {self.root}. Check if subfolders (0000, 0001) exist.")
        else:
            print(f"[DATASET] Initialized KITTI Tracking: {len(self.image_paths)} frames found across all sequences.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.image_paths[idx]