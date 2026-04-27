from pathlib import Path
import cv2
import numpy as np

class DAVISDataset:
    """Dynamic loader for DAVIS 2017 sequences."""
    def __init__(self, root: str = 'data/davis/DAVIS', sequence: str = 'car-roundabout'):
        self.root = Path(root)
        # Based on your screenshot path structure
        self.image_dir = self.root / 'JPEGImages' / '480p' / sequence
        self.image_paths = sorted(self.image_dir.glob('*.jpg'))
        
        if not self.image_paths:
            raise FileNotFoundError(f"No frames found for sequence '{sequence}' at {self.image_dir}")

    @classmethod
    def list_sequences(cls, root: str = 'data/davis/DAVIS') -> list:
        """Find all available sequence folders in the 480p directory."""
        path = Path(root) / 'JPEGImages' / '480p'
        if not path.exists():
            return []
        # Return folder names that contain images
        return sorted([d.name for d in path.iterdir() if d.is_dir()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(str(path))
        return path.stem, img