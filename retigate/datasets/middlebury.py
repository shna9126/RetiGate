from pathlib import Path
import cv2
import numpy as np

class MiddleburyDataset:
    """Loader for Middlebury 'other-data' sequences."""
    def __init__(self, root: str = 'data/middlebury', sequence: str = 'Dimetrodon'):
        self.root = Path(root)
        self.image_dir = self.root / 'other-data' / sequence
        # Middlebury typically uses frame10.png and frame11.png
        self.image_paths = sorted(self.image_dir.glob('frame*.png'))

    @classmethod
    def list_sequences(cls, root: str = 'data/middlebury') -> list:
        path = Path(root) / 'other-data'
        return sorted([d.name for d in path.iterdir() if d.is_dir()]) if path.exists() else []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_paths[idx]))
        return self.image_paths[idx].stem, img