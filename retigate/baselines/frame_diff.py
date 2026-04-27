import cv2
import numpy as np

class FrameDiffBaseline:
    """Simple temporal subtraction baseline."""
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.prev_gray = None

    def process_frame(self, gray: np.ndarray) -> dict:
        if self.prev_gray is None:
            self.prev_gray = gray
            return {'active_mask': np.zeros_like(gray, dtype=bool), 'sparsity': 1.0}
        
        diff = cv2.absdiff(gray, self.prev_gray).astype(np.float32) / 255.0
        active_mask = diff > self.threshold
        self.prev_gray = gray
        return {'active_mask': active_mask, 'sparsity': 1.0 - (np.sum(active_mask) / active_mask.size)}