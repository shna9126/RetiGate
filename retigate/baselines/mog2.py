import cv2
import numpy as np

class MOG2Baseline:
    """OpenCV Mixture of Gaussians (v2) baseline."""
    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    def process_frame(self, gray: np.ndarray) -> dict:
        fg_mask = self.backSub.apply(gray)
        active_mask = fg_mask > 0
        return {'active_mask': active_mask, 'sparsity': 1.0 - (np.sum(active_mask) / active_mask.size)}