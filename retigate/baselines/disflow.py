import cv2
import numpy as np

class DISFlowBaseline:
    """OpenCV Dense Inverse Search (DIS) Optical Flow baseline."""
    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.prev_gray = None

    def process_frame(self, gray: np.ndarray) -> dict:
        if self.prev_gray is None:
            self.prev_gray = gray
            return {'active_mask': np.zeros_like(gray, dtype=bool), 'sparsity': 1.0}
        
        flow = self.dis.calc(self.prev_gray, gray, None)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        active_mask = mag > 1.5 # Pixels moving more than 1.5px/frame
        self.prev_gray = gray
        return {'active_mask': active_mask, 'sparsity': 1.0 - (np.sum(active_mask) / active_mask.size)}