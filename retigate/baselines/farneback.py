import cv2
import numpy as np

class FarnebackBaseline:
    """OpenCV Gunnar Farneback dense optical flow baseline."""
    def __init__(self):
        self.prev_gray = None
        # Standard parameters for robust dense flow
        self.params = dict(
            pyr_scale=0.5, 
            levels=3, 
            winsize=15, 
            iterations=3, 
            poly_n=5, 
            poly_sigma=1.2, 
            flags=0
        )

    def process_frame(self, gray: np.ndarray) -> dict:
        """
        Calculates dense flow and gates pixels based on motion magnitude.
        Returns active_mask and sparsity.
        """
        if self.prev_gray is None:
            self.prev_gray = gray
            return {
                'active_mask': np.zeros_like(gray, dtype=bool), 
                'sparsity': 1.0
            }
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, **self.params
        )
        
        # Compute magnitude of flow vectors
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Gate pixels moving more than 1.5 pixels per frame
        active_mask = mag > 1.5 
        self.prev_gray = gray
        
        sparsity = 1.0 - (np.sum(active_mask) / active_mask.size)
        
        return {
            'active_mask': active_mask, 
            'sparsity': float(sparsity)
        }