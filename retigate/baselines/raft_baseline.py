# retigate/baselines/raft_baseline.py
# No self-import. No test block that imports itself.

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.models.optical_flow import (
    raft_small, Raft_Small_Weights
)
from torchvision.transforms.functional import to_tensor


class RAFTBaseline:
    """
    RAFT optical flow baseline.
    Teed & Deng, ECCV 2020.
    Drop-in replacement for FarnebackBaseline.
    """

    def __init__(self, device='cpu', threshold=1.5):
        self.device    = device
        self.threshold = threshold
        weights        = Raft_Small_Weights.DEFAULT
        self.model     = raft_small(weights=weights).to(device)
        self.model.eval()
        self.prev_tensor = None

    def _to_tensor(self, gray: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return to_tensor(rgb).unsqueeze(0).to(self.device)

    @staticmethod
    def _pad(tensor: torch.Tensor,
             multiple: int = 8):
        _, _, h, w = tensor.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        padded = F.pad(tensor, (0, pad_w, 0, pad_h),
                       mode='replicate')
        return padded, pad_h, pad_w

    def reset(self):
        self.prev_tensor = None

    @torch.no_grad()
    def process_frame(self, gray: np.ndarray) -> dict:
        h, w         = gray.shape
        curr_tensor  = self._to_tensor(gray)

        if self.prev_tensor is None:
            self.prev_tensor = curr_tensor
            return {
                'active_mask': np.zeros_like(gray, dtype=bool),
                'sparsity':    1.0,
            }

        if self.prev_tensor.shape != curr_tensor.shape:
            self.prev_tensor = curr_tensor
            return {
                'active_mask': np.zeros_like(gray, dtype=bool),
                'sparsity':    1.0,
            }

        prev_pad, _,  _  = self._pad(self.prev_tensor)
        curr_pad, ph, pw = self._pad(curr_tensor)

        flow_preds  = self.model(prev_pad, curr_pad)
        flow        = flow_preds[-1][0, :, :h, :w].cpu().numpy()

        mag         = np.sqrt(flow[0]**2 + flow[1]**2)
        active_mask = mag > self.threshold
        sparsity    = float(
            1.0 - np.sum(active_mask) / active_mask.size
        )

        self.prev_tensor = curr_tensor

        return {
            'active_mask': active_mask,
            'sparsity':    sparsity,
        }