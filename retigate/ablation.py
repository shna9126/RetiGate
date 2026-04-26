import numpy as np
from .core import RetinaCore


class RetinaAblation(RetinaCore):
    """
    Ablation subclass for systematic component removal studies.

    Variants:
        'baseline'    — standard golden-baseline behaviour (no ablation)
        'no_dog'      — removes DoG spatial filtering; uses only centre kernel
        'no_temporal' — removes leaky integration; amacrine_state = m_bipolar each frame
        'no_global'   — removes global inhibition; uses only local amacrine suppression
        'no_sac'      — removes SAC directional tail; DS_Right == DS_Left
    """

    VARIANTS = ('baseline', 'no_dog', 'no_temporal', 'no_global', 'no_sac')

    def __init__(self, variant: str = 'baseline'):
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got '{variant}'")
        super().__init__(
            amacrine_decay=0.1,
            global_weight=1.5,
            tail_len=15,
            shift_factor=0.5,
            threshold=0.05,
            use_global_inh=True,
            use_sac_tail=True,
        )
        self.variant = variant

    def process_frame(self, frame: np.ndarray) -> dict:
        import cv2

        img = frame.astype(np.float32) / 255.0

        # --- Stage 1: Spatial filtering ---
        if self.variant == 'no_dog':
            # Only centre kernel, no surround subtraction
            m_bipolar = cv2.filter2D(img, -1, self.m_center_k,
                                     borderType=cv2.BORDER_REFLECT)
        else:
            m_c = cv2.filter2D(img, -1, self.m_center_k,
                               borderType=cv2.BORDER_REFLECT)
            m_s = cv2.filter2D(img, -1, self.m_surround_k,
                               borderType=cv2.BORDER_REFLECT)
            m_bipolar = np.abs(m_c - m_s)

        # --- Stage 2: Temporal integration ---
        if self.variant == 'no_temporal':
            local_memory = m_bipolar  # no integration; instant response
        else:
            if self.amacrine_state is None:
                self.amacrine_state = np.zeros_like(m_bipolar)
            self.amacrine_state = (
                self.amacrine_decay * m_bipolar +
                (1 - self.amacrine_decay) * self.amacrine_state
            )
            local_memory = self.amacrine_state

        # --- Stage 3: Inhibition ---
        if self.variant == 'no_global':
            inhibition = local_memory
        else:
            inhibition = local_memory + self.global_weight * np.mean(local_memory)

        m_ganglion = np.maximum(0.0, m_bipolar - inhibition)

        # --- Stage 4: SAC tail ---
        if self.variant == 'no_sac':
            ds_r = m_ganglion.copy()
            ds_l = m_ganglion.copy()
        else:
            smeared = cv2.blur(local_memory, (self.tail_len, 1))
            ds_r = np.maximum(0.0, m_ganglion -
                              np.roll(smeared, -self.shift_amount, axis=1))
            ds_l = np.maximum(0.0, m_ganglion -
                              np.roll(smeared, self.shift_amount, axis=1))
            ds_r = cv2.GaussianBlur(ds_r, (5, 5), 1.5)
            ds_l = cv2.GaussianBlur(ds_l, (5, 5), 1.5)

        active_mask = m_ganglion > self.threshold
        sparsity = 1.0 - (np.sum(active_mask) / active_mask.size)

        return {
            'M_Motion': m_ganglion,
            'DS_Right': ds_r,
            'DS_Left': ds_l,
            'sparsity': float(sparsity),
            'active_mask': active_mask,
        }
