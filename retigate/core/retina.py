import numpy as np


class RetinaCore:
    """
    The unified RetiGate architecture.

    All experiments must instantiate via RetinaCore.golden_baseline()
    or RetinaCore.from_config(config_dict). Never call the constructor
    directly in experiment scripts.
    """

    def __init__(self,
                 amacrine_decay: float = 0.1,
                 global_weight: float = 1.5,
                 tail_len: int = 15,
                 shift_factor: float = 0.5,
                 threshold: float = 0.05,
                 use_global_inh: bool = True,
                 use_sac_tail: bool = True):

        self.amacrine_decay = amacrine_decay
        self.global_weight = global_weight
        self.tail_len = tail_len
        self.shift_amount = max(1, int(tail_len * shift_factor))
        self.threshold = threshold
        self.use_global_inh = use_global_inh
        self.use_sac_tail = use_sac_tail
        self.amacrine_state = None

        # Standard bio-kernels (DoG)
        self.m_center_k = self._make_gaussian_kernel(9, 1.5)
        self.m_surround_k = self._make_gaussian_kernel(21, 4.0)

    @classmethod
    def golden_baseline(cls) -> 'RetinaCore':
        """
        Single source of truth for canonical parameters.
        Every experiment in this repo uses this method.
        """
        return cls(
            amacrine_decay=0.1,
            global_weight=1.5,
            tail_len=15,
            shift_factor=0.5,
            threshold=0.05,
            use_global_inh=True,
            use_sac_tail=True,
        )

    @classmethod
    def from_config(cls, config: dict) -> 'RetinaCore':
        """For parameter sweeps. Pass dict of overrides."""
        return cls(**config)

    def _make_gaussian_kernel(self, size: int, sigma: float):
        import cv2
        k = cv2.getGaussianKernel(size, sigma)
        return k @ k.T

    def reset_memory(self):
        self.amacrine_state = None

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process one grayscale frame through the full retinal pipeline.

        Args:
            frame: HxW uint8 grayscale image

        Returns:
            dict with keys:
                'M_Motion':  HxW float32 ganglion saliency map
                'DS_Right':  HxW float32 rightward directional signal
                'DS_Left':   HxW float32 leftward directional signal
                'sparsity':  float, fraction of pixels gated out
                'active_mask': HxW bool, True where M_Motion > threshold
        """
        import cv2

        # Stage 1: Spatial DoG filtering (Bipolar cells)
        img = frame.astype(np.float32) / 255.0
        m_c = cv2.filter2D(img, -1, self.m_center_k,
                           borderType=cv2.BORDER_REFLECT)
        m_s = cv2.filter2D(img, -1, self.m_surround_k,
                           borderType=cv2.BORDER_REFLECT)
        m_bipolar = np.abs(m_c - m_s)

        # Stage 2: Temporal leaky integration (Amacrine cells)
        if self.amacrine_state is None:
            self.amacrine_state = np.zeros_like(m_bipolar)
        self.amacrine_state = (
            self.amacrine_decay * m_bipolar +
            (1 - self.amacrine_decay) * self.amacrine_state
        )

        # Stage 3: Global inhibition (Ganglion cells)
        if self.use_global_inh:
            inhibition = (self.amacrine_state +
                          self.global_weight * np.mean(self.amacrine_state))
        else:
            inhibition = self.amacrine_state

        m_ganglion = np.maximum(0.0, m_bipolar - inhibition)

        # Stage 4: SAC directional tail
        if self.use_sac_tail:
            smeared = cv2.blur(self.amacrine_state,
                               (self.tail_len, 1))
            ds_r = np.maximum(0.0, m_ganglion -
                              np.roll(smeared, -self.shift_amount, axis=1))
            ds_l = np.maximum(0.0, m_ganglion -
                              np.roll(smeared, self.shift_amount, axis=1))
            ds_r = cv2.GaussianBlur(ds_r, (5, 5), 1.5)
            ds_l = cv2.GaussianBlur(ds_l, (5, 5), 1.5)
        else:
            ds_r = m_ganglion.copy()
            ds_l = m_ganglion.copy()

        # Sparsity metrics
        active_mask = m_ganglion > self.threshold
        sparsity = 1.0 - (np.sum(active_mask) / active_mask.size)

        return {
            'M_Motion': m_ganglion,
            'DS_Right': ds_r,
            'DS_Left': ds_l,
            'sparsity': float(sparsity),
            'active_mask': active_mask,
        }

    def get_roi_clusters(self, out: dict, pad: int = 40,
                         frame_shape: tuple = None,
                         max_area_frac: float = 0.10) -> list:
        """
        Return individual kinetic-zone bounding boxes, one per motion cluster.
        Uses the same background-blob filter as get_roi_bbox (> max_area_frac),
        but returns each surviving cluster separately instead of their union.
        Sorted largest-first so callers can cheaply cap with [:N].

        Returns list of (x1, y1, x2, y2) tuples (empty list if none found).
        """
        import cv2
        mask = out['active_mask'].astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        H_f, W_f = (frame_shape[:2] if frame_shape is not None
                    else out['active_mask'].shape[:2])
        area_limit = max_area_frac * H_f * W_f

        clusters = []
        for c in contours:
            a = cv2.contourArea(c)
            if a <= 200 or a > area_limit:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if frame_shape is not None:
                H, W = frame_shape[:2]
                x1 = max(0, x - pad);    y1 = max(0, y - pad)
                x2 = min(W, x + w + pad); y2 = min(H, y + h + pad)
            else:
                x1, y1, x2, y2 = x, y, x + w, y + h
            clusters.append((a, x1, y1, x2, y2))

        clusters.sort(reverse=True)
        return [(x1, y1, x2, y2) for _, x1, y1, x2, y2 in clusters]

    def get_roi_bbox(self, out: dict, pad: int = 40,
                     frame_shape: tuple = None,
                     max_area_frac: float = 0.10,
                     margin: float = 0.10) -> tuple:
        """
        From process_frame output, return (x1,y1,x2,y2) bounding box
        of the active kinetic zone, with padding and optional relative margin.
        Returns None if no active pixels.

        max_area_frac: skip contours whose area exceeds this fraction of the
        frame.  Prevents the large "background motion mesh" blob — formed when
        scattered active pixels merge after dilation across unrelated scenes —
        from dominating the ROI.  Vehicle-scale clusters (< 10 % of frame)
        survive and produce a tight, task-relevant crop.

        margin: adds a relative buffer around the detected motion. For example,
        margin=0.10 expands the ROI by 10% in each direction before clamping to
        the frame bounds.
        """
        import cv2
        mask = out['active_mask'].astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        H_f, W_f = (frame_shape[:2] if frame_shape is not None
                    else out['active_mask'].shape[:2])
        area_limit = max_area_frac * H_f * W_f

        rects = []
        for c in contours:
            a = cv2.contourArea(c)
            if a > 200 and a <= area_limit:
                rects.append(cv2.boundingRect(c))

        # Fallback: if every contour is too large, use the smallest one
        if not rects:
            all_valid = sorted(
                [(cv2.contourArea(c), cv2.boundingRect(c)) for c in contours
                 if cv2.contourArea(c) > 200]
            )
            if not all_valid:
                return None
            rects = [all_valid[0][1]]

        x1 = min(b[0] for b in rects)
        y1 = min(b[1] for b in rects)
        x2 = max(b[0]+b[2] for b in rects)
        y2 = max(b[1]+b[3] for b in rects)
        if frame_shape is not None:
            H, W = frame_shape[:2]
            w = x2 - x1
            h = y2 - y1
            dx = int(w * margin)
            dy = int(h * margin)
            x1 = max(0, x1 - pad - dx)
            y1 = max(0, y1 - pad - dy)
            x2 = min(W, x2 + pad + dx)
            y2 = min(H, y2 + pad + dy)
        return (x1, y1, x2, y2)
