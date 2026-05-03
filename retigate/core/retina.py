#!/usr/bin/env python3
import cv2
import numpy as np

class RetinaCore:
    """
    LOCKED - April 2026. Do not modify.
    
    Scientific Justification:
    These parameters represent the steady-state calibration for RetiGate's 
    biological fidelity. They maximize the signal-to-noise ratio between 
    temporal flux and static scene geometry.
    
    Validated against:
    - KITTI Tracking: 95.5% fidelity recall (21 sequences)
    - KITTI Tracking: 50.23% mAP@0.5 (Experiment 14)
    - DAVIS: 97%+ sparsity, 99.28% recall (102 sequences)
    """
    def __init__(self, 
                 amacrine_decay=0.1, 
                 global_weight=1.5, 
                 tail_len=15, 
                 shift_factor=0.5, 
                 threshold=0.05,
                 use_vos=True,
                 use_global_inh=True, 
                 use_sac_tail=True):
        
        self.amacrine_decay = amacrine_decay
        self.global_weight = global_weight
        self.tail_len = tail_len
        self.shift_amount = max(1, int(tail_len * shift_factor))
        self.threshold = threshold
        self.use_vos = use_vos
        self.use_global_inh = use_global_inh
        self.use_sac_tail = use_sac_tail
        
        self.amacrine_state = None
        self.prev_gray = None
        
        # VOS (Vestibulo-Ocular Stabilization) components
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kps = None
        self.prev_des = None

        # Bio-kernels (DoG)
        self.m_center_k = self._make_gaussian_kernel(9, 1.5)
        self.m_surround_k = self._make_gaussian_kernel(21, 4.0)

    @classmethod
    def golden_baseline(cls, use_vos=True):
        """Single source of truth for canonical parameters."""
        return cls(amacrine_decay=0.1, global_weight=1.5, use_vos=use_vos)

    def reset_memory(self):
        self.amacrine_state = None
        self.prev_gray = None
        self.prev_kps, self.prev_des = None, None

    def _make_gaussian_kernel(self, size, sigma):
        k = cv2.getGaussianKernel(size, sigma)
        return k @ k.T

    def _vos_stabilize(self, gray):
        if self.prev_gray is None:
            kps, des = self.orb.detectAndCompute(gray, None)
            self.prev_kps, self.prev_des = kps, des
            return gray
        
        kps, des = self.orb.detectAndCompute(gray, None)
        stabilized = gray  # default: no stabilization
        
        if (self.prev_des is not None and des is not None 
                and len(self.prev_des) >= 4 and len(des) >= 4):
            matches = self.matcher.match(self.prev_des, des)
            if len(matches) > 10:
                src_pts = np.float32(
                    [self.prev_kps[m.queryIdx].pt for m in matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kps[m.trainIdx].pt for m in matches]
                ).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(
                    dst_pts, src_pts, cv2.RANSAC, 5.0
                )
                if H is not None:
                    stabilized = cv2.warpPerspective(
                        gray, H, 
                        (gray.shape[1], gray.shape[0]),
                        borderMode=cv2.BORDER_REPLICATE
                    )
        
        # ALWAYS update reference — this was the bug
        self.prev_kps, self.prev_des = kps, des
        return stabilized

    def process_frame(self, frame):
        # --- SHAPE GUARD ---
        # If the frame resolution changes (e.g., sequence jump), reset memory
        if self.prev_gray is not None:
            if frame.shape != self.prev_gray.shape:
                self.reset_memory()
        
        # Stage 0: VOS Stabilization
        work_gray = self._vos_stabilize(frame) if self.use_vos else frame
        img = work_gray.astype(np.float32) / 255.0

        # Stage 1: Spatial DoG filtering
        m_c = cv2.filter2D(img, -1, self.m_center_k, borderType=cv2.BORDER_REFLECT)
        m_s = cv2.filter2D(img, -1, self.m_surround_k, borderType=cv2.BORDER_REFLECT)
        m_bipolar = np.abs(m_c - m_s)

        # Stage 2: Temporal leaky integration
        if self.amacrine_state is None: self.amacrine_state = np.zeros_like(m_bipolar)
        self.amacrine_state = (self.amacrine_decay * m_bipolar + 
                              (1 - self.amacrine_decay) * self.amacrine_state)

        # Stage 3: Global inhibition
        inhibition = (self.amacrine_state + self.global_weight * np.mean(self.amacrine_state)) if self.use_global_inh else self.amacrine_state
        m_ganglion = np.maximum(0.0, m_bipolar - inhibition)

        # Stage 4: SAC directional tail
        if self.use_sac_tail:
            smeared = cv2.blur(self.amacrine_state, (self.tail_len, 1))
            ds_r = np.maximum(0.0, m_ganglion - np.roll(smeared, -self.shift_amount, axis=1))
            ds_l = np.maximum(0.0, m_ganglion - np.roll(smeared, self.shift_amount, axis=1))
        else:
            ds_r, ds_l = m_ganglion.copy(), m_ganglion.copy()

        active_mask = m_ganglion > self.threshold
        self.prev_gray = work_gray

        return {
            'M_Motion': m_ganglion,
            'DS_Right': ds_r,
            'DS_Left': ds_l,
            'sparsity': 1.0 - (np.sum(active_mask) / active_mask.size),
            'active_mask': active_mask
        }
    

    def get_roi_clusters(self, out: dict, pad: int = 40,
                         frame_shape: tuple = None,
                         max_area_frac: float = 0.15,
                         max_clusters: int = 5) -> list:
        """
        Return up to max_clusters tight ROI boxes,
        one per independent motion cluster.

        Unlike get_roi_bbox (single bounding box of ALL motion),
        this returns separate boxes per cluster —
        avoiding the "pedestrian left + car right = 100% frame"
        problem of single-box gating.

        Fallback: if all clusters exceed max_area_frac,
        use the smallest oversized cluster rather than
        returning empty (same safety logic as get_roi_bbox).
        """
        import cv2

        mask   = out['active_mask'].astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (35, 35)
        )
        dilated   = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        H_f, W_f = (frame_shape[:2] if frame_shape is not None
                    else out['active_mask'].shape[:2])
        area_limit = max_area_frac * H_f * W_f

        clusters      = []   # valid clusters
        skipped_large = []   # too big but not deleted

        for c in contours:
            a = cv2.contourArea(c)
            if a <= 200:
                continue   # noise — ignore

            x, y, w, h = cv2.boundingRect(c)

            if frame_shape is not None:
                H, W = frame_shape[:2]
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(W, x + w + pad)
                y2 = min(H, y + h + pad)
            else:
                x1, y1, x2, y2 = x, y, x + w, y + h

            if a > area_limit:
                # Don't delete — track for fallback
                skipped_large.append((a, x1, y1, x2, y2))
            else:
                clusters.append((a, x1, y1, x2, y2))

        # Safety fallback — mirrors get_roi_bbox logic
        # If nothing survives filtering, use smallest large cluster
        # This prevents silent misses of large nearby objects
        if not clusters and skipped_large:
            skipped_large.sort(key=lambda x: x[0])  # smallest first
            clusters = [skipped_large[0]]

        # Sort by area descending, keep top N
        clusters.sort(key=lambda x: x[0], reverse=True)
        clusters = clusters[:max_clusters]

        return [(x1, y1, x2, y2)
                for _, x1, y1, x2, y2 in clusters]

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