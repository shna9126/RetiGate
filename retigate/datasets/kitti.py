from pathlib import Path
from typing import Iterator, Tuple, List, Optional
import numpy as np


CLASS_MAP = {
    'Car': 2,        # COCO: car
    'Pedestrian': 0, # COCO: person
    # Van, Truck, Cyclist excluded: ambiguous COCO mapping destabilises mAP
}


class KITTIDataset:
    """
    Loads KITTI object detection / scene flow data from data/kitti/.

    Actual on-disk layout after downloading both KITTI packages:
        data/kitti/data_scene_flow/training/image_2/   — _10.png reference frames
        data/kitti/data_scene_flow/training/flow_occ/  — optical flow GT
        data/kitti/training/label_2/                   — object detection labels

    Only the _10.png frames (reference frames) are indexed; _11.png are
    the second frames used for flow computation.
    """

    def __init__(self, root: str = 'data/kitti'):
        self.root = Path(root)

        # Scene-flow pack has images and flow
        sf = self.root / 'data_scene_flow' / 'training'
        self.image_dir = sf / 'image_2'
        self.flow_dir = sf / 'flow_occ'

        # Object-detection pack has labels
        self.label_dir = self.root / 'training' / 'label_2'

        if self.image_dir.exists():
            # Only index the reference (first) frames: *_10.png
            self.image_paths = sorted(self.image_dir.glob('*_10.png'))
        else:
            self.image_paths = []

    def __len__(self) -> int:
        return len(self.image_paths)

    def __iter__(self) -> Iterator[Tuple[Path, np.ndarray, np.ndarray]]:
        for img_path in self.image_paths:
            gt_boxes, gt_classes = self._load_label(img_path.stem)
            yield img_path, gt_boxes, gt_classes

    def _label_stem(self, img_stem: str) -> str:
        """Convert image stem '000042_10' → label stem '000042'."""
        return img_stem.replace('_10', '')

    def _load_label(self, img_stem: str,
                    max_truncation: float = 0.3,
                    max_occlusion: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load 2-D boxes and COCO-aligned class IDs for one frame.

        max_truncation: drop objects with truncation flag > this value.
        max_occlusion:  drop objects with occlusion flag > this value.
            KITTI levels: 0=fully visible, 1=partly, 2=largely, 3=unknown.
            Defaults keep Easy + Moderate only (occlusion 0-1, truncation ≤ 0.3).
            Hard/impossible targets (largely occluded, heavily truncated) are
            excluded because they destabilise mAP for a zero-shot detector.
        """
        label_path = self.label_dir / f'{self._label_stem(img_stem)}.txt'
        boxes: List[List[float]] = []
        classes: List[int] = []

        if not label_path.exists():
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_name = parts[0]
                if cls_name not in CLASS_MAP:
                    continue
                # KITTI cols 1-2: truncation (float), occlusion (int)
                if float(parts[1]) > max_truncation or int(parts[2]) > max_occlusion:
                    continue
                # KITTI cols 4-7: left top right bottom (2-D bbox)
                x1, y1, x2, y2 = (float(parts[4]), float(parts[5]),
                                   float(parts[6]), float(parts[7]))
                boxes.append([x1, y1, x2, y2])
                classes.append(CLASS_MAP[cls_name])

        if boxes:
            return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int64)
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

    def load_flow_gt(self, img_stem: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load KITTI optical flow GT for image stem (e.g. '000042_10').

        Returns:
            (flow_u, flow_v, valid_mask) as float32/bool arrays,
            or None if the flow file does not exist.

        KITTI encoding:
            flow_u = (img[:,:,2].astype(float) - 2**15) / 64.0
            flow_v = (img[:,:,1].astype(float) - 2**15) / 64.0
            valid  = img[:,:,0] > 0
        """
        import cv2
        # Flow files share the same _10 suffix as reference frames
        flow_path = self.flow_dir / f'{img_stem}.png'
        if not flow_path.exists():
            return None

        img = cv2.imread(str(flow_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        flow_u = (img[:, :, 2].astype(float) - 2**15) / 64.0
        flow_v = (img[:, :, 1].astype(float) - 2**15) / 64.0
        valid_mask = img[:, :, 0] > 0

        return (flow_u.astype(np.float32),
                flow_v.astype(np.float32),
                valid_mask)
