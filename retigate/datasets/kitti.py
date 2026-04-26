from pathlib import Path
from typing import Iterator, Tuple, List, Optional
import numpy as np


CLASS_MAP = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Cyclist': 4,
}


class KITTIDataset:
    """
    Loads KITTI object detection / scene flow data from data/kitti/.

    Expects:
        data/kitti/training/image_2/   — input images
        data/kitti/training/label_2/   — object detection labels
        data/kitti/training/flow_occ/  — optical flow GT (scene flow pack)
    """

    def __init__(self, root: str = 'data/kitti'):
        self.root = Path(root)
        self.image_dir = self.root / 'training' / 'image_2'
        self.label_dir = self.root / 'training' / 'label_2'
        self.flow_dir = self.root / 'training' / 'flow_occ'

        if self.image_dir.exists():
            self.image_paths = sorted(self.image_dir.glob('*.png'))
        else:
            self.image_paths = []

    def __len__(self) -> int:
        return len(self.image_paths)

    def __iter__(self) -> Iterator[Tuple[Path, np.ndarray, np.ndarray]]:
        for img_path in self.image_paths:
            gt_boxes, gt_classes = self._load_label(img_path.stem)
            yield img_path, gt_boxes, gt_classes

    def _load_label(self, stem: str) -> Tuple[np.ndarray, np.ndarray]:
        label_path = self.label_dir / f'{stem}.txt'
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
                # KITTI format: left top right bottom (columns 4-7, 0-indexed)
                x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                boxes.append([x1, y1, x2, y2])
                classes.append(CLASS_MAP[cls_name])

        if boxes:
            return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int64)
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

    def load_flow_gt(self, stem: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load KITTI optical flow GT for image `stem`.

        Returns:
            (flow_u, flow_v, valid_mask) as float32/bool arrays,
            or None if the flow file does not exist.

        KITTI encoding:
            flow_u = (img[:,:,2].astype(float) - 2**15) / 64.0
            flow_v = (img[:,:,1].astype(float) - 2**15) / 64.0
            valid  = img[:,:,0] > 0
        """
        import cv2
        flow_path = self.flow_dir / f'{stem}_10.png'
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
