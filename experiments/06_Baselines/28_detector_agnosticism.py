"""
experiments/05_Baselines/28_detector_agnosticism.py

Evaluates RetiGate's plug-and-play compatibility across
detector architectures (YOLO11m and RT-DETR-L).
Runs on active KITTI sequences where RetiGate ROI
activates reliably (low ego-motion scenes).

Usage:
    python experiments/05_Baselines/28_detector_agnosticism.py
    python experiments/05_Baselines/28_detector_agnosticism.py --sequence 0006
    python experiments/05_Baselines/28_detector_agnosticism.py --sequence active
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from ultralytics import YOLO, RTDETR

sys.path.append(str(Path(__file__).parents[2]))
from retigate import RetinaCore

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# Sequences where RetiGate ROI reliably activates.
# Selected from per_sequence_area.py output:
#   0006: median ROI 52.4%  — slow urban
#   0012: median ROI 20.4%  — best ROI, slow scene
#   0015: median ROI 44.5%  — slow suburban
#   0020: median ROI 81.6%  — mixed suburban
ACTIVE_SEQUENCES = ["0006", "0012", "0015", "0020"]

DETECTORS = {
    "YOLO11m":   "yolo11m.pt",
    "RT-DETR-L": "rtdetr-l.pt",
}

CONF_THRESHOLD = 0.40
GT_CLASSES     = ["Car", "Pedestrian", "Cyclist"]

# ─────────────────────────────────────────────────────────────
# Device Selection
# ─────────────────────────────────────────────────────────────

def get_device(model_path: str) -> str:
    """
    Select the best available device.
    RT-DETR falls back to CPU on MPS due to known
    operator support gaps in PyTorch MPS backend.
    """
    if torch.cuda.is_available():
        return "cuda"

    is_rtdetr = "rtdetr" in model_path.lower()

    if torch.backends.mps.is_available():
        if is_rtdetr:
            logger.warning(
                "RT-DETR: MPS has incomplete op support "
                "— falling back to CPU. "
                "Expect ~4-8× slower inference."
            )
            return "cpu"
        return "mps"

    return "cpu"


# ─────────────────────────────────────────────────────────────
# Label Parser
# ─────────────────────────────────────────────────────────────

def parse_labels(label_path: Path) -> pd.DataFrame:
    """
    Parse KITTI tracking label file.
    Returns DataFrame with columns:
        frame, type, x1, y1, x2, y2
    filtered to GT_CLASSES only.
    """
    try:
        df = pd.read_csv(
            label_path, sep=" ", header=None
        )
        df = df[[0, 2, 6, 7, 8, 9]]
        df.columns = [
            "frame", "type", "x1", "y1", "x2", "y2"
        ]
        df = df[df["type"].isin(GT_CLASSES)].copy()
        df[["x1","y1","x2","y2"]] = df[
            ["x1","y1","x2","y2"]
        ].astype(float)
        return df
    except Exception as e:
        logger.error(f"Label parse failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def run_inference(
    model,
    frame: np.ndarray,
    use_retigate: bool,
    retina: RetinaCore,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run dense or sparse inference on a single frame.

    Returns:
        boxes  : (N, 4) float32 array in full-frame coords
        scores : (N,)   float32 confidence scores
        roi_area_pct: ROI area as % of frame (100 = full)
    """
    H, W    = frame.shape[:2]
    offset  = (0, 0)
    crop    = frame
    roi_pct = 100.0

    if use_retigate:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rout = retina.process_frame(gray)
        roi  = retina.get_roi_bbox(
            rout, frame_shape=frame.shape
        )
        if roi is not None:
            x1, y1, x2, y2 = roi
            crop    = frame[y1:y2, x1:x2]
            offset  = (x1, y1)
            roi_pct = (x2-x1)*(y2-y1) / (W*H) * 100.0

    results   = model(
        crop, conf=CONF_THRESHOLD,
        verbose=False, device=device
    )
    boxes_obj = results[0].boxes

    if boxes_obj is None or len(boxes_obj) == 0:
        return np.zeros((0,4)), np.zeros(0), roi_pct

    bx = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
    cf = boxes_obj.conf.cpu().numpy().astype(np.float32)

    # Remap coordinates to full frame
    if offset != (0, 0):
        bx[:, 0] += offset[0]
        bx[:, 2] += offset[0]
        bx[:, 1] += offset[1]
        bx[:, 3] += offset[1]

    return bx, cf, roi_pct


# ─────────────────────────────────────────────────────────────
# Single Sequence Evaluator
# ─────────────────────────────────────────────────────────────

def evaluate_sequence(
    seq_id:       str,
    model_name:   str,
    model_path:   str,
    use_retigate: bool,
    data_root:    Path,
) -> dict:
    """
    Evaluate one detector on one sequence,
    dense or sparse.

    Returns dict with mAP@50, mAP@75, mean ROI area,
    and per-frame timing.
    """
    img_dir    = data_root / "image_02" / seq_id
    label_path = data_root / "label_02" / f"{seq_id}.txt"

    if not img_dir.exists():
        raise FileNotFoundError(
            f"Image dir not found: {img_dir}"
        )
    if not label_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {label_path}"
        )

    device = get_device(model_path)
    logger.info(
        f"  [{model_name}] "
        f"{'Sparse' if use_retigate else 'Dense '} "
        f"seq={seq_id}  device={device}"
    )

    # Load model
    ModelClass = (
        RTDETR if "rtdetr" in model_path.lower()
        else YOLO
    )
    model = ModelClass(model_path)

    gt_df     = parse_labels(label_path)
    img_paths = sorted(list(img_dir.glob("*.png")))
    retina    = RetinaCore.golden_baseline()

    metric    = MeanAveragePrecision(
        iou_type="bbox",
        iou_thresholds=[0.5, 0.75]
    )

    roi_areas  = []
    frame_times = []

    for i, img_path in enumerate(
        tqdm(img_paths,
             desc=(f"    {model_name} "
                   f"{'sparse' if use_retigate else 'dense '}"),
             leave=False)
    ):
        frame_gt = gt_df[gt_df["frame"] == i]
        if frame_gt.empty:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Could not read {img_path}")
            continue

        t0 = time.perf_counter()
        bx, cf, roi_pct = run_inference(
            model, img, use_retigate, retina, device
        )
        t1 = time.perf_counter()

        roi_areas.append(roi_pct)
        frame_times.append((t1 - t0) * 1000)

        # Build torchmetrics update
        preds = [dict(
            boxes  = torch.tensor(
                bx, dtype=torch.float32
            ),
            scores = torch.tensor(
                cf, dtype=torch.float32
            ),
            labels = torch.zeros(
                len(bx), dtype=torch.int64
            ),
        )]
        gt_boxes = frame_gt[
            ["x1","y1","x2","y2"]
        ].values.astype(np.float32)
        targets = [dict(
            boxes  = torch.tensor(
                gt_boxes, dtype=torch.float32
            ),
            labels = torch.zeros(
                len(gt_boxes), dtype=torch.int64
            ),
        )]
        metric.update(preds, targets)

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    computed  = metric.compute()
    map50     = float(computed["map_50"])  * 100
    map75     = float(computed["map_75"])  * 100

    return {
        "map50":          map50,
        "map75":          map75,
        "mean_roi_area":  float(np.mean(roi_areas))
                          if roi_areas else 100.0,
        "median_roi_area":float(np.median(roi_areas))
                          if roi_areas else 100.0,
        "mean_latency_ms":float(np.mean(frame_times))
                          if frame_times else 0.0,
        "n_frames":       len(frame_times),
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args):
    data_root = Path(args.data_root)

    # Resolve sequence list
    if args.sequence == "active":
        sequences = ACTIVE_SEQUENCES
        logger.info(
            "Mode: active sequences "
            f"{sequences}"
        )
    else:
        sequences = [args.sequence.zfill(4)]
        logger.info(
            f"Mode: single sequence {sequences[0]}"
        )

    rows = []

    for seq in sequences:
        logger.info(f"\n{'─'*55}")
        logger.info(f"Sequence {seq}")
        logger.info(f"{'─'*55}")

        for det_name, det_path in DETECTORS.items():
            # Dense
            try:
                dense = evaluate_sequence(
                    seq, det_name, det_path,
                    use_retigate=False,
                    data_root=data_root
                )
            except FileNotFoundError as e:
                logger.warning(f"Skip dense: {e}")
                continue

            # Sparse
            try:
                sparse = evaluate_sequence(
                    seq, det_name, det_path,
                    use_retigate=True,
                    data_root=data_root
                )
            except FileNotFoundError as e:
                logger.warning(f"Skip sparse: {e}")
                continue

            retention_50 = (
                sparse["map50"] / dense["map50"] * 100
                if dense["map50"] > 0 else 0.0
            )
            retention_75 = (
                sparse["map75"] / dense["map75"] * 100
                if dense["map75"] > 0 else 0.0
            )

            rows.append({
                "Sequence":          seq,
                "Detector":          det_name,
                "Dense mAP@50":      dense["map50"],
                "Sparse mAP@50":     sparse["map50"],
                "Retention@50 (%)":  retention_50,
                "Dense mAP@75":      dense["map75"],
                "Sparse mAP@75":     sparse["map75"],
                "Retention@75 (%)":  retention_75,
                "Mean ROI area (%)": sparse["median_roi_area"],
                "Dense lat (ms)":    dense["mean_latency_ms"],
                "Sparse lat (ms)":   sparse["mean_latency_ms"],
                "N frames":          dense["n_frames"],
            })

            logger.info(
                f"  {det_name}: "
                f"Dense={dense['map50']:.2f}%  "
                f"Sparse={sparse['map50']:.2f}%  "
                f"Retention={retention_50:.1f}%  "
                f"ROI={sparse['median_roi_area']:.1f}%"
            )

    if not rows:
        logger.error("No results produced. Check data paths.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Aggregate across sequences
    agg = (
        df.groupby("Detector")
        .agg({
            "Dense mAP@50":      "mean",
            "Sparse mAP@50":     "mean",
            "Retention@50 (%)":  "mean",
            "Dense mAP@75":      "mean",
            "Sparse mAP@75":     "mean",
            "Retention@75 (%)":  "mean",
            "Mean ROI area (%)": "mean",
        })
        .reset_index()
    )

    # Save
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        out_dir / "table_detector_agnosticism.csv",
        index=False
    )
    agg.to_csv(
        out_dir / "table_detector_agnosticism_agg.csv",
        index=False
    )

    # Print
    print(f"\n{'='*65}")
    print("DETECTOR AGNOSTICISM — Per Sequence Results")
    print(f"{'='*65}")
    print(df[[
        "Sequence","Detector",
        "Dense mAP@50","Sparse mAP@50",
        "Retention@50 (%)","Mean ROI area (%)"
    ]].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    ))

    print(f"\n{'='*65}")
    print("AGGREGATED ACROSS SEQUENCES")
    print(f"{'='*65}")
    print(agg[[
        "Detector",
        "Dense mAP@50","Sparse mAP@50",
        "Retention@50 (%)","Mean ROI area (%)"
    ]].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    ))

    # LaTeX table
    print(f"\n\n% ── LATEX TABLE ─────────────────────────────")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(
        r"\caption{Detector agnosticism: RetiGate with "
        r"YOLO11m (one-stage) and RT-DETR-L "
        r"(transformer-based). Evaluated on KITTI Tracking "
        r"active sequences (0006, 0012, 0015, 0020) "
        r"where RetiGate ROI activates reliably. "
        r"Retention = Sparse mAP@50 / Dense mAP@50.}"
    )
    print(r"\label{tab:agnosticism}")
    print(r"\small")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(
        r"\textbf{Detector} & \textbf{Type} & "
        r"\textbf{Dense} & \textbf{Sparse} & "
        r"\textbf{Retention} & \textbf{Med.\ ROI} \\"
    )
    print(r"\midrule")

    types = {
        "YOLO11m":   "One-stage",
        "RT-DETR-L": "Transformer",
    }
    for _, row in agg.iterrows():
        det  = row["Detector"]
        typ  = types.get(det, "—")
        print(
            f"{det} & {typ} & "
            f"{row['Dense mAP@50']:.2f}\\% & "
            f"{row['Sparse mAP@50']:.2f}\\% & "
            f"{row['Retention@50 (%)']:.1f}\\% & "
            f"{row['Mean ROI area (%)']:.1f}\\% \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    logger.info(
        f"\nSaved → results/table_detector_agnosticism.csv"
        f"\nSaved → results/table_detector_agnosticism_agg.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RetiGate detector agnosticism study"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="active",
        help=(
            "KITTI sequence ID (e.g. '0006') "
            "or 'active' for all low-ego-motion "
            "sequences [default: active]"
        )
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/kitti/data_tracking_image",
        help="Path to KITTI tracking data root"
    )
    main(parser.parse_args())