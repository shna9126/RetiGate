#!/usr/bin/env python3
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
class SAHIBaseline:
    def __init__(self, model_path='yolo11n.pt', conf=0.25):
        # Initialize the SAHI-wrapped model
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8', # SAHI uses yolov8 type for yolov11/v8/v9/v10
            model_path=model_path,
            confidence_threshold=conf,
            device="cpu" # For your MacBook CPU run
        )

    def predict(self, image_path, slice_size=640, overlap=0.2):
        """Standard SAHI grid-based inference."""
        result = get_sliced_prediction(
            image_path,
            self.detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=0
        )
        # We extract the boxes to compare with RetiGate later
        boxes = []
        for det in result.object_prediction_list:
            boxes.append(det.bbox.to_xyxy())
        return np.array(boxes), len(result.object_prediction_list)