from torchmetrics.detection import MeanAveragePrecision
import torch


class DetectionEvaluator:
    def __init__(self):
        self.metric = MeanAveragePrecision(
            box_format='xyxy',
            iou_thresholds=[0.5, 0.55, 0.60, 0.65, 0.70,
                            0.75, 0.80, 0.85, 0.90, 0.95],
            class_metrics=True,
        )

    def update(self, pred_boxes, pred_scores, pred_labels,
               gt_boxes, gt_labels):
        """
        pred_boxes: tensor [N, 4] xyxy
        pred_scores: tensor [N]
        pred_labels: tensor [N] int
        gt_boxes: tensor [M, 4] xyxy
        gt_labels: tensor [M] int
        """
        preds = [{'boxes': pred_boxes,
                  'scores': pred_scores,
                  'labels': pred_labels}]
        targets = [{'boxes': gt_boxes,
                    'labels': gt_labels}]
        self.metric.update(preds, targets)

    def compute(self) -> dict:
        result = self.metric.compute()
        return {k: float(v) for k, v in result.items()
                if v.numel() == 1}

    def reset(self):
        self.metric.reset()
