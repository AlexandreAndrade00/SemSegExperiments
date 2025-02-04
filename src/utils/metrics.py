import torch
from torch import Tensor, sigmoid
from sklearn.metrics import jaccard_score
from torchmetrics.functional.segmentation import mean_iou as miou

def iou(y_true: Tensor, y_pred: Tensor) -> float:
    y_true = y_true.cpu().numpy().ravel()

    y_pred = (sigmoid(y_pred) > 0.5).cpu().float().numpy().ravel()

    return jaccard_score(y_true, y_pred)

def mean_iou(y_true: Tensor, y_pred: Tensor, num_classes: int) -> float:
    return miou(y_pred.unsqueeze(1).to(torch.uint8), y_true.unsqueeze(1).to(torch.uint8), num_classes).cpu().mean().item()