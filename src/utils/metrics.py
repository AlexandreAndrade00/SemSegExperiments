from sklearn.metrics import jaccard_score
from torch import Tensor
from torchmetrics.functional.segmentation import mean_iou as miou


def iou(y_true: Tensor, y_pred: Tensor) -> float:
    y_true = y_true.cpu().numpy().ravel()

    y_pred = y_pred.cpu().numpy().ravel()

    return jaccard_score(y_true, y_pred)


def mean_iou(y_true: Tensor, y_pred: Tensor, num_classes: int) -> float:
    return miou(y_pred, y_true, num_classes).cpu().mean().item()
