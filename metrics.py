from torch import Tensor, sigmoid
from sklearn.metrics import jaccard_score

def iou(y_true: Tensor, y_pred: Tensor) -> float:
    y_true = y_true.cpu().numpy().ravel()
    y_pred = (sigmoid(y_pred) > 0.5).cpu().numpy().ravel()

    return jaccard_score(y_true, y_pred)
