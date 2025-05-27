from sklearn.metrics import f1_score
import torch
from torch import Tensor


def f1_score_multiclass(pred_mask: Tensor, true_mask: Tensor) -> Tensor:
    score = f1_score(true_mask.cpu(), pred_mask.cpu(), average=None)

    # print(pred_mask)
    # print(true_mask)
    # print(score)

    return torch.tensor(score)


def calculate_iou(pred_mask: Tensor, true_mask: Tensor, num_classes: int) -> Tensor:
    device = pred_mask.device

    true_mask = true_mask.to(device)

    if pred_mask.ndim == 2:
        pred_mask = pred_mask.unsqueeze(0)
        true_mask = true_mask.unsqueeze(0)

    if pred_mask.ndim == 4:
        pred_mask = pred_mask.argmax(dim=1)

    batch_size = pred_mask.shape[0]

    iou_per_class = torch.zeros(num_classes, device=device)

    for class_id in range(num_classes):
        intersection = torch.zeros(batch_size, device=device)
        union = torch.zeros(batch_size, device=device)

        for batch_idx in range(batch_size):
            intersection[batch_idx] = torch.sum(
                torch.where(
                    (pred_mask[batch_idx] == class_id)
                    & (true_mask[batch_idx] == class_id),
                    1,
                    0,
                ).float()
            )

            union[batch_idx] = torch.sum(
                torch.where(
                    (pred_mask[batch_idx] == class_id)
                    | (true_mask[batch_idx] == class_id),
                    1,
                    0,
                ).float()
            )

        iou_per_class[class_id] = (intersection.sum() + 1e-7) / (union.sum() + 1e-7)

    return iou_per_class


def calculate_dice(pred_mask: Tensor, true_mask: Tensor, num_classes: int) -> Tensor:
    device = pred_mask.device

    true_mask = true_mask.to(device)

    if pred_mask.ndim == 2:
        pred_mask = pred_mask.unsqueeze(0)
        true_mask = true_mask.unsqueeze(0)

    if pred_mask.ndim == 4:
        pred_mask = pred_mask.argmax(dim=1)

    batch_size = pred_mask.shape[0]

    dice_per_class = torch.zeros(num_classes, device=device)

    for class_id in range(num_classes):
        intersection = torch.zeros(batch_size, device=device)
        pred_class = torch.zeros(batch_size, device=device)
        true_class = torch.zeros(batch_size, device=device)

        for batch_idx in range(batch_size):
            intersection[batch_idx] = torch.sum(
                torch.where(
                    (pred_mask[batch_idx] == class_id)
                    & (true_mask[batch_idx] == class_id),
                    1,
                    0,
                ).float()
            )

            pred_class[batch_idx] = torch.sum(
                torch.where(
                    pred_mask[batch_idx] == class_id,
                    1,
                    0,
                ).float()
            )

            true_class[batch_idx] = torch.sum(
                torch.where(
                    true_mask[batch_idx] == class_id,
                    1,
                    0,
                ).float()
            )

        dice_per_class[class_id] = (2 * (intersection.sum() + 1e-7)) / (
            pred_class.sum() + true_class.sum() + 1e-7
        )

    return dice_per_class
