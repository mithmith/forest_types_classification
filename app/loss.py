import torch
import numpy as np

def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Intersection over Union (IoU) for binary masks.
    :param pred_mask: Predicted mask (logits or probabilities).
    :param true_mask: Ground truth mask.
    :param threshold: Threshold to binarize the predicted mask.
    :return: IoU score.
    """
    pred_mask = (pred_mask > threshold).float()  # Binarize predicted mask
    intersection = (pred_mask * true_mask).sum()  # Intersection area
    union = pred_mask.sum() + true_mask.sum() - intersection  # Union area
    iou = intersection / union if union > 0 else 0.0  # IoU formula
    return iou.item()


def iou_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Вычисление IoU Loss.
    :param pred: Предсказанная маска (без sigmoid, сырые логиты).
    :param target: Истинная маска.
    :param smooth: Маленькая добавка для предотвращения деления на 0.
    :return: Значение IoU Loss.
    """
    pred = torch.sigmoid(pred)  # Преобразуем логиты в вероятности
    # if np.all(np.isnan(pred.detach().cpu().numpy())): print("pred is NaN")
    # if np.all(np.isnan(target.detach().cpu().numpy())): print("target is NaN")
    intersection = (pred * target).sum(dim=(2, 3))  # Пересечение
    # print(f"intersection={intersection}")
    union = (pred + target - pred * target).sum(dim=(2, 3))  # Объединение
    # print(f"union={union}")
    iou = (intersection + smooth) / (union + smooth)  # IoU
    # print(f"iou={iou}")
    # print(f"loss={1 - iou.mean()}")
    return 1 - iou.mean()  # Возвращаем 1 - IoU
