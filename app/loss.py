import torch


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
