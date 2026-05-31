import torch


def tusimple_accuracy(pred: torch.Tensor, gt: torch.Tensor, thresh: float = 5.0):
    """Point-wise lane accuracy on normalized x coordinates."""
    normalized_thresh = thresh / 1280.0
    mask = gt >= 0
    if not mask.any():
        return 0.0
    diff = (pred - gt).abs()
    correct = (diff[mask] < normalized_thresh).float().mean()
    return correct.item()
