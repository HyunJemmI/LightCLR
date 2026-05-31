import torch
import torch.nn.functional as F


def lane_point_loss(pred, target):
    mask = target >= 0
    if not mask.any():
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[mask], target[mask])


def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)

    similarity = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, -1e9)

    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])
    return F.cross_entropy(similarity, labels)
