import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet34(pretrained: bool = False, simclr_weights: str | None = None) -> nn.Module:
    """Return ResNet‑34 feature extractor.

    Args:
        pretrained (bool): If True, load ImageNet weights.
        simclr_weights (str | None): Optional path to SimCLR‑pretrained weights.
    """
    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    backbone = models.resnet34(weights=weights)
    encoder = nn.Sequential(*list(backbone.children())[:-2])

    if simclr_weights:
        state = torch.load(simclr_weights, map_location="cpu")
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        print(f"[encoder] Loaded SimCLR weights → missing={len(missing)}, unexpected={len(unexpected)}")

    return encoder


def set_trainable(module: nn.Module, trainable: bool):
    for p in module.parameters():
        p.requires_grad = trainable
    module.train(trainable)


def freeze(module: nn.Module):
    set_trainable(module, False)
