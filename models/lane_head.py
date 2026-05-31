import torch.nn as nn
import torch.nn.functional as F
import torch


class LaneATTHead(nn.Module):
    """Minimal lane head that predicts normalized x positions per anchor."""

    def __init__(self, in_channels: int, num_anchors: int = 4, num_points: int = 56):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_points = num_points
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, 256)
        self.fc2 = nn.Linear(256, num_anchors * num_points)

    def forward(self, feat):
        x = self.gap(feat).flatten(1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        out = out.view(out.size(0), self.num_anchors, self.num_points)
        return torch.sigmoid(out)
