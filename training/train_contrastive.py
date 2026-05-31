from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.lane_det_config import CFG
from data.contrastive_dataset import NoisePairDataset
from models.encoder import get_resnet34
from training.losses import nt_xent_loss
from utils.logger import get_logger


logger = get_logger("contrastive")


class ProjectionHead(nn.Module):
    def __init__(self, in_channels=512, hidden_dim=512, out_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features):
        pooled = self.pool(features).flatten(1)
        return self.layers(pooled)


class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = get_resnet34(pretrained=False)
        self.projector = ProjectionHead()

    def forward(self, images):
        return self.projector(self.encoder(images))


def main():
    dataset = NoisePairDataset(CFG["contrastive_roots"], image_size=CFG["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    model = SimCLRModel().to(CFG["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["learning_rate"],
        weight_decay=CFG["weight_decay"],
    )

    for epoch in range(CFG["num_epochs"]):
        model.train()
        running_loss = 0.0

        for clean_view, noisy_view in loader:
            clean_view = clean_view.to(CFG["device"])
            noisy_view = noisy_view.to(CFG["device"])

            z_clean = model(clean_view)
            z_noisy = model(noisy_view)
            loss = nt_xent_loss(z_clean, z_noisy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        mean_loss = running_loss / max(1, len(loader))
        logger.info("epoch %s contrastive-loss %.4f", epoch, mean_loss)

    Path(CFG["simclr_weights"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), CFG["simclr_weights"])
    logger.info("Saved encoder -> %s", CFG["simclr_weights"])


if __name__ == "__main__":
    main()
