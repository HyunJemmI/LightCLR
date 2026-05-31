from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.lane_det_config import CFG
from data.tusimple_dataset import TuSimpleDataset
from models.encoder import get_resnet34
from models.lane_att import LaneDetectionModel
from models.lane_head import LaneATTHead
from training.losses import lane_point_loss
from utils.logger import get_logger
from utils.metrics import tusimple_accuracy


logger = get_logger("finetune")


def main():
    train_ds = TuSimpleDataset(CFG["data_root"], "train", image_size=CFG["image_size"])
    val_ds = TuSimpleDataset(CFG["data_root"], "val", image_size=CFG["image_size"])
    train_ld = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=4)
    val_ld = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False)

    encoder = get_resnet34(False, CFG["simclr_weights"])
    head = LaneATTHead(512, CFG["num_anchors"], CFG["lane_points"])
    model = LaneDetectionModel(encoder, head, freeze_encoder=False).to(CFG["device"])

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": CFG["encoder_learning_rate"]},
            {"params": model.head.parameters(), "lr": CFG["learning_rate"]},
        ],
        weight_decay=CFG["weight_decay"],
    )

    for epoch in range(CFG["num_epochs"]):
        model.train()
        running_loss = 0.0

        for imgs, targets in train_ld:
            imgs = imgs.to(CFG["device"])
            targets = targets.to(CFG["device"])

            pred = model(imgs)
            loss = lane_point_loss(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        acc = 0.0
        with torch.no_grad():
            for imgs, targets in val_ld:
                imgs = imgs.to(CFG["device"])
                targets = targets.to(CFG["device"])
                pred = model(imgs)
                acc += tusimple_accuracy(pred, targets)

        mean_loss = running_loss / max(1, len(train_ld))
        mean_acc = acc / max(1, len(val_ld))
        logger.info("epoch %s train-loss %.4f val-acc %.4f", epoch, mean_loss, mean_acc)

    Path(CFG["finetuned_weights"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CFG["finetuned_weights"])
    logger.info("Saved fine-tuned weights -> %s", CFG["finetuned_weights"])


if __name__ == "__main__":
    main()
