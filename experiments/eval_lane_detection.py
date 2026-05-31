import time
import torch
from torch.utils.data import DataLoader

from configs.lane_det_config import CFG
from models.encoder import get_resnet34
from models.lane_head import LaneATTHead
from models.lane_att import LaneDetectionModel
from data.tusimple_dataset import TuSimpleDataset
from utils.metrics import tusimple_accuracy


def build_model(weight_path: str, freeze_enc: bool):
    encoder = get_resnet34(False).to(CFG["device"])
    head = LaneATTHead(512, CFG["num_anchors"], CFG["lane_points"]).to(CFG["device"])
    model = LaneDetectionModel(encoder, head, freeze_encoder=freeze_enc).to(CFG["device"])
    state = torch.load(weight_path, map_location=CFG["device"])
    model.load_state_dict(state)
    model.eval()
    return model


def measure_fps(model, sample):
    torch.cuda.synchronize()
    st = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(sample)
    torch.cuda.synchronize(); return 100 / (time.time() - st)


if __name__ == "__main__":
    test_ds = TuSimpleDataset(CFG["data_root"], "test")
    test_ld = DataLoader(test_ds, batch_size=1, shuffle=False)
    sample, _ = test_ds[0]
    sample = sample.unsqueeze(0).to(CFG["device"])

    checkpoints = {
        "baseline": ("outputs/baseline_laneatt.pt", False),
        "simclr_ft": ("outputs/finetuned_laneatt.pt", False),
    }

    for name, (ckpt, frz) in checkpoints.items():
        model = build_model(ckpt, frz)
        # accuracy
        acc = 0.0
        with torch.no_grad():
            for imgs, gts in test_ld:
                imgs, gts = imgs.to(CFG["device"]), gts.to(CFG["device"])
                pred = model(imgs)
                acc += tusimple_accuracy(pred, gts)
        acc /= len(test_ld)
        fps = measure_fps(model, sample)
        print(f"{name}: acc={acc:.4f} fps={fps:.2f}")
