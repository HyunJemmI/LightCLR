import os
import json
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class TuSimpleDataset(Dataset):
    """Minimal TuSimple loader (image → RGB tensor, label → lane points tensor)."""

    def __init__(self, root: str, split: str = "train", transform=None, image_size=(224, 224), max_lanes=4):
        self.root = root
        self.split = split  # train / val / test
        self.image_size = image_size
        self.max_lanes = max_lanes
        self.transform = transform or T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
        ])
        self.samples = self._load_annotations()

    # -----------------------------------------------------------------
    def _load_annotations(self) -> List[Tuple[str, list, list]]:
        label_file = os.path.join(self.root, f"label_data_{self.split}.json")
        data = []
        with open(label_file) as f:
            content = f.read().strip()
            if content.startswith("["):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.splitlines() if line.strip()]

        samples = []
        for entry in data:
            img_path = os.path.join(self.root, "clips", entry["raw_file"])
            samples.append((img_path, entry["lanes"], entry["h_samples"]))
        return samples

    # -----------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lanes, h_samples = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        target = self._encode_lanes(lanes, h_samples)
        return image, target

    # -----------------------------------------------------------------
    def _normalize_x(value: float, image_width: float = 1280.0) -> float:
        if value < 0:
            return -1.0
        return float(value) / image_width

    def _encode_lanes(self, lanes: list, h_samples: list):
        padded = np.full((self.max_lanes, len(h_samples)), -1.0, dtype=np.float32)
        for i, lane in enumerate(lanes[: self.max_lanes]):
            padded[i, :] = [self._normalize_x(x) for x in lane]
        return torch.tensor(padded)
