from pathlib import Path
from typing import Iterable

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class NoisePairDataset(Dataset):
    """Return positive pairs made from the same image and different noise views."""

    def __init__(self, roots: Iterable[str], image_size=(224, 224)):
        self.image_paths = []
        for root in roots:
            root_path = Path(root)
            if root_path.exists():
                self.image_paths.extend(
                    path for path in root_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
                )

        if not self.image_paths:
            raise RuntimeError(f"No training images found in roots: {list(roots)}")

        self.clean_transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
        ])
        self.noisy_transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
            T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.5),
            T.ToTensor(),
            T.Lambda(lambda image: (image + 0.08 * torch.randn_like(image)).clamp(0.0, 1.0)),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.clean_transform(image), self.noisy_transform(image)
