import torch

CFG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 16,
    "num_epochs": 30,
    "num_anchors": 4,
    "image_size": (224, 224),
    "lane_points": 56,
    "data_root": "dataset/tusimple",
    "contrastive_roots": [
        "dataset/tusimple/clips",
        "dataset/bdd100k/images/100k/train",
    ],
    "simclr_weights": "outputs/simclr_encoder.pt",
    "baseline_weights": "outputs/baseline_laneatt.pt",
    "finetuned_weights": "outputs/finetuned_laneatt.pt",
    "learning_rate": 1e-3,
    "encoder_learning_rate": 1e-4,
    "weight_decay": 1e-4,
}
