import torch
from torch import nn


def local_model(seed=2026, width=16):
    """Untrained local convolution fixture, radius two, for numerical systems tests."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, 4, 3, padding=1)
        )
    return model.eval()
