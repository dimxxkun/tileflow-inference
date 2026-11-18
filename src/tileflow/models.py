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


class DenoisingCNN(nn.Module):
    """Radius-two residual RGB denoiser; no global or tile-dependent operations."""

    def __init__(self):
        super().__init__()
        self.noise = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 3, 3, padding=1)
        )

    def forward(self, image):
        return image - self.noise(image)


def load_denoiser(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "residual-rgb-3-16-3-v1":
        raise ValueError("Unsupported denoising checkpoint architecture")
    model = DenoisingCNN()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite weights")
    return model.eval()
