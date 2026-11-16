"""Export a reproducible context-ablation plot; no trained segmentation claim."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tileflow import TileConfig, tiled_predict
from tileflow.models import local_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/seam-diagnostic.png"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    image = torch.rand((3, 193, 257), generator=torch.Generator().manual_seed(2026))
    model = local_model()
    with torch.inference_mode():
        reference = model(image.unsqueeze(0))[0]
    errors = []
    for halo in (0, 2):
        prediction = tiled_predict(model, image, TileConfig(64, 0, halo, 4, "uniform"))
        errors.append((prediction - reference).abs().amax(0).numpy())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, error, halo in zip(axes, errors, (0, 2)):
        heatmap = ax.imshow(error, cmap="magma", vmin=0, vmax=errors[0].max())
        ax.set_title(f"Halo {halo} px · max error {error.max():.2e}")
        ax.set_axis_off()
    fig.colorbar(heatmap, ax=axes, label="Max absolute logit error across channels", shrink=0.75)
    fig.suptitle("Tile boundaries need real convolution context")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
