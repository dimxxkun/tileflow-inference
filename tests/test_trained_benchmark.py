import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from tileflow.benchmark import load_image
from tileflow.models import DenoisingCNN, load_denoiser


def test_saved_trained_model_runs_through_public_benchmark(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = DenoisingCNN()
    image = torch.rand(3, 31, 47)
    initial = model(image[None]).detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    (model(image[None]) - image[None]).square().mean().backward()
    optimizer.step()
    checkpoint, source = tmp_path / "model.pt", tmp_path / "image.npy"
    torch.save(
        {"architecture": "residual-rgb-3-16-3-v1", "state_dict": model.state_dict()}, checkpoint
    )
    np.save(source, image.numpy())
    restored = load_denoiser(checkpoint)
    with torch.inference_mode():
        expected = restored(image[None])[0].numpy()
    assert not torch.equal(initial, torch.from_numpy(expected)[None])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tileflow.benchmark",
            "benchmark",
            "--image",
            str(source),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "run"),
            "--tile-size",
            "16",
            "--overlap",
            "4",
            "--threads",
            "1",
            "--repeats",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "run/benchmark.json").read_text())
    assert len(report["fixture"]["image_sha256"]) == 64
    for row in report["results"]:
        assert row["parity_passed"]
        assert row["image_shape"] == [3, 31, 47]
        assert "argmax_agreement" not in row  # RGB regression has no class axis.
        np.testing.assert_allclose(
            np.load(tmp_path / f"run/{row['mode']}.npy"), expected, rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((8, 8, 3), np.float32),
        np.zeros((3, 8, 8), np.float64),
        np.full((3, 8, 8), np.nan, np.float32),
    ],
)
def test_invalid_external_image_is_rejected(tmp_path, array):
    path = tmp_path / "input.npy"
    np.save(path, array)
    with pytest.raises(ValueError, match="float32 RGB CHW"):
        load_image(path)
