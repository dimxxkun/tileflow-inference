"""Train on real EuroSAT patches with synthetic Gaussian noise, then evaluate once.

Install the sibling geovision-bench package for its verified dataset loader.
This is a small denoising baseline, not satellite segmentation or sensor-noise modelling.
"""

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from geovision.data import digest, load_split
from torch.utils.data import DataLoader, TensorDataset

from tileflow.benchmark import write_json
from tileflow.models import DenoisingCNN, load_denoiser


def corrupt(clean, generator, sigma):
    noise = torch.randn(clean.shape, generator=generator) * sigma
    return (clean + noise).clamp(0, 1)


@torch.inference_mode()
def evaluate(model, noisy, clean, batch_size=32):
    squared_error = 0.0
    for inputs, targets in DataLoader(TensorDataset(noisy, clean), batch_size=batch_size):
        squared_error += float((model(inputs) - targets).square().sum())
    return squared_error / clean.numel()


def train(args):
    if min(args.epochs, args.threads, args.train_per_class, args.eval_per_class) < 1:
        raise ValueError("Epochs, threads and subset sizes must be positive")
    if args.eval_per_class < 20:
        raise ValueError("The 12x16 benchmark mosaic requires at least 20 test images per class")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use an empty output directory to preserve earlier experiments")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "threads": args.threads,
        "train_per_class": args.train_per_class,
        "eval_per_class": args.eval_per_class,
        "noise_sigma": 0.1,
        "validation_noise_seed": args.seed + 1,
        "test_noise_seed": args.seed + 2,
        "mosaic_noise_seed": args.seed + 3,
        "selection": "lowest fixed-noise validation MSE; no test-based model selection",
        "optimizer": "Adam; lr=0.001; batch_size=32",
        "architecture": "residual-rgb-3-16-3-v1",
        "manifest_sha256": digest(args.manifest),
        "subset_rule": "first N manifest rows per label in each existing split",
    }
    write_json(args.output / "config.json", config)
    train_set, train_rows = load_split(args.root, manifest, "train", limit=args.train_per_class)
    val_set, val_rows = load_split(args.root, manifest, "val", limit=args.eval_per_class)
    clean_train = train_set.tensors[0].float() / 255
    clean_val = val_set.tensors[0].float() / 255
    noisy_val = corrupt(clean_val, torch.Generator().manual_seed(args.seed + 1), 0.1)
    loader = DataLoader(
        TensorDataset(clean_train),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = DenoisingCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    noise_generator = torch.Generator().manual_seed(args.seed)
    history = []
    best = math.inf
    before, before_cpu = time.perf_counter(), time.process_time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for (clean,) in loader:
            noisy = corrupt(clean, noise_generator, 0.1)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(noisy) - clean).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(clean)
        model.eval()
        val_mse = evaluate(model, noisy_val, clean_val)
        row = {"epoch": epoch, "train_mse": total / len(clean_train), "val_mse": val_mse}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_mse < best:
            best = val_mse
            torch.save(
                {
                    "architecture": config["architecture"],
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "config": config,
                },
                args.output / "best.pt",
            )
    training_seconds = time.perf_counter() - before
    training_cpu_seconds = time.process_time() - before_cpu
    # Test pixels are loaded only after all model selection is finished.
    test_set, test_rows = load_split(args.root, manifest, "test", limit=args.eval_per_class)
    if set(row["sha256"] for row in train_rows + val_rows) & set(
        row["sha256"] for row in test_rows
    ):
        raise ValueError("Exact-image leakage into the test subset")
    clean_test = test_set.tensors[0].float() / 255
    noisy_test = corrupt(clean_test, torch.Generator().manual_seed(args.seed + 2), 0.1)
    model = load_denoiser(args.output / "best.pt")
    test_mse = evaluate(model, noisy_test, clean_test)
    baseline_mse = float((noisy_test - clean_test).square().mean())
    report = {
        "dataset": "EuroSAT RGB: real images, synthetic Gaussian corruption (sigma=0.1)",
        "counts": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "noisy_test_mse": baseline_mse,
        "denoised_test_mse": test_mse,
        "noisy_test_psnr_db": -10 * math.log10(baseline_mse),
        "denoised_test_psnr_db": -10 * math.log10(test_mse),
        "psnr_definition": "-10 log10(mean pixel MSE), unit data range; outputs not clipped",
        "selected_epoch": min(history, key=lambda row: row["val_mse"])["epoch"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_wall_seconds": training_seconds,
        "training_process_cpu_seconds": training_cpu_seconds,
        "training_timing_scope": "epoch loop including validation and checkpoint writes; input setup excluded",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "threads": args.threads,
            "device": "cpu",
        },
        "limitations": [
            "One seed and a small balanced subset; no claim of state-of-the-art accuracy.",
            "Image split, not geographic holdout; near duplicates are not controlled.",
            "Noise is simulated; no evidence of performance on real sensor corruption.",
            "The performance mosaic has artificial patch adjacency and no georeferencing.",
        ],
    }
    write_json(args.output / "history.json", history)
    write_json(args.output / "metrics.json", report)
    write_json(
        args.output / "subset.json", {"train": train_rows, "val": val_rows, "test": test_rows}
    )
    # Permute 192 held-out tiles deterministically; this is a systems fixture, not a scene.
    order = torch.randperm(len(clean_test), generator=torch.Generator().manual_seed(args.seed))[
        :192
    ]
    mosaic = clean_test[order].reshape(12, 16, 3, 64, 64).permute(2, 0, 3, 1, 4)
    mosaic = mosaic.reshape(3, 768, 1024).contiguous()
    mosaic = corrupt(mosaic, torch.Generator().manual_seed(args.seed + 3), 0.1)
    np.save(args.output / "mosaic.npy", mosaic.numpy())
    write_json(
        args.output / "mosaic.json",
        {
            "shape": list(mosaic.shape),
            "test_row_indices": order.tolist(),
            "sha256": digest(args.output / "mosaic.npy"),
            "checkpoint_sha256": digest(args.output / "best.pt"),
        },
    )
    plot_examples(model, noisy_test, clean_test, args.output)
    print(json.dumps(report), flush=True)


@torch.inference_mode()
def plot_examples(model, noisy, clean, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = [0, len(clean) // 3, 2 * len(clean) // 3]
    predictions = model(noisy[indices])
    fig, axes = plt.subplots(3, 3, figsize=(7, 7), layout="constrained")
    for row, index in enumerate(indices):
        for col, image in enumerate([clean[index], noisy[index], predictions[row]]):
            axes[row, col].imshow(image.clamp(0, 1).permute(1, 2, 0).numpy())
            axes[row, col].axis("off")
    for ax, title in zip(axes[0], ["Original", "Synthetic noise", "Trained CNN"]):
        ax.set_title(title)
    fig.suptitle("EuroSAT test subset · fixed indices, no visual selection")
    fig.savefig(output / "examples.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--eval-per-class", type=int, default=20)
    train(parser.parse_args())
