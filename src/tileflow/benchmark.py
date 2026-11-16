"""Run each configuration in a fresh process; compare its output to whole-frame logits."""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil
import torch

from .core import TileConfig, tiled_predict
from .models import local_model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def worker(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(2026)
    image = torch.rand((3, args.height, args.width), generator=generator)
    model = local_model()
    config = TileConfig(args.tile_size, args.overlap, args.halo, args.batch_size)
    process = psutil.Process()
    base_rss = process.memory_info().rss
    peak_rss = [base_rss]
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.005):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    sampler = threading.Thread(target=monitor, daemon=True)
    sampler.start()
    wall, cpu = [], []
    try:
        for index in range(args.repeats + 1):
            before_cpu, before = time.process_time(), time.perf_counter()
            with torch.inference_mode():
                if args.mode == "whole":
                    output = model(image.unsqueeze(0))[0]
                else:
                    output = tiled_predict(
                        model, image, config, materialize=args.mode == "materialized"
                    )
            wall.append(time.perf_counter() - before)
            cpu.append(time.process_time() - before_cpu)
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            if index < args.repeats:
                del output
    finally:
        stop.set()
        sampler.join()
    np.save(args.output.with_suffix(".npy"), output.numpy())
    report = {
        "mode": args.mode,
        "image_shape": list(image.shape),
        "tile_config": asdict(config),
        "cold_seconds": wall[0],
        "warm_seconds": wall[1:],
        "warm_median_seconds": statistics.median(wall[1:]),
        "warm_median_megapixels_per_second": args.height
        * args.width
        / 1e6
        / statistics.median(wall[1:]),
        "cold_process_cpu_seconds": cpu[0],
        "warm_process_cpu_seconds": cpu[1:],
        "baseline_rss_mib": base_rss / 2**20,
        "sampled_peak_rss_mib": peak_rss[0] / 2**20,
        "sampled_peak_increase_mib": (peak_rss[0] - base_rss) / 2**20,
        "memory_sampling_interval_ms": 5,
        "repeats": args.repeats,
    }
    write_json(args.output, report)


def benchmark(args):
    if min(args.height, args.width, args.threads, args.repeats) < 1:
        raise ValueError("Dimensions, threads and repeats must be positive")
    config = TileConfig(args.tile_size, args.overlap, args.halo, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    reference = None
    for mode in ("whole", "streaming", "materialized"):
        destination = args.output / f"{mode}.json"
        command = [
            sys.executable,
            "-m",
            "tileflow.benchmark",
            "_worker",
            "--mode",
            mode,
            "--output",
            str(destination),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--threads",
            str(args.threads),
            "--repeats",
            str(args.repeats),
            "--tile-size",
            str(config.tile_size),
            "--overlap",
            str(config.overlap),
            "--halo",
            str(config.halo),
            "--batch-size",
            str(config.batch_size),
        ]
        subprocess.run(command, check=True, timeout=300)
        report = json.loads(destination.read_text(encoding="utf-8"))
        prediction = np.load(destination.with_suffix(".npy"), mmap_mode="r")
        if reference is None:
            reference = prediction
        error = np.abs(prediction - reference)
        report["max_absolute_error"] = float(error.max())
        report["mean_absolute_error"] = float(error.mean())
        report["parity_passed"] = bool(np.allclose(prediction, reference, rtol=1e-5, atol=1e-6))
        report["argmax_agreement"] = float((prediction.argmax(0) == reference.argmax(0)).mean())
        results.append(report)
        print(json.dumps(report), flush=True)
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "threads": args.threads,
            "device": "cpu",
        },
        "fixture": "seed 2026 uniform random RGB; untrained 3->16->4 two-layer 3x3 CNN",
        "timing_scope": "input/model setup excluded; tile planning, stacking, model and stitching included",
        "memory_scope": "RSS sampled every 5ms during cold and warm runs; setup baseline included; estimate, not allocator peak",
        "tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "results": results,
    }
    write_json(args.output / "benchmark.json", report)
    plot_report(report, args.output)
    if not all(item["parity_passed"] for item in results):
        raise RuntimeError("Numerical parity failed; results retained for diagnosis")


def plot_report(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["results"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    names = [row["mode"].capitalize() for row in rows]
    colours = ["#a6b4bf", "#167d9a", "#d89154"]
    for ax, key, label in (
        (axes[0], "warm_median_seconds", "Warm median · seconds"),
        (axes[1], "sampled_peak_rss_mib", "Sampled peak RSS · MiB"),
    ):
        bars = ax.bar(names, [row[key] for row in rows], color=colours)
        ax.bar_label(bars, fmt="%.2f", padding=4)
        ax.set_ylabel(label)
        ax.set_ylim(0, max(row[key] for row in rows) * 1.2)
        ax.spines[["top", "right"]].set_visible(False)
    shape = rows[0]["image_shape"]
    fig.suptitle(
        f"CPU tiled inference · {shape[1]} × {shape[2]} synthetic image\n"
        f"{report['environment']['threads']} threads · identical local CNN weights"
    )
    fig.savefig(output / "tradeoff.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["benchmark", "_worker"])
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--halo", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--mode", choices=["whole", "streaming", "materialized"], default="streaming"
    )
    args = parser.parse_args()
    (worker if args.command == "_worker" else benchmark)(args)


if __name__ == "__main__":
    main()
