"""Streaming tile activations; the source image and stitched output remain resident."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TileConfig:
    tile_size: int = 256
    overlap: int = 64
    halo: int = 2
    batch_size: int = 4
    blend: str = "hann"

    def __post_init__(self):
        if any(
            type(value) is not int
            for value in (self.tile_size, self.overlap, self.halo, self.batch_size)
        ):
            raise ValueError("Tile dimensions, halo and batch size must be integers")
        if self.tile_size < 1 or not 0 <= self.overlap < self.tile_size:
            raise ValueError("Require tile_size > 0 and 0 <= overlap < tile_size")
        if self.halo < 0 or self.batch_size < 1 or self.blend not in ("uniform", "hann"):
            raise ValueError("Require halo >= 0, batch_size > 0, and uniform or hann blending")


@dataclass(frozen=True)
class Tile:
    y: int
    x: int
    height: int
    width: int
    input_y: int
    input_x: int
    input_height: int
    input_width: int


def starts(length, size, stride):
    if length <= size:
        return [0]
    positions = list(range(0, length - size + 1, stride))
    if positions[-1] != length - size:
        positions.append(length - size)
    return positions


def tiles(height, width, config):
    stride = config.tile_size - config.overlap
    for y in starts(height, config.tile_size, stride):
        for x in starts(width, config.tile_size, stride):
            core_h = min(config.tile_size, height - y)
            core_w = min(config.tile_size, width - x)
            iy, ix = max(0, y - config.halo), max(0, x - config.halo)
            bottom, right = (
                min(height, y + core_h + config.halo),
                min(width, x + core_w + config.halo),
            )
            yield Tile(y, x, core_h, core_w, iy, ix, bottom - iy, right - ix)


def window(height, width, blend):
    if blend == "uniform":
        return torch.ones((height, width), dtype=torch.float32)
    # Strictly positive edges keep even singly covered boundary pixels defined.
    vertical = torch.hann_window(height, periodic=False).clamp_min(1e-3)
    horizontal = torch.hann_window(width, periodic=False).clamp_min(1e-3)
    return vertical[:, None] * horizontal[None, :]


def _batches(items, batch_size):
    """Only consecutive equal-shaped inputs are batched; no unbounded shape buckets."""
    pending = []
    for tile, crop in items:
        if pending and (crop.shape != pending[0][1].shape or len(pending) == batch_size):
            yield pending
            pending = []
        pending.append((tile, crop))
    if pending:
        yield pending


@torch.inference_mode()
def tiled_predict(model: nn.Module, image: torch.Tensor, config=TileConfig(), *, materialize=False):
    """Return CPU float32 [channels,H,W] predictions, preserving the model's boundary.

    model must be in eval mode and map BCHW to BCHW at unchanged spatial resolution.
    Exact whole-frame parity requires local operations, sufficient halo, and no
    tile-dependent normalization/global context. This implementation is CPU-only.
    materialize=True is a deliberately memory-heavy benchmark control, not a fast path.
    """
    if model.training:
        raise ValueError("Call model.eval() before inference")
    if image.ndim != 3 or min(image.shape) < 1:
        raise ValueError("Expected nonempty CHW input")
    if image.device.type != "cpu" or image.dtype != torch.float32:
        raise ValueError("Expected a CPU float32 input")
    if not torch.isfinite(image).all():
        raise ValueError("Input contains non-finite values")
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.device.type != "cpu":
            raise ValueError("This implementation supports CPU models only")
    _, height, width = image.shape
    iterator = (
        (
            tile,
            image[
                :,
                tile.input_y : tile.input_y + tile.input_height,
                tile.input_x : tile.input_x + tile.input_width,
            ],
        )
        for tile in tiles(height, width, config)
    )
    if materialize:
        iterator = [(tile, crop.clone()) for tile, crop in iterator]
    output = None
    denominator = torch.zeros((height, width), dtype=torch.float32)
    for batch in _batches(iterator, config.batch_size):
        predictions = model(torch.stack([crop for _, crop in batch]))
        if not isinstance(predictions, torch.Tensor):
            raise TypeError("Model must return a tensor of dense predictions")
        expected_hw = batch[0][1].shape[-2:]
        if (
            predictions.ndim != 4
            or predictions.shape[0] != len(batch)
            or tuple(predictions.shape[-2:]) != tuple(expected_hw)
            or predictions.shape[1] < 1
        ):
            raise ValueError("Model must preserve batch and spatial dimensions in BCHW output")
        if predictions.device.type != "cpu" or not torch.is_floating_point(predictions):
            raise ValueError("Model must return floating-point CPU predictions")
        if not torch.isfinite(predictions).all():
            raise FloatingPointError("Model returned non-finite predictions")
        if output is None:
            output = torch.zeros((predictions.shape[1], height, width), dtype=torch.float32)
        elif output.shape[0] != predictions.shape[1]:
            raise ValueError("Model changed its output channel count between batches")
        for (tile, _), prediction in zip(batch, predictions):
            offset_y, offset_x = tile.y - tile.input_y, tile.x - tile.input_x
            core = prediction[
                :, offset_y : offset_y + tile.height, offset_x : offset_x + tile.width
            ].float()
            weights = window(tile.height, tile.width, config.blend)
            output[:, tile.y : tile.y + tile.height, tile.x : tile.x + tile.width].add_(
                core * weights
            )
            denominator[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width].add_(weights)
    if output is None or not torch.all(denominator > 0):
        raise RuntimeError("Internal coverage failure")
    output.div_(denominator.unsqueeze(0))
    return output
