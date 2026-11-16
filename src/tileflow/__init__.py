"""Spatially preserving, CPU tiled inference with explicit context boundaries."""

from .core import TileConfig, tiled_predict

__all__ = ["TileConfig", "tiled_predict"]
