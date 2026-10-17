"""Image ops that move tiles between the art grid and pixel space.

Ground-truth rules (SPEC §1, §2.4 — do not drift):
- upscale for editing/reading: **nearest neighbor** (a 32x32 art tile at
  write_size/32 is exactly "32x32 pixel art at 1:1 art-pixel scale");
- downscale write_size -> art_size: per (write/art) px block, quantize the
  block's pixels to the global palette first, then take the **mode** palette
  index; median RGB -> nearest as the tie fallback; **never a plain mean**
  (means blend edges into mud).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .palette import Palette


def nearest_resize(rgb: np.ndarray, size: int) -> np.ndarray:
    """(h,w,3) uint8 -> (size,size,3) uint8, nearest neighbor (square output)."""
    rgb = np.ascontiguousarray(rgb)
    img = Image.fromarray(rgb, mode="RGB")
    if img.size == (size, size):
        return rgb.copy()
    out = img.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(out)


def upscale_to_write(art_rgb: np.ndarray, write_size: int) -> np.ndarray:
    """art_size square (ground truth) -> write_size square, nearest."""
    return nearest_resize(art_rgb, write_size)


def block_mode_downscale(rgb: np.ndarray, art_size: int, palette: Palette) -> np.ndarray:
    """(write,write,3) uint8 -> (art,art) uint8 palette indices.

    Per art-size block: mode over palette indices of the block's pixels;
    if no unique mode (tie), median pixel color -> nearest palette index.
    """
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    if h != w or h % art_size != 0:
        raise ValueError(f"block downscale needs square rgb with side divisible by art_size {art_size}, got {w}x{h}")
    b = h // art_size

    idx = palette.nearest(rgb)                            # (h, w)
    blocks = idx.reshape(art_size, b, art_size, b)
    flat = blocks.transpose(0, 2, 1, 3).reshape(art_size * art_size, b * b)  # (A*A, b*b) per (a,b) block

    counts = np.zeros((art_size * art_size, palette.n), dtype=np.uint16)
    np.add.at(counts, (np.arange(art_size * art_size)[:, None], flat), 1)
    best = counts.max(-1)
    argmax = counts.argmax(-1)
    tied = (counts == best[:, None]).sum(-1) > 1

    out = argmax
    if tied.any():
        blocks_px = rgb.reshape(art_size, b, art_size, b, 3)
        blocks_px = blocks_px.transpose(0, 2, 1, 3, 4).reshape(art_size * art_size, b * b, 3)[tied]
        med = np.median(blocks_px.astype(np.int16), axis=1)
        out[tied] = palette.nearest(np.round(med).astype(np.uint8))
    return out.reshape(art_size, art_size).astype(np.uint8)
