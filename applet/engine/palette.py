"""Global fixed palette: the ground-truth color space of the CA.

A cell's ground truth is palette-indexed (SPEC §1); every image that enters
the field is quantized to this palette (nearest color in RGB), and every
image that leaves it is decoded through it. Format on disk: JSON
(`{"colors": [[r,g,b], ...]}` or a bare list) or `.npy` (n,3) uint8.
The default 64-color palette is picked in experiment E5.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Palette:
    colors: np.ndarray  # (n, 3) uint8

    def __post_init__(self):
        c = np.asarray(self.colors)
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError(f"palette colors must be (n,3), got {c.shape}")
        if not (c.dtype == np.uint8):
            c = c.astype(np.uint8)
        object.__setattr__(self, "colors", c)

    @property
    def n(self) -> int:
        return int(self.colors.shape[0])

    # -- color math ------------------------------------------------------ #
    def nearest(self, rgb: np.ndarray) -> np.ndarray:
        """Map (t,3) or (h,w,3) uint8 pixels to the nearest palette index.

        An exact palette match always wins (distance 0 is the minimum)."""
        rgb = np.ascontiguousarray(rgb)
        if rgb.ndim not in (2, 3) or rgb.shape[-1] != 3:
            raise ValueError(f"expected (t,3) or (h,w,3) rgb, got {rgb.shape}")
        diff = rgb.astype(np.int32)[..., None, :] - self.colors.astype(np.int32)[None, None, :, :]
        d = (diff * diff).sum(-1)
        return d.argmin(-1).reshape(rgb.shape[:-1]).astype(np.uint8)

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """(h,w) uint8 indices -> (h,w,3) uint8 rgb."""
        return self.colors[indices]

    def to_hex(self) -> list[str]:
        return [f"#{int(r):02x}{int(g):02x}{int(b):02x}" for r, g, b in self.colors]

    # -- persistence ----------------------------------------------------- #
    @staticmethod
    def load(path: str | Path) -> "Palette":
        path = Path(path)
        if path.suffix.lower() == ".npy":
            return Palette(np.load(path))
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            colors = data.get("colors")
            if colors is None:
                raise ValueError(f"{path}: missing 'colors'")
        else:
            colors = data
        return Palette(np.asarray(colors, dtype=np.uint8))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".npy":
            np.save(path, self.colors)
        else:
            path.write_text(json.dumps(
                {"n": self.n, "colors": [ [int(x) for x in row] for row in self.colors ]},
                indent=0))

    # -- PIL interop ----------------------------------------------------- #
    def pil_image(self, indices: np.ndarray) -> Image.Image:
        """Palette-indexed array -> 'P' mode PNG-savable image."""
        img = Image.fromarray(np.ascontiguousarray(indices, dtype=np.uint8), mode="P")
        flat = [int(x) for row in self.colors for x in row]
        img.putpalette(flat + [0] * (768 - len(flat)))
        return img

    @staticmethod
    def from_pil(img: Image.Image, n: int) -> "Palette":
        pal = list(img.getpalette()[: n * 3])
        return Palette(np.asarray(pal, dtype=np.uint8).reshape(n, 3))


def make_test_palette(n: int, seed: int = 0) -> Palette:
    """Deterministic spread palette for tests/dev (not the real E5 palette)."""
    rng = np.random.default_rng(seed)
    cols = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        cols[i] = rng.integers(0, 256, 3)
    return Palette(cols)
