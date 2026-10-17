"""The CA field: an N x N grid of art_size x art_size tiles, palette-indexed.

The index array **is** the ground truth (SPEC §1): (n, n, art, art) uint8
values in [0, palette_n). Borders wrap (torus topology); the 4-neighborhood
is N/E/S/W + self.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .imageops import nearest_resize
from .palette import Palette


class Field:
    def __init__(self, n: int, art: int, palette: Palette,
                 indices: np.ndarray | None = None):
        self.n = n
        self.art = art
        self.palette = palette
        if indices is None:
            indices = np.zeros((n, n, art, art), dtype=np.uint8)
        if indices.shape != (n, n, art, art):
            raise ValueError(f"indices shape {indices.shape} != ({n},{n},{art},{art})")
        if indices.max() >= palette.n:
            raise ValueError(f"palette index {indices.max()} out of range for palette of {palette.n}")
        self.indices = indices.astype(np.uint8)

    # -- construction ---------------------------------------------------- #
    @classmethod
    def random(cls, n: int, art: int, palette: Palette, seed: int | None = None) -> "Field":
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, palette.n, size=(n, n, art, art), dtype=np.uint8)
        return cls(n, art, palette, idx)

    @classmethod
    def from_png(cls, path: str | Path, n: int, art: int, palette: Palette) -> "Field":
        """A square PNG of (n*art) x (n*art) px: tile it into the field and quantize.

        Used for the initial field (SPEC §8.4: one coherent generation over the
        whole art field, tiled) and for test fixtures.
        """
        img = Image.open(path).convert("RGB")
        side = n * art
        if img.size != (side, side):
            raise ValueError(f"{path}: expected {side}x{side} px, got {img.size[0]}x{img.size[1]}")
        rgb = np.asarray(img)
        idx = palette.nearest(rgb).reshape(n, art, n, art).transpose(0, 2, 1, 3)
        return cls(n, art, palette, idx.astype(np.uint8))

    # -- access ---------------------------------------------------------- #
    def cell(self, r: int, c: int) -> np.ndarray:
        """Cell (r,c) as (art,art,3) uint8 rgb."""
        return self.palette.decode(self.indices[r, c])

    def cell_index(self, r: int, c: int) -> np.ndarray:
        return self.indices[r, c]

    def set_cell(self, r: int, c: int, new_index: np.ndarray) -> bool:
        """Write a palette-indexed tile. Returns True if the cell changed."""
        new_index = np.asarray(new_index, dtype=np.uint8)
        if new_index.shape != (self.art, self.art):
            raise ValueError(f"tile shape {new_index.shape} != ({self.art},{self.art})")
        if new_index.max() >= self.palette.n:
            raise ValueError("palette index out of range")
        changed = not np.array_equal(self.indices[r, c], new_index)
        self.indices[r, c] = new_index
        return changed

    def snapshot(self) -> "Field":
        """Frozen copy of the whole field (SPEC §1.6: snapshot per step)."""
        return Field(self.n, self.art, self.palette, self.indices.copy())

    def neighborhood(self, r: int, c: int) -> dict[str, np.ndarray]:
        """5 tiles (art,art,3) rgb for (r,c); borders wrap (torus)."""
        a = self.indices
        return {
            "center": self.palette.decode(a[r, c]),
            "north": self.palette.decode(a[(r - 1) % self.n, c]),
            "east": self.palette.decode(a[r, (c + 1) % self.n]),
            "south": self.palette.decode(a[(r + 1) % self.n, c]),
            "west": self.palette.decode(a[r, (c - 1) % self.n]),
        }

    # -- rendering ------------------------------------------------------- #
    def to_art_image(self) -> np.ndarray:
        """Whole art field: (n*art, n*art, 3) uint8."""
        return self.palette.decode(self.indices.reshape(self.n * self.art, self.n * self.art))

    def to_display_image(self, render_scale: int) -> np.ndarray:
        """Display field: art field nearest-upscaled by render_scale."""
        return nearest_resize(self.to_art_image(), self.n * self.art * render_scale)

    def save_snapshot(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.indices)

    @classmethod
    def load_snapshot(cls, path: str | Path, palette: Palette) -> "Field":
        idx = np.load(path)
        n = idx.shape[0]
        art = idx.shape[2]
        return cls(n, art, palette, idx)
