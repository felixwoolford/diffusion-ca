"""Neighborhood render for the rule LLM (SPEC §2.6).

The 5 tiles are nearest-upscaled to read_size and composed as a labeled
cross layout. Labels are drawn *into* the composite so the LLM can't
misassign positions:

        [ N ]
  [ W ] [ C ] [ E ]
        [ S ]

Per-tile visual-token cost at read_size (Qwen3.8-27B: 32x32 px/token):
256 -> 64, 512 -> 256, 1024 -> 1024. A 512 composite ~ 1280 visual tokens.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .imageops import nearest_resize

POS = {  # name -> (row, col) in the 3x3 cross layout
    "north": (0, 1),
    "west": (1, 0),
    "center": (1, 1),
    "east": (1, 2),
    "south": (2, 1),
}
LABELS = {"north": "N", "west": "W", "center": "C", "east": "E", "south": "S"}


def _letter_bitmap(ch: str, height: int) -> np.ndarray:
    """Render a letter as a chunky monochrome bitmap (alpha), nearest-scaled
    from PIL's default font — no font files needed, pixel-art appropriate."""
    font = ImageFont.load_default()
    pad = 4
    tmp = Image.new("L", (height + 2 * pad, height + 2 * pad), 0)
    d = ImageDraw.Draw(tmp)
    d.text((pad, pad), ch, font=font, fill=255)
    arr = np.asarray(tmp)
    ys, xs = np.nonzero(arr)
    if len(xs) == 0:  # empty glyph; draw a square instead
        arr = np.zeros_like(arr)
        arr[pad:pad + 6, pad:pad + 6] = 255
    else:
        arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    img = Image.fromarray(arr).resize((max(1, int(arr.shape[1] * height / arr.shape[0])), height),
                                      Image.Resampling.NEAREST)
    return np.asarray(img)


def render_cross(tiles: dict[str, np.ndarray], read_size: int,
                 labels: bool = True,
                 label_color: tuple[int, int, int] = (255, 255, 255),
                 label_outline: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """5 tiles (art,art,3) -> (3*read_size, 3*read_size, 3) uint8 composite."""
    missing = set(POS) - set(tiles)
    if missing:
        raise ValueError(f"missing tiles: {sorted(missing)}")
    canvas = np.zeros((3 * read_size, 3 * read_size, 3), dtype=np.uint8)
    for name, (row, col) in POS.items():
        t = nearest_resize(tiles[name], read_size)
        canvas[row * read_size:(row + 1) * read_size, col * read_size:(col + 1) * read_size] = t
    if labels:
        img = Image.fromarray(canvas, mode="RGB")
        d = ImageDraw.Draw(img)
        m = max(4, read_size // 32)          # margin
        lh = max(16, read_size // 8)         # label height
        for name, (row, col) in POS.items():
            glyph = _letter_bitmap(LABELS[name], lh)
            gy, gx = np.nonzero(glyph)
            if len(gx) == 0:
                continue
            h, w = glyph.shape
            x0, y0 = col * read_size + m, row * read_size + m
            # outline: dark copy offset by ~lh/8, then the label on top
            off = max(1, lh // 8)
            oarr = np.zeros((h, w, 4), dtype=np.uint8)
            oarr[gy, gx] = (*label_outline, 255)
            img.paste(Image.fromarray(oarr, mode="RGBA"), (x0 + off, y0 + off))
            larr = np.zeros((h, w, 4), dtype=np.uint8)
            larr[gy, gx] = (*label_color, 255)
            img.paste(Image.fromarray(larr, mode="RGBA"), (x0, y0))
        canvas = np.asarray(img)
    return canvas
