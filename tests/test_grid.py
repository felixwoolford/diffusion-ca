import numpy as np
import pytest
from PIL import Image

from applet.engine.grid import Field


def test_wrap_neighborhood(pal):
    f = Field(2, 4, pal)
    f.set_cell(0, 0, np.full((4, 4), 2, np.uint8))   # red at (0,0)
    f.set_cell(1, 0, np.full((4, 4), 3, np.uint8))   # green at (1,0)
    f.set_cell(0, 1, np.full((4, 4), 4, np.uint8))   # blue at (0,1)
    nb = f.neighborhood(0, 0)
    assert (nb["center"][0, 0] == [255, 0, 0]).all()
    assert (nb["north"][0, 0] == [0, 255, 0]).all()  # row 0 wraps to row 1
    assert (nb["west"][0, 0] == [0, 0, 255]).all()   # col 0 wraps to col 1
    assert (nb["south"][0, 0] == [0, 255, 0]).all()
    assert (nb["east"][0, 0] == [0, 0, 255]).all()


def test_snapshot_independent(pal):
    f = Field(2, 4, pal)
    s = f.snapshot()
    f.set_cell(0, 0, np.full((4, 4), 2, np.uint8))
    assert (s.cell_index(0, 0) == 0).all()
    assert (f.cell_index(0, 0) == 2).all()


def test_set_cell_changed_flag(pal):
    f = Field(2, 4, pal)
    tile = np.full((4, 4), 5, np.uint8)
    assert f.set_cell(1, 1, tile) is True
    assert f.set_cell(1, 1, tile) is False


def test_set_cell_bad_shape(pal):
    f = Field(2, 4, pal)
    with pytest.raises(ValueError):
        f.set_cell(0, 0, np.zeros((3, 3), np.uint8))


def test_set_cell_bad_index(pal):
    f = Field(2, 4, pal)
    with pytest.raises(ValueError, match="out of range"):
        f.set_cell(0, 0, np.full((4, 4), 8, np.uint8))


def test_from_png_tiles_and_quantizes(pal, tmp_path):
    # 2x2 field, art=4 -> 8x8 png; quadrants are solid cell colors
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[0:4, 0:4] = pal.colors[2]  # cell (0,0) red
    rgb[0:4, 4:8] = pal.colors[3]  # cell (0,1) green
    rgb[4:8, 0:4] = pal.colors[4]  # cell (1,0) blue
    rgb[4:8, 4:8] = pal.colors[5]  # cell (1,1) yellow
    p = tmp_path / "field.png"
    Image.fromarray(rgb).save(p)
    f = Field.from_png(p, 2, 4, pal)
    assert f.cell_index(0, 0)[0, 0] == 2
    assert f.cell_index(0, 1)[0, 0] == 3
    assert f.cell_index(1, 0)[0, 0] == 4
    assert f.cell_index(1, 1)[0, 0] == 5


def test_from_png_wrong_size(pal, tmp_path):
    p = tmp_path / "bad.png"
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(p)
    with pytest.raises(ValueError, match="expected"):
        Field.from_png(p, 2, 4, pal)


def test_render_sizes(pal):
    f = Field(2, 4, pal)
    assert f.to_art_image().shape == (8, 8, 3)
    assert f.to_display_image(2).shape == (16, 16, 3)
    assert f.to_display_image(3).shape == (24, 24, 3)


def test_snapshot_roundtrip(pal, tmp_path):
    f = Field.random(2, 4, pal, seed=1)
    p = tmp_path / "s.npy"
    f.save_snapshot(p)
    g = Field.load_snapshot(p, pal)
    assert (g.indices == f.indices).all()
