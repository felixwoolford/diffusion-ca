import numpy as np
import pytest

from applet.engine.imageops import block_mode_downscale, nearest_resize, upscale_to_write


def test_nearest_resize_squares_and_keeps_colors(pal):
    src = pal.decode(np.array([[0, 2], [4, 6]]))  # 2x2
    out = nearest_resize(src, 8)
    assert out.shape == (8, 8, 3)
    assert (out[0, 0] == [0, 0, 0]).all()
    assert (out[0, 7] == [255, 0, 0]).all()      # quadrant 0,1
    assert (out[7, 0] == [0, 0, 255]).all()      # quadrant 1,0
    assert (out[7, 7] == [0, 255, 255]).all()    # quadrant 1,1
    # exactly 4 distinct colors, each a solid 4x4 quadrant
    assert len(np.unique(out.reshape(-1, 3), axis=0)) == 4


def test_nearest_resize_identity(pal):
    src = pal.decode(np.zeros((32, 32), dtype=np.uint8))
    assert (nearest_resize(src, 32) == src).all()


def test_block_mode_majority_wins(pal):
    # write=16, art=8 -> 2x2 px blocks. Block (0,0): 3 red + 1 blue -> red (2)
    write = pal.decode(np.zeros((16, 16), dtype=np.uint8))
    write[0:2, 0:2] = [255, 0, 0]
    write[0, 1] = [0, 0, 255]
    idx = block_mode_downscale(write, 8, pal)
    assert idx.shape == (8, 8)
    assert idx[0, 0] == 2


def test_block_mode_median_fallback_not_mean(pal2):
    # 2x2 block, four distinct colors (tie -> median fallback).
    # median R = 105 -> M (index 4); mean R = 121.25 -> would be N (index 5).
    # If the code ever used a plain mean, the result would be index 5.
    write = np.zeros((2, 2, 3), dtype=np.uint8)
    write[0, 0] = [0, 0, 0]
    write[0, 1] = [10, 0, 0]
    write[1, 0] = [200, 0, 0]
    write[1, 1] = [255, 0, 0]
    idx = block_mode_downscale(write, 1, pal2)
    assert idx.shape == (1, 1)
    assert idx[0, 0] == 4  # M (105,0,0): the median, not the mean


def test_roundtrip_art_write_art(pal):
    art = np.random.default_rng(0).integers(0, 8, (8, 8), dtype=np.uint8)
    up = upscale_to_write(pal.decode(art), 32)
    down = block_mode_downscale(up, 8, pal)
    assert (down == art).all()


def test_block_mode_rejects_bad_size(pal):
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="divisible"):
        block_mode_downscale(rgb, 4, pal)


@pytest.fixture
def pal2():
    from applet.engine.palette import Palette
    # 0 A, 1 B, 2 C, 3 D, 4 M=median, 5 N=mean
    return Palette(np.array(
        [[0, 0, 0], [10, 0, 0], [200, 0, 0], [255, 0, 0], [105, 0, 0], [121, 0, 0]],
        dtype=np.uint8))
