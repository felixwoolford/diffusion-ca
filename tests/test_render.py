import numpy as np
import pytest

from applet.engine.grid import Field
from applet.engine.render import render_cross


def tiles_for(pal, distinct=True):
    """Five tiles, each a solid distinct color (indices 2..7)."""
    if distinct:
        return {
            "center": pal.decode(np.full((4, 4), 2, np.uint8)),
            "north": pal.decode(np.full((4, 4), 3, np.uint8)),
            "east": pal.decode(np.full((4, 4), 4, np.uint8)),
            "south": pal.decode(np.full((4, 4), 5, np.uint8)),
            "west": pal.decode(np.full((4, 4), 6, np.uint8)),
        }
    z = lambda i: pal.decode(np.full((4, 4), i, np.uint8))
    return {"center": z(0), "north": z(1), "east": z(0), "south": z(1), "west": z(0)}


def test_composite_layout_and_size(pal):
    rs = 32
    out = render_cross(tiles_for(pal), rs, labels=False)
    assert out.shape == (3 * rs, 3 * rs, 3)
    # center tile red at the middle of the canvas
    assert (out[rs + 5, rs + 5] == [255, 0, 0]).all()
    # north green at top-middle
    assert (out[rs // 2, rs + rs // 2] == [0, 255, 0]).all()
    # west cyan at middle-left
    assert (out[rs + rs // 2, rs // 2] == [0, 255, 255]).all()
    # east blue at middle-right
    assert (out[rs + rs // 2, 2 * rs + rs // 2] == [0, 0, 255]).all()
    # south yellow at bottom-middle
    assert (out[2 * rs + rs // 2, rs + rs // 2] == [255, 255, 0]).all()
    # corners stay black (unused)
    assert (out[0, 0] == 0).all()


def test_nearest_upscale_in_composite(pal):
    # checkerboard art tile, read_size 32 -> each art px is 8x8
    art = np.zeros((4, 4), np.uint8)
    art[::2, ::2] = 2
    art[1::2, 1::2] = 2
    t = tiles_for(pal)
    t["center"] = pal.decode(art)
    out = render_cross(t, 32, labels=False)
    assert (out[32:40, 32:40] == [255, 0, 0]).all()      # art px (0,0) -> 8x8
    assert (out[32:40, 40:48] == [0, 0, 0]).all()        # art px (0,1)
    assert (out[40:48, 32:40] == [0, 0, 0]).all()        # art px (1,0)


def test_labels_present_and_distinct(pal):
    t = tiles_for(pal)
    plain = render_cross(t, 32, labels=False)
    labeled = render_cross(t, 32, labels=True)
    assert (plain != labeled).any()
    # white label pixels exist inside the top-left margin of every tile
    for (row, col) in [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]:
        box = labeled[row * 32:row * 32 + 16, col * 32:col * 32 + 16]
        white = (box > 240).all(-1)
        assert white.any(), f"no label in tile at ({row},{col})"


def test_labels_survive_on_white_tiles(pal):
    # all-white tiles: outline must make the label visible
    t = {"center": None, "north": None, "east": None, "south": None, "west": None}
    white = pal.decode(np.full((4, 4), 1, np.uint8))
    for k in t:
        t[k] = white
    out = render_cross(t, 32, labels=True)
    box = out[32:48, 32:48]  # center label area
    assert (box < 200).any()  # dark outline pixels present


def test_missing_tile_raises(pal):
    t = tiles_for(pal)
    del t["north"]
    with pytest.raises(ValueError, match="missing tiles"):
        render_cross(t, 32)
