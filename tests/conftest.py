import numpy as np
import pytest

from applet.engine.palette import Palette

# 8 well-separated colors; indices: 0 black, 1 white, 2 red, 3 green,
# 4 blue, 5 yellow, 6 cyan, 7 magenta
STANDARD = np.array(
    [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0],
     [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]],
    dtype=np.uint8)


@pytest.fixture
def pal() -> Palette:
    return Palette(STANDARD.copy())


@pytest.fixture
def pal_path(tmp_path):
    p = Palette(STANDARD.copy())
    f = tmp_path / "pal.json"
    p.save(f)
    return f
