import numpy as np
import pytest

from applet.engine.palette import Palette, make_test_palette


def test_nearest_exact_and_nearest(pal):
    rgb = pal.decode(np.array([[0, 2, 4], [5, 7, 1]]))
    assert (pal.nearest(rgb) == np.array([[0, 2, 4], [5, 7, 1]])).all()
    # (127, 0, 0) is nearest to black (0) over red (2);
    # (200, 10, 10) is nearest to red (2)
    n = pal.nearest(np.array([[127, 0, 0], [200, 10, 10]], dtype=np.uint8))
    assert n.tolist() == [0, 2]


def test_decode(pal):
    idx = np.array([[0, 2], [3, 4]])
    rgb = pal.decode(idx)
    assert rgb[0, 1].tolist() == [255, 0, 0]
    assert rgb[1, 0].tolist() == [0, 255, 0]


def test_roundtrip_json(pal, tmp_path):
    f = tmp_path / "p.json"
    pal.save(f)
    back = Palette.load(f)
    assert (back.colors == pal.colors).all()
    assert back.n == 8


def test_roundtrip_npy(pal, tmp_path):
    f = tmp_path / "p.npy"
    pal.save(f)
    back = Palette.load(f)
    assert (back.colors == pal.colors).all()


def test_pil_roundtrip(pal):
    idx = np.random.default_rng(0).integers(0, 8, (4, 4), dtype=np.uint8)
    img = pal.pil_image(idx)
    assert img.mode == "P"
    assert (np.asarray(img) == idx).all()
    back = Palette.from_pil(img, 8)
    assert (back.colors == pal.colors).all()


def test_to_hex(pal):
    assert pal.to_hex()[2] == "#ff0000"
    assert len(pal.to_hex()) == 8


def test_make_test_palette_deterministic():
    a, b = make_test_palette(16, seed=3), make_test_palette(16, seed=3)
    assert (a.colors == b.colors).all()


def test_bad_shape():
    with pytest.raises(ValueError):
        Palette(np.zeros((3,), dtype=np.uint8))
