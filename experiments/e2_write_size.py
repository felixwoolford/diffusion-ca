"""E2: write-size matrix (SPEC section 6).

Question: {1024, 512, 256} x {t2i, edit_self, edit_neighborhood}:
post-quantize 32-art fidelity, style adherence, latency.
Expectation (SPEC 4): 512 >= 1024 quality-per-second; 256 capped at 16-art.

Test tile (32x32, dev palette):
  - TR quadrant: flat green;  BL quadrant: flat blue          (16px features)
  - TL and BR quadrants: 2x2 checker of 8x8 white/black squares
    (white TL, black TR, black BL, white BR)                  (8px features)
At write 256 (latent 16x16) the 8px checker is half a latent -> expected
to smear; the 16px quadrant layout should survive.

Prompts are canned (no vision rule model needed): E2 isolates what the
diffusion model does at each size; rule quality is E3's job. Fixed seed 42.

Usage:
  uv run python experiments/e2_write_size.py [--server URL] [--no-generate]
Outputs: images_out/e2/<arm>_<size>/ + e2_results.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from applet.engine.comfy import ComfyClient  # noqa: E402
from applet.engine.config import load_config  # noqa: E402
from applet.engine.imageops import block_mode_downscale, nearest_resize  # noqa: E402
from applet.engine.palette import Palette  # noqa: E402
from applet.engine.pipeline import ComfyGenerator, GenRequest  # noqa: E402

SERVERS = ["http://127.0.0.1:8188", "http://127.0.0.1:8189"]
SEED = 42
SIZES = [1024, 512, 256]
ARMS = ["t2i", "edit_self", "edit_neighborhood"]
OUT_ROOT = Path("images_out/e2")

TILE_DESC = ("a 32x32 pixel-art tile with flat colors and sharp 8px/16px block "
             "edges: the top-right quadrant is a flat green 16x16 block; the "
             "bottom-left quadrant is a flat blue 16x16 block; the top-left and "
             "bottom-right quadrants each contain a 2x2 checkerboard of 8x8 "
             "squares, white top-left, black top-right, black bottom-left, "
             "white bottom-right")

PROMPTS = {
    "t2i": ("A 32x32 pixel-art tile: " + TILE_DESC +
            ". Completely flat colors, sharp straight edges, no gradient, no "
            "texture, no shading, no border, no perspective, filling the canvas."),
    "edit_self": ("Keep the reference tile exactly as it is: same layout, same "
                  "flat colors, same 8px/16px block positions, sharp edges. "
                  "The tile is: " + TILE_DESC + "."),
    "edit_neighborhood": ("Keep image 1 (the center tile) exactly as it is: same "
                          "layout, flat colors, 8px/16px blocks, sharp edges. "
                          "Ignore images 2-5 (neutral context). The center tile "
                          "is: " + TILE_DESC + "."),
}


def make_test_tile(pal: Palette) -> tuple[np.ndarray, dict]:
    def idx_of(rgb) -> int:
        return int(pal.nearest(np.array([rgb], dtype=np.uint8))[0])

    GRN = idx_of([0, 190, 0])
    BLU = idx_of([0, 60, 255])
    W = idx_of([255, 255, 255])
    B = idx_of([0, 0, 0])
    tile = np.zeros((32, 32), np.uint8)
    tile[0:16, 16:32] = GRN        # TR quadrant
    tile[16:32, 0:16] = BLU        # BL quadrant
    for (r0, c0) in ((0, 0), (16, 16)):  # TL and BR checkers
        for sr in range(2):
            for sc in range(2):
                sw = W if (sr + sc) % 2 == 0 else B
                tile[r0 + sr * 8:r0 + sr * 8 + 8, c0 + sc * 8:c0 + sc * 8 + 8] = sw
    expected = {"TR": GRN, "BL": BLU}
    checkers = {"TL": W, "BR": W}   # expected color of each checker's TL 8x8 square
    return tile, {"quads": expected, "checker_tl": checkers, "W": W, "B": B,
                  "GRN": GRN, "BLU": BLU}


def mode_of(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    return int(vals[counts.argmax()])


def score(out32: np.ndarray, tile: np.ndarray, exp: dict) -> dict:
    res = {"exact_match": round(float((out32 == tile).mean()), 4)}
    # 16px quadrant survival (TR, BL are the solid ones)
    for name, (r0, c0), want in (("TR", (0, 16), exp["GRN"]), ("BL", (16, 0), exp["BLU"])):
        res[f"quad_{name}_dominant_ok"] = int(mode_of(out32[r0:r0+16, c0:c0+16]) == want)
    # 8px checker survival (TL, BR): per-8x8-square dominant vs expected,
    # plus white/black share of the quadrant
    for name, (r0, c0) in (("TL", (0, 0)), ("BR", (16, 16))):
        q = out32[r0:r0+16, c0:c0+16]
        w_share = float((q == exp["W"]).mean())
        b_share = float((q == exp["B"]).mean())
        squares = []
        for sr in range(2):
            for sc in range(2):
                want = exp["W"] if (sr + sc) % 2 == 0 else exp["B"]
                squares.append(mode_of(q[sr*8:sr*8+8, sc*8:sc*8+8]) == want)
        res[f"checker_{name}_w_share"] = round(w_share, 3)
        res[f"checker_{name}_b_share"] = round(b_share, 3)
        res[f"checker_{name}_squares_ok"] = int(sum(squares))
    return res


def generate_all(client: ComfyClient, url: str, cfg, pal: Palette, tile: np.ndarray) -> None:
    for size in SIZES:
        cfg_s = dataclasses.replace(cfg, write_size=size)
        gen = ComfyGenerator(client, cfg_s, OUT_ROOT)
        tile_w = nearest_resize(pal.decode(tile), size)
        neutral_w = nearest_resize(pal.decode(np.zeros((32, 32), np.uint8)), size)
        for arm in ARMS:
            tag = f"{arm}_{size}"
            out_dir = OUT_ROOT / tag
            full = out_dir / "out.png"
            out_dir.mkdir(parents=True, exist_ok=True)
            if full.exists():
                print(f"[gen] {tag}: reusing {full}")
                continue
            images = {}
            if arm == "edit_self":
                images = {"image_1": tile_w}
            elif arm == "edit_neighborhood":
                images = {"image_1": tile_w, "image_2": neutral_w, "image_3": neutral_w,
                          "image_4": neutral_w, "image_5": neutral_w}
            print(f"[gen] {tag}: {size}x{size}, seed {SEED} ...")
            t0 = time.time()
            img = gen.generate(GenRequest(arm=arm, prompt=PROMPTS[arm], seed=SEED,
                                          cell=(0, 0), step=0, images=images))
            dt = time.time() - t0
            assert img.shape[:2] == (size, size), f"{tag}: got {img.shape}"
            Image.fromarray(img, mode="RGB").save(full)
            Image.fromarray(img, mode="RGB").resize((256, 256), Image.NEAREST).save(
                out_dir / "preview256.png")
            print(f"[gen] {tag}: {dt:.1f}s")
            (out_dir / "latency.json").write_text(json.dumps({"s": round(dt, 2)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=None)
    ap.add_argument("--no-generate", action="store_true")
    args = ap.parse_args()

    pal = Palette.load("config/palette_dev64.json")
    tile, exp = make_test_tile(pal)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pal.decode(tile), mode="RGB").save(OUT_ROOT / "test_tile_32.png")
    Image.fromarray(pal.decode(tile), mode="RGB").resize((256, 256), Image.NEAREST).save(
        OUT_ROOT / "test_tile_256.png")

    client = url = None
    if not args.no_generate:
        for cand in ([args.server] if args.server else SERVERS):
            try:
                c = ComfyClient(cand)
                c.models()
            except Exception as e:  # noqa: BLE001
                print(f"[server] {cand} down: {e}")
                continue
            client, url = c, cand
            break
        if client is None:
            sys.exit("no ComfyUI server available")
        print(f"[server] {url} — {client.models()}")

    cfg_yaml = OUT_ROOT / "e2.config.yaml"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfg_yaml.write_text("""\
grid_n: 4
art_size: 32
palette_n: 64
palette_path: config/palette_dev64.json
write_size: 1024
read_size: 512
arm: t2i
schedule: random_uniform
K: 1
seed_mode: deterministic
seed: 42
rule_backend: template
generator: comfy
comfy_url: %s
state_dir: images_out/e2/state
run_name: e2
concurrency: 1
""" % (url or SERVERS[1]))
    cfg = load_config(cfg_yaml)
    if client is not None:
        generate_all(client, url, cfg, pal, tile)

    results = {}
    for size in SIZES:
        for arm in ARMS:
            out_dir = OUT_ROOT / f"{arm}_{size}"
            full = out_dir / "out.png"
            if not full.exists():
                results[f"{arm}_{size}"] = {"skipped": True}
                print(f"[skip] {arm}_{size}: no output (server refused or missing)")
                continue
            img = np.asarray(Image.open(full).convert("RGB"))
            out32 = block_mode_downscale(img, 32, pal)
            Image.fromarray(pal.decode(out32), mode="RGB").save(out_dir / "quantized_32.png")
            Image.fromarray(pal.decode(out32), mode="RGB").resize(
                (256, 256), Image.NEAREST).save(out_dir / "quantized_256.png")
            entry = score(out32, tile, exp)
            entry["latency_s"] = json.loads((out_dir / "latency.json").read_text())["s"] \
                if (out_dir / "latency.json").exists() else None
            results[f"{arm}_{size}"] = entry
            print(json.dumps(entry))

    (OUT_ROOT / "e2_results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] -> {OUT_ROOT / 'e2_results.json'}")


if __name__ == "__main__":
    main()
