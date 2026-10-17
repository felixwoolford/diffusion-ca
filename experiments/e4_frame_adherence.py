"""E4: pixel-art frame adherence (SPEC section 6).

Question: does the model actually emit blocky grid-aligned art at each
write size when asked with the pixel-art frame prompt? Measure
blockiness BEFORE quantize (E2 measured after).

Method (t2i arm, same test-tile description as E2, seed 42 + 7):
  For each output at write size W (latent grid g = W/16):
  * edge alignment: where the color transitions land vs the expected
    block boundaries (256/128 px at W=1024, etc.), and the residual to
    the latent grid (mod g).
  * block flatness: per-pixel RGB std inside each expected block
    (flat pixel-art block -> ~0; gradients/smoothing -> high).
  * quantize loss: per-pixel RGB distance from the output to its
    nearest dev-palette color (flat palette art -> low; gradients -> high).

Usage:
  uv run python experiments/e4_frame_adherence.py [--server URL] [--no-generate]
Outputs: images_out/e4/<size>_seed<seed>/ + e4_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from applet.engine.comfy import ComfyClient  # noqa: E402
from applet.engine.config import load_config  # noqa: E402
from applet.engine.imageops import nearest_resize  # noqa: E402
from applet.engine.palette import Palette  # noqa: E402
from applet.engine.pipeline import ComfyGenerator, GenRequest  # noqa: E402
from experiments.e1_alignment import row_crossings  # noqa: E402
from experiments.e2_write_size import PROMPTS, make_test_tile  # noqa: E402

SERVERS = ["http://127.0.0.1:8188", "http://127.0.0.1:8189"]
SEEDS = [42, 7]
SIZES = [1024, 512, 256]
OUT_ROOT = Path("images_out/e4")
PROMPT = PROMPTS["t2i"]  # full-description prompt with the pixel-art frame

# expected interior block boundaries per write size (art=32):
# checker squares 8 art-px, quadrants 16 art-px -> lines at 8/16/24 art-px
def expected_boundaries(W: int) -> list[int]:
    s = W / 32  # art-pixel scale in px
    return [int(round(v * s)) for v in (8, 16, 24)]


def strong_edges(arr: np.ndarray, vertical: bool, vote_threshold: float) -> list[float]:
    """Positions of strong luminance steps along rows (vertical edges)
    or columns (horizontal edges), voted across lines and clustered."""
    from collections import Counter
    lum = arr.astype(np.float32).mean(axis=-1)
    votes: Counter = Counter()
    if vertical:
        lines = np.abs(np.diff(lum, axis=1))  # per row: change along x
    else:
        lines = np.abs(np.diff(lum, axis=0)).T  # per column: change along y
    for line in lines:
        thr = max(16.0, 0.08 * line.max())
        for x in np.nonzero(line > thr)[0]:
            votes[int(x)] += 1
    clusters: list[list[tuple[int, int]]] = []
    for x in sorted(votes):
        if clusters and x - clusters[-1][-1][0] <= 6:
            clusters[-1].append((x, votes[x]))
        else:
            clusters.append([(x, votes[x])])
    out = []
    for cl in clusters:
        tot = sum(v for _, v in cl)
        if tot >= vote_threshold:
            out.append(sum(p * v for p, v in cl) / tot)
    return sorted(out)


def analyze(tag: str, W: int, img: np.ndarray, pal: Palette) -> dict:
    g = W / 16  # latent grid spacing in px
    exp = expected_boundaries(W)

    # -- block flatness: per-pixel RGB std inside each 8-art-px block (4x4)
    s = W // 32        # px per art pixel
    bs = 8 * s         # px per 8-art-px block
    blocks = img.reshape(4, bs, 4, bs, 3).transpose(0, 2, 1, 3, 4)  # (4,4,bs,bs,3)
    block_std = blocks.reshape(16, bs * bs, 3).std(axis=1).mean(axis=1)
    # -- quantize loss vs dev palette (max-channel distance to nearest color)
    idx = pal.nearest(img)
    qloss = np.abs(pal.colors[idx].astype(np.int32) - img.astype(np.int32)).max(axis=-1)
    # -- edge alignment
    edge_v = strong_edges(img, vertical=True, vote_threshold=W * 0.4)
    edge_h = strong_edges(img, vertical=False, vote_threshold=W * 0.4)

    def align(edges: list[float]) -> list[dict]:
        rows = []
        for e in edges:
            rows.append({"edge": round(float(e), 1),
                         "res_latent": round(min(e % g, g - e % g), 2),
                         "nearest_expected": min(exp, key=lambda x: abs(x - e)),
                         "delta_expected": round(float(e - min(exp, key=lambda x: abs(x - e))), 1)})
        return rows

    return {
        "tag": tag, "size": W, "latent_grid": g,
        "block_std_per_block": [round(float(x), 2) for x in block_std],
        "block_std_mean": round(float(block_std.mean()), 2),
        "block_std_max": round(float(block_std.max()), 2),
        "flat_block_frac(std<8)": round(float((block_std < 8).mean()), 3),
        "quantize_loss_mean": round(float(qloss.mean()), 2),
        "quantize_loss_p95": round(float(np.percentile(qloss, 95)), 2),
        "palette_frac(<=16)": round(float((qloss <= 16).mean()), 4),
        "edges_v": align(edge_v),
        "edges_h": align(edge_h),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=None)
    ap.add_argument("--no-generate", action="store_true")
    args = ap.parse_args()

    pal = Palette.load("config/palette_dev64.json")
    make_test_tile(pal)  # sanity
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

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
        cfg_yaml = OUT_ROOT / "e4.config.yaml"
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
state_dir: images_out/e4/state
run_name: e4
concurrency: 1
""" % url)
        import dataclasses
        cfg0 = load_config(cfg_yaml)
        for W in SIZES:
            cfg = dataclasses.replace(cfg0, write_size=W)
            gen = ComfyGenerator(client, cfg, OUT_ROOT)
            for seed in SEEDS:
                d = OUT_ROOT / f"{W}_seed{seed}"
                d.mkdir(parents=True, exist_ok=True)
                out = d / "out.png"
                if out.exists():
                    print(f"[gen] {W}_seed{seed}: reusing {out}")
                    continue
                print(f"[gen] {W}_seed{seed} ...")
                t0 = time.time()
                img = gen.generate(GenRequest(arm="t2i", prompt=PROMPT, seed=seed,
                                              cell=(0, 0), step=0, images={}))
                dt = time.time() - t0
                Image.fromarray(img, mode="RGB").save(out)
                Image.fromarray(img, mode="RGB").resize((256, 256), Image.NEAREST).save(
                    d / "preview256.png")
                (d / "latency.json").write_text(json.dumps({"s": round(dt, 2)}))
                print(f"[gen] {W}_seed{seed}: {dt:.1f}s")

    results = []
    for W in SIZES:
        for seed in SEEDS:
            p = OUT_ROOT / f"{W}_seed{seed}" / "out.png"
            if not p.exists():
                continue
            img = np.asarray(Image.open(p).convert("RGB"))
            res = analyze(f"{W}_seed{seed}", W, img, pal)
            results.append(res)
            slim = {k: v for k, v in res.items() if k not in ("edges_v", "edges_h")}
            print(json.dumps(slim))

    (OUT_ROOT / "e4_results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] -> {OUT_ROOT / 'e4_results.json'}")


if __name__ == "__main__":
    main()
