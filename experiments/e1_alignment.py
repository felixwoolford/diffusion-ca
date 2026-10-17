"""E1: VAE alignment probe (SPEC section 6).

Question: a checkerboard prompt at 1024^2 — do the block edges land on
16-px multiples? This confirms the Qwen-Image-2.1 VAE 16x spatial
compression factor (SPEC section 4) and the write-size alignment argument.

Method:
  * t2i arm, 1024x1024, fixed seed, flat two-color checkerboard prompts
    (4x4 grid -> 256px blocks; 8x8 grid -> 128px blocks).
  * Edge detection per row (for vertical edges) and per column (for
    horizontal edges): sign of (R vs B) per pixel, ambiguous pixels
    forward-filled, sign transitions interpolated to sub-pixel. Every row
    crosses every vertical edge exactly once, so a true edge accumulates
    ~size votes in the histogram; noise crossings do not.
  * For each detected edge: position mod 16 (residual to the 16px grid)
    and delta to the nearest expected edge position.

Usage:
  uv run python experiments/e1_alignment.py [--server URL] [--no-generate]
Outputs: images_out/e1/* (full-size + 256px downscales) + e1_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from applet.engine.comfy import ComfyClient  # noqa: E402
from applet.engine.config import load_config  # noqa: E402
from applet.engine.pipeline import ComfyGenerator, GenRequest  # noqa: E402

SERVERS = ["http://127.0.0.1:8188", "http://127.0.0.1:8189"]
SEED = 42
OUT_ROOT = Path("images_out/e1")

PROMPTS = {
    "c4": ("A perfect 4x4 checkerboard of flat solid squares, alternating pure "
           "red and pure blue, filling the entire canvas edge to edge. 2D pixel "
           "art style, completely flat colors, sharp straight edges, no gradient, "
           "no texture, no shading, no border, no anti-aliasing, no perspective."),
    "c8": ("A perfect 8x8 checkerboard of flat solid squares, alternating pure "
           "red and pure blue, filling the entire canvas edge to edge. 2D pixel "
           "art style, completely flat colors, sharp straight edges, no gradient, "
           "no texture, no shading, no border, no anti-aliasing, no perspective."),
}
# expected interior edge positions (px) for a 1024px canvas
EXPECTED = {"c4": [256, 512, 768], "c8": [128, 256, 384, 512, 640, 768, 896]}


def row_crossings(line: np.ndarray) -> list[float]:
    """Sub-pixel x positions where the R-vs-B sign of a 1-D line flips.

    line: (+1 red-dominant, -1 blue-dominant, 0 ambiguous). The edge is
    placed at the center of the ambiguous transition zone between two
    opposite-sign runs (the VAE blurs each boundary over a few px).
    """
    n = len(line)
    edges: list[float] = []
    i = 0
    while i < n:
        if line[i] == 0:
            i += 1
            continue
        s = line[i]
        i1 = i
        while i1 < n and line[i1] == s:
            i1 += 1
        j = i1
        while j < n and line[j] == 0:
            j += 1
        if j >= n:
            break
        if line[j] == -s:          # real transition over zone [i1, j)
            edges.append(i1 + (j - i1) / 2.0)
        i = j
    return edges


def detect(delta: np.ndarray, size: int, vertical: bool, vote_threshold: float) -> list[float]:
    """Crossing votes per row (vertical edges) or column (horizontal edges).

    Clusters votes within 6 px of each other; keeps clusters whose total
    votes reach the threshold; reports the vote-weighted centroid.
    """
    votes: Counter = Counter()
    lines = delta if vertical else delta.T  # rows / columns as 1-D lines
    for line in lines:
        sign = np.where(line > 0, 1, np.where(line < 0, -1, 0))
        for e in row_crossings(sign):
            votes[round(e)] += 1
    clusters: list[list[tuple[int, int]]] = []  # (position, votes)
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


def analyze(img: np.ndarray, tag: str) -> dict:
    size = img.shape[0]
    r = img[..., 0].astype(np.int32)
    b = img[..., 2].astype(np.int32)
    delta = r - b  # >0 red-dominant, <0 blue-dominant
    v_edges = detect(delta, size, vertical=True, vote_threshold=size * 0.5)
    h_edges = detect(delta, size, vertical=False, vote_threshold=size * 0.5)
    exp = EXPECTED[tag]

    def report(edges: list[float]) -> list[dict]:
        out = []
        for e in edges:
            mod16 = e % 16
            res = min(mod16, 16 - mod16)
            nearest = min(exp, key=lambda x: abs(x - e))
            out.append({"edge_px": round(float(e), 1),
                        "mod16_residual": round(res, 2),
                        "nearest_expected": nearest,
                        "delta_expected_px": round(e - nearest, 1)})
        return out

    return {"tag": tag, "size": size,
            "vertical": report(v_edges),
            "horizontal": report(h_edges)}


def generate(client: ComfyClient, url: str) -> None:
    cfg_yaml = OUT_ROOT / "e1.config.yaml"
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
state_dir: images_out/e1/state
run_name: e1
concurrency: 1
""" % url)
    cfg = load_config(cfg_yaml)
    gen = ComfyGenerator(client, cfg, OUT_ROOT)
    for tag, prompt in PROMPTS.items():
        out = OUT_ROOT / f"{tag}_1024.png"
        if out.exists():
            print(f"[gen] {tag}: reusing {out}")
            continue
        print(f"[gen] {tag}: 1024x1024 t2i, seed {SEED} ...")
        img = gen.generate(GenRequest(arm="t2i", prompt=prompt, seed=SEED,
                                      cell=(0, 0), step=0, images={}))
        assert img.shape[:2] == (1024, 1024), f"unexpected size {img.shape}"
        Image.fromarray(img, mode="RGB").save(out)
        Image.fromarray(img, mode="RGB").resize((256, 256), Image.NEAREST).save(
            OUT_ROOT / f"{tag}_256.png")
        print(f"[gen] {tag}: saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=None, help="ComfyUI URL (default: first live of %s)" % SERVERS)
    ap.add_argument("--no-generate", action="store_true", help="analyze existing images_out/e1/*_1024.png")
    args = ap.parse_args()

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
        generate(client, url)

    results = []
    for tag in PROMPTS:
        p = OUT_ROOT / f"{tag}_1024.png"
        if not p.exists():
            print(f"[skip] {tag}: no image at {p}")
            continue
        img = np.asarray(Image.open(p).convert("RGB"))
        res = analyze(img, tag)
        results.append(res)
        print(json.dumps(res, indent=2))

    (OUT_ROOT / "e1_results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] results -> {OUT_ROOT / 'e1_results.json'}")


if __name__ == "__main__":
    main()
