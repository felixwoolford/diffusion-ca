"""E3: read-size matrix (SPEC section 6).

Question: at neighborhood-render read sizes {256, 512, 1024}, does the rule
model correctly identify the tiles and their positions? The rule model
answers FIXED probe questions about known 5-tile composites — exactly the
labeled cross renders the pipeline would show it (render_cross, labels on).

Design (kept minimal — the multimodal deployment is slower and has a
smaller context; Felix 2026-10-17):
  - main composite  at read sizes 256 / 512 / 1024  (3 calls)
  - alt composite   at read size 512 only           (1 call, fluke check)
  - 7 short questions per call, numbered one-line answers, temp 0.1.

Tiles are dev64-palette art (32x32), flat, unambiguous colors:
  main: C=red N=blue E=yellow S=green, W=quadrant split red/white
  alt:  C=green N=yellow E=red S=blue, W=horizontal split white/green
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from applet.engine.imageops import nearest_resize
from applet.engine.palette import Palette
from applet.engine.render import render_cross

ART = 32
# dev64 indices (verified 2026-10-17): unambiguous hues
RED, BLUE, YELLOW, GREEN, WHITE = 58, 3, 22, 39, 12

OUT_ROOT = Path("images_out/e3")

SYSTEM = (
    "You are a precise visual question-answering model. You are shown one "
    "image: a cross layout of five square pixel-art tiles on a black "
    "background, arranged as N (top), W (left), C (center), E (right), "
    "S (bottom). A letter label is drawn in the top-left corner of each "
    "tile. Answer the numbered questions about the tiles. Read each tile's "
    "position from its letter label, not from its screen position."
)

SEPARATE_SYSTEM = (
    "You are a precise visual question-answering model. You are shown five "
    "SEPARATE square pixel-art tiles, one per image, in the fixed order "
    "given in the text (image 1 = C center, image 2 = N north, image 3 = E "
    "east, image 4 = S south, image 5 = W west). Answer the numbered "
    "questions about the tiles, using those image numbers."
)


FORMAT = (
    "Answer each question on its own line, starting with the question "
    "number followed by a period and a space. Use the shortest answer "
    "that fully answers the question (a color word, a letter, or "
    "yes/no). No other text."
)


def make_tiles(pal: Palette, kind: str) -> dict[str, np.ndarray]:
    def solid(idx: int) -> np.ndarray:
        return pal.decode(np.full((ART, ART), idx, np.uint8))

    if kind == "main":
        w = np.zeros((ART, ART), np.uint8)
        w[:16, :16] = RED
        w[:16, 16:] = WHITE
        w[16:, :16] = WHITE
        w[16:, 16:] = RED
        return {"center": solid(RED), "north": solid(BLUE), "east": solid(YELLOW),
                "south": solid(GREEN), "west": pal.decode(w)}
    # alt
    w = np.zeros((ART, ART), np.uint8)
    w[:16, :] = WHITE
    w[16:, :] = GREEN
    return {"center": solid(GREEN), "north": solid(YELLOW), "east": solid(RED),
            "south": solid(BLUE), "west": pal.decode(w)}


# (question, kind, expected) — kind: color / colors2 / pos / yesno
QUESTIONS = {
    "main": [
        ("What is the dominant color of tile C?", "color", "red"),
        ("What is the dominant color of tile N?", "color", "blue"),
        ("What is the dominant color of tile E?", "color", "yellow"),
        ("What is the dominant color of tile S?", "color", "green"),
        ("Tile W is split into four quadrants of two colors. Name both colors.",
         "colors2", ("red", "white")),
        ("Which tile is yellow? Answer with the letter label (N, E, S, W or C).",
         "pos", "e"),
        ("Are tiles N and S the same color? Answer yes or no.", "yesno", "no"),
    ],
    "alt": [
        ("What is the dominant color of tile C?", "color", "green"),
        ("What is the dominant color of tile N?", "color", "yellow"),
        ("What is the dominant color of tile E?", "color", "red"),
        ("What is the dominant color of tile S?", "color", "blue"),
        ("Tile W is split into two horizontal halves of two colors. "
         "Which half is white — top or bottom?", "pos", "top"),
        ("Which tile is red? Answer with the letter label (N, E, S, W or C).",
         "pos", "e"),
        ("Are tiles N and E the same color? Answer yes or no.", "yesno", "no"),
    ],
}

COLOR_WORDS = {
    "red": ["red"],
    "blue": ["blue"],
    "yellow": ["yellow"],
    "green": ["green"],
    "white": ["white", "cream", "ivory", "off-white", "pink"],  # idx 12 is a pale pink
}


def grade(kind: str, expected, answer: str) -> bool:
    a = answer.lower().strip()
    if not a:
        return False
    if kind == "color":
        return any(w in a for w in COLOR_WORDS[expected])
    if kind == "colors2":
        return all(any(w in a for w in COLOR_WORDS[c]) for c in expected)
    if kind == "pos":
        if expected in ("n", "e", "s", "w", "c"):
            return bool(re.search(rf"\b{expected}\b", a)) or expected + "ast" in a
        return expected in a  # "top" / "bottom"
    if kind == "yesno":
        first = a.split()[0].strip(".,!?")
        return first == expected
    return False


def parse_answers(content: str, n: int) -> list[str]:
    """Extract '1. answer' style lines; missing numbers stay ''."""
    out = [""] * n
    for line in content.splitlines():
        m = re.match(r"^\s*(\d+)\s*[.):]\s*(.+?)\s*$", line)
        if m:
            i = int(m.group(1)) - 1
            if 0 <= i < n:
                out[i] = m.group(2)
    return out


def img_to_data_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(url: str, model: str, data_url: str, text: str,
        timeout: int = 600) -> tuple[str, dict, float]:
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 300,
        # Qwen3.8 is a thinking model: without this, all max_tokens go to
        # reasoning_content and the visible content stays empty.
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": text},
            ]},
        ],
    }
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:800]}") from None
    dt = time.time() - t0
    content = data["choices"][0]["message"]["content"]
    return content, data.get("usage", {}), dt


SEPARATE_NAMES = ("center", "north", "east", "south", "west")


def ask_separate(url: str, model: str, tiles: dict[str, np.ndarray],
                 text: str, timeout: int = 600) -> tuple[str, dict, float]:
    """Same probe, but the five tiles go in as separate images (Felix's
    original design): one intro text part fixing the image->role order,
    five image parts, then the questions."""
    intro = (
        "Here are five separate pixel-art tiles, one per image, in this "
        "order: image 1 = C (center), image 2 = N (north), image 3 = E "
        "(east), image 4 = S (south), image 5 = W (west).")
    content: list[dict] = [{"type": "text", "text": intro}]
    for name in SEPARATE_NAMES:
        content.append({"type": "image_url",
                        "image_url": {"url": img_to_data_url(tiles[name])}})
    content.append({"type": "text", "text": text})
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": SEPARATE_SYSTEM},
            {"role": "user", "content": content},
        ],
    }
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:800]}") from None
    dt = time.time() - t0
    content_txt = data["choices"][0]["message"]["content"]
    return content_txt, data.get("usage", {}), dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen3.8-27b-ud-q6-k-xl-multimodal")
    ap.add_argument("--sizes", default="256,512,1024")
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--separate", action="store_true",
                    help="send the five tiles as separate images instead of "
                         "one labeled cross composite")
    args = ap.parse_args()

    pal = Palette.load("config/palette_dev64.json")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = [int(s) for s in args.sizes.split(",")]

    results = []
    for kind in ("main", "alt"):
        tiles = make_tiles(pal, kind)
        qs = QUESTIONS[kind]
        user_text = "Questions:\n" + "\n".join(
            f"{i+1}. {q}" for i, (q, _, _) in enumerate(qs)) + "\n\n" + FORMAT
        for rs in (sizes if kind == "main" else [512]):
            if args.separate:
                tag = f"{kind}_{rs}_sep"
                sep = {n: nearest_resize(tiles[n], rs) for n in SEPARATE_NAMES}
                strip = np.hstack([nearest_resize(tiles[n], 256)
                                   for n in SEPARATE_NAMES])
                strip_path = out / f"separate_{tag}.png"
                Image.fromarray(strip, mode="RGB").save(strip_path)
                print(f"[{tag}] 5 separate tiles @ {rs} -> {strip_path}")
                try:
                    content, usage, dt = ask_separate(args.url, args.model,
                                                      sep, user_text)
                except RuntimeError as e:
                    print(f"[{tag}] FAILED: {e}")
                    results.append({"tag": tag, "size": rs, "error": str(e)})
                    continue
            else:
                comp = render_cross(tiles, rs)
                tag = f"{kind}_{rs}"
                comp_path = out / f"composite_{tag}.png"
                Image.fromarray(comp, mode="RGB").save(comp_path)
                print(f"[{tag}] composite {comp.shape[0]}x{comp.shape[1]} -> {comp_path}")
                try:
                    content, usage, dt = ask(args.url, args.model,
                                              img_to_data_url(comp), user_text)
                except RuntimeError as e:
                    print(f"[{tag}] FAILED: {e}")
                    results.append({"tag": tag, "size": rs, "error": str(e)})
                    continue
            answers = parse_answers(content, len(qs))
            per_q = []
            for (q, kindq, expected), ans in zip(qs, answers):
                ok = grade(kindq, expected, ans)
                per_q.append({"question": q, "expected": expected,
                              "answer": ans, "correct": ok})
                print(f"[{tag}] Q {'OK ' if ok else 'MISS'} {q[:52]:<52} -> {ans[:60]}")
            n_ok = sum(p["correct"] for p in per_q)
            print(f"[{tag}] {n_ok}/{len(qs)} correct in {dt:.1f}s "
                  f"usage={usage}")
            results.append({"tag": tag, "size": rs, "seconds": round(dt, 1),
                            "usage": usage, "n_correct": n_ok,
                            "n_total": len(qs), "questions": per_q,
                            "raw": content})

    fname = out / ("e3_separate_results.json" if args.separate
                   else "e3_results.json")
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {fname}")


if __name__ == "__main__":
    main()
