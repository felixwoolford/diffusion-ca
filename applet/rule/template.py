"""`template` rule backend: deterministic canned prompts.

For pipeline tests without an LLM (SPEC §2.3). Output is a pure function of
(arm, step, cell) — same request, same prompt, every time.
"""
from __future__ import annotations

import hashlib

from . import RuleRequest, RuleResult
from .prompts import PROMPT_MAX

T2I_PATTERNS = [
    "a {art}x{art} pixel-art tile with a chunky {shape} in {color} on a flat {bg} background",
    "a {art}x{art} pixel-art tile: two stacked bands, {color} over {bg}, with a small {shape} glyph in the corner",
    "a {art}x{art} pixel-art tile: a bold diagonal stripe in {color} across a {bg} field, flat colors only",
    "a {art}x{art} pixel-art tile: a centered {shape} made of blocky {color} squares on {bg}",
]
SHAPES = ["diamond", "cross", "arrow", "mushroom", "key", "star", "spiral", "house"]
COLORS = ["red", "teal", "amber", "lavender", "moss", "cyan", "coral", "slate"]
BGS = ["dark", "light", "mid-tone", "deep blue", "warm gray", "pale green", "charcoal", "sand"]


class TemplateBackend:
    name = "template"

    def __init__(self, art_size: int, palette_n: int, relation: str = "evolve"):
        self.art_size = art_size
        self.palette_n = palette_n
        self.relation = relation

    def _pick(self, step: int, r: int, c: int, salt: str) -> int:
        h = hashlib.sha256(f"{salt}|{step}|{r}|{c}".encode()).digest()
        return int.from_bytes(h[:4], "big")

    def rule(self, req: RuleRequest) -> RuleResult:
        r, c = req.cell
        i = self._pick(req.step, r, c, "pattern")
        if req.arm == "t2i":
            prompt = T2I_PATTERNS[i % len(T2I_PATTERNS)].format(
                art=self.art_size,
                shape=SHAPES[i % len(SHAPES)],
                color=COLORS[i % len(COLORS)],
                bg=BGS[i % len(BGS)],
            )
        elif req.arm == "edit_self":
            prompt = (
                f"edit the tile: shift the main color one hue toward {COLORS[i % len(COLORS)]}, "
                f"grow the central {SHAPES[i % len(SHAPES)]} by one art-pixel, keep the background flat"
            )
        else:  # edit_neighborhood
            prompt = (
                f"edit the center tile (image 1): continue the border motif from {COLORS[i % len(COLORS)]} "
                f"in the north neighbor (image 2) across its edge; keep the existing {SHAPES[i % len(SHAPES)]}"
            )
        if len(prompt) > PROMPT_MAX:
            prompt = prompt[:PROMPT_MAX]
        return RuleResult(prompt=prompt)
