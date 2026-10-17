"""Rule backends: the CA's update function (SPEC §2.3).

Contract — a pure stateless function: the 5 neighborhood tiles in (as one
labeled cross composite, or as five separate one-tile images — config
`rule_render`, SPEC §2.6), 1 generation prompt out. The backend holds no
state and owns no ComfyUI plumbing; the applet does all of that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..engine.config import Config


@dataclass
class RuleRequest:
    """Input to one rule call (SPEC §2.3).

    The 5 tiles (nearest-upscaled to read_size) arrive in one of two
    renderings (config `rule_render`, SPEC §2.6); exactly one is set:
      composite — one labeled cross layout, (3*read_size)² RGB
      separate  — five separate one-tile images (Felix's original design;
                  E3: cheaper at 512, and 1024 stays under the per-image cap)
    """
    step: int
    cell: tuple[int, int]          # (row, col)
    arm: str                       # "t2i" | "edit_self" | "edit_neighborhood"
    frame: str                     # pixel-art framing text, injected by the applet
    instruction: str               # optional per-run evolution instruction
    composite: np.ndarray | None = None      # labeled cross render (3*read_size)² RGB
    separate: dict[str, np.ndarray] | None = None  # {"center","north","east","south","west"} -> read_size RGB

    @property
    def render(self) -> str:
        return "separate" if self.separate is not None else "composite"


@dataclass
class RuleResult:
    prompt: str                    # exactly one prompt string (<=500 chars)
    reasoning: str = ""            # optional; logged only


class RuleBackend(Protocol):
    name: str

    def rule(self, req: RuleRequest) -> RuleResult: ...


def make_backend(cfg: Config) -> RuleBackend:
    if cfg.rule_backend == "template":
        from .template import TemplateBackend
        return TemplateBackend(art_size=cfg.art_size, palette_n=cfg.palette_n,
                               relation=cfg.relation)
    if cfg.rule_backend == "llamacpp":
        from .llamacpp import LlamaCppBackend
        return LlamaCppBackend(url=cfg.llm_url, model=cfg.llm_model,
                               temperature=cfg.rule_temperature,
                               art_size=cfg.art_size, palette_n=cfg.palette_n,
                               relation=cfg.relation, timeout=cfg.pi_timeout)
    if cfg.rule_backend == "pi_agent":
        from .pi_agent import PiAgentBackend
        return PiAgentBackend(pi_bin=cfg.pi_bin, model=cfg.pi_model,
                              timeout=cfg.pi_timeout, art_size=cfg.art_size,
                              palette_n=cfg.palette_n, relation=cfg.relation)
    raise ValueError(f"unknown rule backend {cfg.rule_backend!r}")
