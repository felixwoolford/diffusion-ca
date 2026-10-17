"""Rule backends: the CA's update function (SPEC §2.3).

Contract — a pure stateless function: 5 tiles in (as one labeled cross
composite), 1 generation prompt out. The backend holds no state and owns no
ComfyUI plumbing; the applet does all of that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..engine.config import Config


@dataclass
class RuleRequest:
    composite: np.ndarray          # labeled cross render (3*read_size)² RGB
    step: int
    cell: tuple[int, int]          # (row, col)
    arm: str                       # "t2i" | "edit_self" | "edit_neighborhood"
    frame: str                     # pixel-art framing text, injected by the applet
    instruction: str               # optional per-run evolution instruction


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
