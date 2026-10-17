"""`pi_agent` rule backend: headless pi invocation (SPEC §2.3).

Constrained per the spec: single multimodal model call, fixed system prompt,
no tools, strict output format. In the dev container the rule model *is* the
dev model (qwen3.8-27b via llama.cpp), so rule calls are headless `pi -p`
subagents on the same model — RAM-cached and cheap, but they cannot run in
parallel with the dev agent's own work: use `concurrency: 1` with this
backend (see notes/DEV_NOTES.md).

Command shape (verified from `pi --help`, 2026-10-17):
  pi -p --no-tools --no-session --no-extensions --no-skills --no-mcp
     --no-context-files [--model PATTERN] --system-prompt <SYS>
     -- "@<composite.png>" "<user text>"

Separate render mode (SPEC §2.6): five @image files in the fixed order
(image 1 = C, 2 = N, 3 = E, 4 = S, 5 = W) with SEPARATE_INTRO as the first
text part. Note: pi's docs only guarantee the @files are "included in the
first prompt" — the exact part ordering is not documented; the system
prompt states the fixed image order, so either layout is unambiguous.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from . import RuleRequest, RuleResult
from .llamacpp import RuleError, _extract
from .prompts import (SEPARATE_INTRO, SEPARATE_ORDER, build_system_prompt,
                      build_user_text)


class PiAgentBackend:
    name = "pi_agent"

    def __init__(self, pi_bin: str = "pi", model: str | None = None,
                 timeout: int = 600, art_size: int = 32, palette_n: int = 64,
                 relation: str = "evolve", palette_hex: list[str] | None = None):
        self.pi_bin = pi_bin
        self.model = model
        self.timeout = timeout
        self.art_size = art_size
        self.palette_n = palette_n
        self.relation = relation
        self.palette_hex = palette_hex

    def rule(self, req: RuleRequest) -> RuleResult:
        system = build_system_prompt(self.art_size, self.palette_n, arm=req.arm,
                                     relation=self.relation, palette_hex=self.palette_hex,
                                     render=req.render)
        from PIL import Image
        # one temp PNG per input image (composite: 1, separate: 5)
        if req.separate is not None:
            arrays = [req.separate[n] for n, _ in SEPARATE_ORDER]
            first_text = SEPARATE_INTRO
        else:
            arrays = [req.composite]
            first_text = None
        tmps: list[Path] = []
        try:
            for arr in arrays:
                fd, tmp = tempfile.mkstemp(suffix=".png", prefix="sdca_rule_")
                os.close(fd)
                p = Path(tmp)
                Image.fromarray(arr, mode="RGB").save(p)
                tmps.append(p)
            cmd = [self.pi_bin, "-p",
                   "--no-tools", "--no-session",
                   "--no-extensions", "--no-skills", "--no-mcp",
                   "--no-context-files", "--mode", "text",
                   "--system-prompt", system,
                   "--"]
            if first_text is not None:
                cmd.append(first_text)
            cmd.extend(f"@{p}" for p in tmps)
            cmd.append(build_user_text(req))
            if self.model:
                cmd[1:1] = ["--model", self.model]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=self.timeout)
            except subprocess.TimeoutExpired as e:
                raise RuleError(f"pi_agent timed out after {self.timeout}s") from e
            if proc.returncode != 0:
                raise RuleError(f"pi_agent exit {proc.returncode}: {proc.stderr[-1000:]}")
            return _extract(proc.stdout)
        finally:
            for p in tmps:
                p.unlink(missing_ok=True)
