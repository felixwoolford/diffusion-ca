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
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from . import RuleRequest, RuleResult
from .llamacpp import RuleError, _extract
from .prompts import build_system_prompt, build_user_text


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
                                     relation=self.relation, palette_hex=self.palette_hex)
        from PIL import Image
        fd, tmp = tempfile.mkstemp(suffix=".png", prefix="sdca_rule_")
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            Image.fromarray(req.composite, mode="RGB").save(tmp_path)
            cmd = [self.pi_bin, "-p",
                   "--no-tools", "--no-session",
                   "--no-extensions", "--no-skills", "--no-mcp",
                   "--no-context-files", "--mode", "text",
                   "--system-prompt", system,
                   "--", f"@{tmp_path}", build_user_text(req)]
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
            tmp_path.unlink(missing_ok=True)
