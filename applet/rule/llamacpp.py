"""`llamacpp` rule backend: direct HTTP to a llama.cpp server.

OpenAI-compatible /v1/chat/completions, multimodal (image as data URL PNG).
Default target model: Qwen3.8-27B (SPEC §2.3).
"""
from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from . import RuleRequest, RuleResult
from .prompts import PROMPT_MAX, build_system_prompt, build_user_text


class RuleError(RuntimeError):
    pass


def composite_to_data_url(composite: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(composite, mode="RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class LlamaCppBackend:
    name = "llamacpp"

    def __init__(self, url: str | None, model: str, temperature: float = 1.0,
                 art_size: int = 32, palette_n: int = 64,
                 relation: str = "evolve", palette_hex: list[str] | None = None,
                 timeout: int = 300):
        if not url:
            raise RuleError("llamacpp backend requires llm_url in config")
        self.url = url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.art_size = art_size
        self.palette_n = palette_n
        self.relation = relation
        self.palette_hex = palette_hex
        self.timeout = timeout

    def rule(self, req: RuleRequest) -> RuleResult:
        system = build_system_prompt(self.art_size, self.palette_n, arm=req.arm,
                                     relation=self.relation, palette_hex=self.palette_hex)
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            # Qwen3.8 is a thinking model: without this, the whole budget
            # goes to reasoning_content and the visible content is empty
            # (found in E3, 2026-10-17).
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": composite_to_data_url(req.composite)}},
                    {"type": "text", "text": build_user_text(req)},
                ]},
            ],
        }
        body = json.dumps(payload).encode()
        r = urllib.request.Request(f"{self.url}/v1/chat/completions", data=body,
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            raise RuleError(f"llamacpp HTTP {e.code}: {e.read().decode(errors='replace')[:1000]}") from None
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuleError(f"llamacpp unexpected response: {json.dumps(data)[:500]}") from e
        return _extract(content)


def _extract(content: str) -> RuleResult:
    """Enforce the output contract: exactly one prompt, <= PROMPT_MAX chars."""
    content = content.strip()
    # strip a leading "PROMPT:" style label if the model added one
    if content[:8].upper().startswith("PROMPT"):
        content = content.split(":", 1)[1].strip() if ":" in content else ""
    reasoning = ""
    if len(content) > PROMPT_MAX:
        # keep the first sentence-ish chunk that fits; log the rest as reasoning
        cut = content.rfind(" ", 0, PROMPT_MAX)
        if cut > PROMPT_MAX // 2:
            reasoning, content = content[cut:].strip(), content[:cut].rstrip()
        else:
            reasoning, content = content[PROMPT_MAX:].strip(), content[:PROMPT_MAX]
    if not content:
        raise RuleError("rule LLM returned an empty prompt")
    return RuleResult(prompt=content, reasoning=reasoning)
