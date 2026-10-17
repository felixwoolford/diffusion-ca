import numpy as np

from applet.rule import RuleRequest, make_backend
from applet.rule.prompts import (PROMPT_MAX, build_system_prompt, build_user_text,
                                 frame_text)
from applet.engine.config import Config
from applet.engine.palette import Palette
from conftest import STANDARD


def make_cfg(**kw):
    base = dict(rule_backend="template", art_size=8, palette_n=8)
    base.update(kw)
    return Config(**{k: v for k, v in base.items() if k in
                     {f.name for f in Config.__dataclass_fields__.values()}})


def req(arm="t2i", step=3, cell=(1, 2), composite=None, instruction=""):
    comp = composite if composite is not None else np.zeros((96, 96, 3), np.uint8)
    return RuleRequest(composite=comp, step=step, cell=cell, arm=arm,
                       frame=frame_text(8, 8), instruction=instruction)


def test_template_deterministic():
    b = make_backend(make_cfg())
    r1, r2 = b.rule(req()), b.rule(req())
    assert r1.prompt == r2.prompt
    assert r2.reasoning == ""


def test_template_varies_by_cell_and_step():
    b = make_backend(make_cfg())
    prompts = {b.rule(req(cell=(r, c), step=s)).prompt
               for r in range(4) for c in range(4) for s in range(2)}
    assert len(prompts) > 4  # not one constant prompt


def test_template_arm_styles():
    b = make_backend(make_cfg())
    assert "pixel-art tile" in b.rule(req("t2i")).prompt
    e = b.rule(req("edit_self")).prompt
    assert e.startswith("edit")
    n = b.rule(req("edit_neighborhood")).prompt
    assert "image 1" in n and "image 2" in n


def test_template_within_prompt_limit():
    b = make_backend(make_cfg())
    for arm in ("t2i", "edit_self", "edit_neighborhood"):
        p = b.rule(req(arm)).prompt
        assert 0 < len(p) <= PROMPT_MAX


def test_make_backend_unknown():
    cfg = make_cfg(rule_backend="bogus")
    import pytest
    with pytest.raises(ValueError):
        make_backend(cfg)


def test_system_prompt_contents():
    sp = build_system_prompt(32, 64, arm="edit_neighborhood", relation="contrast")
    assert "32x32 pixel-art tile" in sp
    assert "no gradients" in sp
    assert "labels" in sp.lower()
    assert "visually *distinct*" in sp
    spl = sp.lower()
    assert "image 1" in spl and "images 2-5" in spl
    assert f"at most {PROMPT_MAX} characters" in sp
    sp_t = build_system_prompt(16, 8, arm="t2i", relation="evolve")
    assert "16x16 pixel-art tile" in sp_t
    assert "limited 8-color palette" in sp_t
    assert "text-to-image" in sp_t


def test_system_prompt_palette_hex():
    pal = Palette(STANDARD.copy())
    sp = build_system_prompt(8, 8, arm="t2i", palette_hex=pal.to_hex())
    assert "#ff0000" in sp


def test_system_prompt_separate_render():
    sp = build_system_prompt(32, 64, arm="edit_neighborhood", render="separate")
    assert "five SEPARATE images" in sp
    assert "image 1 = C" in sp and "image 5 = W" in sp
    assert "Labels are drawn" not in sp  # separate mode has no drawn-in labels
    # default stays composite
    sp2 = build_system_prompt(32, 64, arm="t2i")
    assert "cross layout" in sp2


def test_user_text():
    t = build_user_text(req(step=7, cell=(2, 3), instruction="grow the crystals"))
    assert "Step 7" in t and "(2,3)" in t and "grow the crystals" in t
    t2 = build_user_text(req(instruction=""))
    assert "Run instruction" not in t2


def test_llamacpp_disables_thinking(monkeypatch):
    """E3 (2026-10-17): the Qwen3.8 multimodal deployment is a thinking
    model; without enable_thinking=False the whole token budget goes to
    reasoning_content and the visible prompt comes back empty."""
    import json
    from applet.rule.llamacpp import LlamaCppBackend

    captured = {}

    class FakeResp:
        def __init__(self, body):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._body

    def fake_urlopen(r, timeout=None):
        captured["payload"] = json.loads(r.data)
        return FakeResp(json.dumps(
            {"choices": [{"message": {"content": "a red tile"}}]}).encode())

    monkeypatch.setattr("applet.rule.llamacpp.urllib.request.urlopen",
                        fake_urlopen)
    b = LlamaCppBackend(url="http://x", model="m")
    res = b.rule(req())
    assert res.prompt == "a red tile"
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_llamacpp_separate_payload(monkeypatch):
    """Separate render mode (SPEC §2.6): intro text, five image parts in the
    fixed C/N/E/S/W order, question text last; separate-mode system prompt."""
    import json
    from applet.rule.llamacpp import LlamaCppBackend

    captured = {}

    class FakeResp:
        def __init__(self, body):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._body

    def fake_urlopen(r, timeout=None):
        captured["payload"] = json.loads(r.data)
        return FakeResp(json.dumps(
            {"choices": [{"message": {"content": "a red tile"}}]}).encode())

    monkeypatch.setattr("applet.rule.llamacpp.urllib.request.urlopen",
                        fake_urlopen)
    tiles = {n: np.full((32, 32, 3), (i + 1) * 40, np.uint8)
             for i, n in enumerate(("center", "north", "east", "south", "west"))}
    b = LlamaCppBackend(url="http://x", model="m")
    b.rule(RuleRequest(composite=None, separate=tiles, step=0, cell=(0, 0),
                       arm="t2i", frame=frame_text(8, 8), instruction=""))
    user = captured["payload"]["messages"][1]["content"]
    assert user[0]["type"] == "text" and "image 1 = C" in user[0]["text"]
    imgs = [p for p in user if p["type"] == "image_url"]
    assert len(imgs) == 5
    assert all(p["image_url"]["url"].startswith("data:image/png;base64,")
               for p in imgs)
    assert user[-1]["type"] == "text" and "Output the prompt now." in user[-1]["text"]
    assert "five SEPARATE images" in captured["payload"]["messages"][0]["content"]
