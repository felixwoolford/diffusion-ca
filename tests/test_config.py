from applet.engine.config import Config, load_config
from applet.engine.palette import Palette
import numpy as np
import pytest


def test_defaults_match_spec():
    c = Config()
    assert (c.grid_n, c.art_size, c.render_scale, c.palette_n) == (16, 32, 2, 64)
    assert (c.write_size, c.read_size) == (1024, 512)
    assert (c.arm, c.schedule, c.K) == ("edit_self", "random_uniform", 16)
    assert c.rule_backend == "llamacpp"
    assert c.concurrency == 4
    assert c.history == "log+snapshot"


def test_example_yaml_loads():
    c = load_config("config/example.yaml")
    c.validate()
    assert c.arm == "edit_self"
    assert c.arm_spec("t2i").prompt_node == "497"
    assert c.arm_spec("edit_self").image_nodes == {"image_1": "470"}
    assert c.arm_spec("edit_neighborhood").size_node == "505"


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("grid_n: 8\nwrtie_size: 512\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(p)


def test_validation_errors(tmp_path):
    bad = {
        "write_size: 100\n": "multiple of 32",
        "read_size: 100\n": "multiple of 32",
        "art_size: 64\nwrite_size: 96\n": "multiple of art_size",
        "arm: no such arm\n": "arm",
        "K: 0\n": "K",
        "history: ring\n": "not yet implemented",
    }
    for text, msg in bad.items():
        p = tmp_path / "c.yaml"
        p.write_text(text)
        with pytest.raises(ValueError, match=msg):
            load_config(p)


def test_arms_override_merges_with_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("arms:\n  t2i:\n    prompt_node: \"123\"\n")
    c = load_config(p)
    spec = c.arm_spec("t2i")
    assert spec.prompt_node == "123"
    assert spec.workflow == "workflows/t2i_ca.json"  # default kept


def test_arms_unknown_arm_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("arms:\n  bogus:\n    prompt_node: \"1\"\n")
    with pytest.raises(ValueError, match="unknown arm"):
        load_config(p)
