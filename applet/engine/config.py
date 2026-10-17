"""Run configuration (SPEC section 3): every knob, YAML-loaded, defaults from the spec.

The engine reads all sizes, schedules, arms, seeds, servers and paths from here;
nothing is hardcoded in the pipeline. Local run configs are gitignored — commit
`config/*.example.*` instead (AGENTS.md).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ARMS = ("t2i", "edit_self", "edit_neighborhood")
SCHEDULES = ("random_uniform", "ring_scan", "all")
RULE_BACKENDS = ("llamacpp", "pi_agent", "template")
GENERATORS = ("comfy", "fake")
SEED_MODES = ("random", "deterministic")
HISTORY = ("none", "log+snapshot", "ring", "full")


@dataclass
class ArmSpec:
    """Per-arm wiring into its workflow copy (SPEC section 2.4 node IDs)."""

    workflow: str = ""
    prompt_node: str = ""          # node holding the final prompt text
    prompt_field: str = "value"    # its text input name
    size_via: str = "empty_latent"  # "resolution_selector" | "empty_latent"
    size_node: str = ""            # node setting the output size
    image_nodes: dict[str, str] = field(default_factory=dict)  # "image_1".."image_n" -> LoadImage node
    seed_node: str = ""            # KSampler seed input

    @classmethod
    def defaults(cls, arm: str) -> "ArmSpec":
        if arm == "t2i":
            return cls(workflow="workflows/t2i_ca.json", prompt_node="497",
                       size_via="resolution_selector", size_node="13", seed_node="490")
        if arm == "edit_self":
            return cls(workflow="workflows/edit_ca.json", prompt_node="520",
                       size_via="empty_latent", size_node="505",
                       image_nodes={"image_1": "470"}, seed_node="507")
        return cls(workflow="workflows/edit_neigh_ca.json", prompt_node="520",
                   size_via="empty_latent", size_node="505",
                   image_nodes={"image_1": "470"}, seed_node="507")  # image_2..5 filled by the edit_neigh workflow copy


@dataclass
class Config:
    # field (SPEC 1/3)
    grid_n: int = 16
    art_size: int = 32
    render_scale: int = 2
    palette_n: int = 64
    palette_path: str | None = None      # json/npy palette file (picked in E5)
    initial_field: str | None = None     # square PNG, grid_n*art_size px, tiled initial field
    # sizes
    write_size: int = 1024               # generation size (multiple of 32, of art_size)
    read_size: int = 512                 # neighborhood render size (multiple of 32)
    # dynamics
    arm: str = "edit_self"
    schedule: str = "random_uniform"
    K: int = 16
    seed_mode: str = "random"            # "random" | "deterministic"
    seed: int | None = None              # base seed (deterministic mode); None -> os random
    instruction: str = ""                # per-run evolution instruction (GUI text)
    relation: str = "evolve"             # "evolve" | "contrast" (neighbor relation dial, SPEC 2.8)
    # rule
    rule_backend: str = "llamacpp"
    llm_url: str | None = None           # llama.cpp OpenAI-compatible endpoint
    llm_model: str = "qwen3.8-27b"
    rule_temperature: float = 1.0
    pi_bin: str = "pi"                   # pi_agent backend: pi executable
    pi_model: str | None = None          # pi_agent backend: model pattern (None -> pi default)
    pi_timeout: int = 600
    # generation
    comfy_url: str | None = None         # ComfyUI server (check /models/diffusion_models each run)
    generator: str = "comfy"             # "comfy" | "fake" (deterministic, no server; pipeline tests)
    concurrency: int = 4                 # parallel rule/generation dispatch
    arms: dict[str, ArmSpec] = field(default_factory=dict)
    # persistence
    state_dir: str = "state"
    images_out_dir: str = "images_out"
    run_name: str = "run"
    history: str = "log+snapshot"
    # metrics
    uniformity_warn: float = 0.5         # collapse-watch threshold (SPEC 2.8)

    # ------------------------------------------------------------------ #
    def arm_spec(self, arm: str) -> ArmSpec:
        return self.arms.get(arm, ArmSpec.defaults(arm))

    def validate(self) -> None:
        errs: list[str] = []
        if self.grid_n < 1:
            errs.append("grid_n must be >= 1")
        if self.art_size < 1:
            errs.append("art_size must be >= 1")
        if self.write_size % 32 != 0:
            errs.append(f"write_size {self.write_size} must be a multiple of 32")
        if self.write_size % self.art_size != 0:
            errs.append(f"write_size {self.write_size} must be a multiple of art_size {self.art_size}")
        if self.read_size % 32 != 0:
            errs.append(f"read_size {self.read_size} must be a multiple of 32")
        if self.render_scale < 1:
            errs.append("render_scale must be >= 1")
        if self.K < 1:
            errs.append("K must be >= 1")
        if self.arm not in ARMS:
            errs.append(f"arm must be one of {ARMS}")
        if self.schedule not in SCHEDULES:
            errs.append(f"schedule must be one of {SCHEDULES}")
        if self.rule_backend not in RULE_BACKENDS:
            errs.append(f"rule_backend must be one of {RULE_BACKENDS}")
        if self.generator not in GENERATORS:
            errs.append(f"generator must be one of {GENERATORS}")
        if self.seed_mode not in SEED_MODES:
            errs.append(f"seed_mode must be one of {SEED_MODES}")
        if self.history not in HISTORY:
            errs.append(f"history must be one of {HISTORY}")
        if self.history in ("ring", "full"):
            errs.append(f"history={self.history!r} is planned but not yet implemented; use 'log+snapshot' or 'none'")
        if self.relation not in ("evolve", "contrast"):
            errs.append(f"relation must be 'evolve' or 'contrast'")
        if errs:
            raise ValueError("invalid config: " + "; ".join(errs))

    def check_runtime(self, needs_comfy: bool, needs_llm: bool) -> None:
        """Server-specific checks, run at startup (not at config load)."""
        errs = []
        if needs_comfy and not self.comfy_url:
            errs.append("comfy_url is not set")
        if needs_llm and self.rule_backend == "llamacpp" and not self.llm_url:
            errs.append("llm_url is not set (rule_backend=llamacpp)")
        if errs:
            raise ValueError("runtime config: " + "; ".join(errs))


def load_config(path: str | Path) -> Config:
    """Load a YAML config over the defaults. Unknown keys raise (typo guard)."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: config must be a mapping")
    fields = {f.name for f in dataclasses.fields(Config)}
    arms = data.pop("arms", None)
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)} (known: {sorted(fields)})")
    cfg = Config(**data)
    if arms is not None:
        if not isinstance(arms, dict):
            raise ValueError("arms: must be a mapping of arm -> ArmSpec fields")
        known_arms = set(ARMS)
        for name, spec in arms.items():
            if name not in known_arms:
                raise ValueError(f"arms: unknown arm {name!r}")
            spec_fields = {f.name for f in dataclasses.fields(ArmSpec)}
            bad = set(spec) - spec_fields
            if bad:
                raise ValueError(f"arms.{name}: unknown keys {sorted(bad)}")
            base = ArmSpec.defaults(name)
            merged = dataclasses.asdict(base)
            merged.update(spec)
            cfg.arms[name] = ArmSpec(**merged)
    cfg.validate()
    return cfg
