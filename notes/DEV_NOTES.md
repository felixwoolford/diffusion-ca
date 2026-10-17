# Dev notes (engineering discoveries)

Permanent notes from the build phase, so future agents don't re-discover them.
Experiment results (E1–E7) go in their own `notes/E*.md` files; this file is
for environment, tooling, and workflow-graph facts.

## Base workflow node inventory (verified 2026-10-17, "Adjusted workflow" commit)

`base_workflows/OBSERVATIONS.md` describes the **pre-adjustment** graphs in
places. Current graphs (API format, node id → class):

- **`t2i.json` (12 nodes)** — already a *direct* variant, no enhancer chain:
  - `497` `PrimitiveStringMultiline` = the final prompt (override this).
  - `485` `TextEncodeQwenImage21` (prompt, negative_prompt, resolution).
  - `13` `ResolutionSelector` (aspect_ratio, megapixels, multiple) — default 1:1 / 1 MP → 1024².
  - `488` `EmptyLatentImage` (batch_size, height, width) — direct size override.
  - `490` `KSampler` (has `seed`; turbo 8-step/cfg 1/euler/simple via `499` LoRA — do not change).
  - `484` `UNETLoader` (swap `unet_name` per server), `496` `QwenImage21Cache`,
    `486` `CLIPLoader`, `487` `VAELoader`, `489` `VAEDecode`, `461` `SaveImageAdvanced`.
  - The old `491 TextGenerate` / `494 switch` enhancer nodes no longer exist.
    SPEC §2.4 "prompt override node 497" matches the current graph.
- **`edit.json` (15 nodes)** — also already direct (enhancer chain 511–516 gone):
  - `520` `PrimitiveStringMultiline` → `510.prompt` (override `520.value` for the prompt).
  - `510` `TextEncodeQwenImage21` (prompt, negative_prompt, resolution, vae, `images.image_1..n`;
    outputs `[CONDITIONING, CONDITIONING, LATENT]`, latent follows image_1 aspect at resolution 0).
  - `470` `LoadImage` = image_1 (the image being edited); `475` `LoadImage` = image_2 (reference).
  - Single-reference form: still must **delete node `475` and remove `images.image_2` from `510`**
    (OBSERVATIONS' extra steps — deleting `515`, removing the `image` input of `516` — are obsolete,
    those nodes no longer exist).
  - `508` `ComfySwitchNode` chooses KSampler latent: `true` → `505` `EmptyLatentImage`
    (fixed 1024²; set its width/height for write_size), `false` → `510` output 2 (follows input size).
  - `507` `KSampler` (has `seed`), `502` `UNETLoader`, `517` LoRA, `509` Cache,
    `503` `CLIPLoader`, `504` `VAELoader`, `506` `VAEDecode`, `472` `ImageCompare` (display only),
    `461` `SaveImageAdvanced`.

Consequence for the engine: no enhancer bypass needed in any arm — the rule
backend's prompt goes straight into the `PrimitiveStringMultiline` node.

## Environment

- **Rule LLM = the dev model itself.** `PI_MODEL=qwen3.8-27b-ud-q6-k-xl`,
  `PI_PROVIDER=llama.cpp` (env vars in the container). Felix (2026-10-17): rule
  calls should direct headless pi subagents to this same model; it's RAM-cached
  and cheap, but **cannot run in parallel with the dev agent's own work** →
  when `rule_backend: pi_agent`, use `concurrency: 1` (one rule call at a time,
  sequenced, not overlapped with other model work).
- Python 3.14.7 at `/opt/venv`; stack pinned with uv (`.python-version` = 3.14,
  `uv.lock`): numpy, pillow, pyyaml (+ dev pytest). `uv run pytest` from repo root.
- ComfyUI servers (SSH tunnels, check every run via `GET /models/diffusion_models`):
  8188 = 5090/32GB (`qwen_image_2.1_bf16.safetensors`), 8189 = 4080/16GB
  (`qwen_image_2.1_int8_convrot.safetensors`). On 2026-10-17 ~16:05 UTC, 8188 was
  down, 8189 up.

## pi CLI (for the `pi_agent` rule backend)

Docs: `/usr/lib/node_modules/@earendil-works/pi-coding-agent/docs/` (cli.md, usage.md).
Non-interactive shape (verified from `pi --help` 2026-10-17):

```
pi -p --no-tools --no-session --no-extensions --no-skills --no-mcp --no-context-files
   [--model PATTERN] --system-prompt "<SYS>" -- "@<composite.png>" "<user text>"
```

- `@path` includes a text file or **image** in the first prompt (cli.md line ~35).
- `--no-tools` = "single multimodal model call, no tools" (SPEC §2.3 constraint).
- `--mode text` (default) prints the final text to stdout.
- Keep `--no-context-files` so the CA rule session doesn't pick up AGENTS.md.

## Stage plan (agreed with Felix, 2026-10-17)

Stage 1 = **engine core first**: headless modules in `applet/engine/` +
`applet/rule/` with unit tests, no remote generation yet. Then experiments
E1–E7 (SPEC §6) — E1 (VAE alignment probe) first, needs `workflows/t2i_ca.json`
copy. Then GUI (toolkit still unconfirmed, PySide6 suggested).
