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

## Stage 1 status (engine core — DONE 2026-10-17)

Headless engine built and unit-tested (64 tests, `uv run python -m pytest`):

- `applet/engine/config.py` — every SPEC §3 knob, YAML-loaded, strict (unknown
  keys raise), per-arm workflow wiring in `Config.arms` (ArmSpec: prompt node,
  size node, image slots, seed node). `config/example.yaml` documents all of it.
- `palette.py` — global palette, nearest-color quantize (int32 math! see below),
  JSON/npy persistence, P-mode PNG interop. Real 64-color palette still pending (E5);
  `config/palette_dev64.json` is a deterministic dev stand-in (make_test_palette).
- `imageops.py` — nearest upscale; block downscale = mode over quantized palette
  indices per block, median→nearest tie fallback, never mean.
- `grid.py` — field of (n,n,art,art) uint8 indices (the ground truth), wrap
  neighborhood, snapshots, PNG initial-field tiling.
- `schedule.py` — random_uniform (seeded rng), ring_scan (deterministic moving
  diagonal band, covers field in 2n-1 steps), all.
- `render.py` — labeled cross composite (N/W/C/E/S), labels drawn in via
  nearest-scaled default-font letters (no font files), white + dark outline.
- `comfy.py` — generalized client: health-check, UNET auto-swap, submit, poll,
  view/download (temp+output types), multipart upload (server keeps original
  filename → rename/copy before upload), run() = full cycle.
- `statelog.py` — cell-delta log (P-mode PNG per changed cell per step) +
  snapshots (npy); `latest_snapshot(root)` for resume.
- `metrics.py` — changed_cells, neighbor_dist (Euclidean over full tile),
  palette_entropy (bits), uniformity (max identical cells / total), CSV.
- `pipeline.py` — Engine.step(): schedule → frozen snapshot → per-cell
  rule+generation (ThreadPool, `concurrency`) → deterministic in-schedule-order
  write-back → log/snapshot/metrics. Generators: `ComfyGenerator` (uploads
  nearest-upscaled refs, per-arm node overrides incl. size + seed) and
  `FakeGenerator` (deterministic on-distribution tiles for pipeline tests).
- `applet/rule/` — contract (RuleRequest/RuleResult), prompt builder
  (frame/semantics/relation dial/arm style/≤500-char contract), backends:
  template (deterministic), llamacpp (OpenAI-compatible multimodal),
  pi_agent (headless `pi -p --no-tools ...` — in this container the rule model
  IS the dev model; keep concurrency 1 with it).
- `applet/run.py` — `python -m applet.run --config x.yaml --steps N --headless`
  (headless is a no-op flag for now). Field init: resume snapshot > initial
  field PNG > random.

Verified end-to-end offline: 3 steps ring_scan K=4 (log/snapshot/metrics all
correct) + clean resume from snapshot.

## Build-phase gotchas (learned the hard way, 2026-10-17)

- **int16 overflow in palette math**: (255−0)² = 65025 > int16 max → negative
  "distances" and silently scrambled argmin. Use int32.
- **Pillow `np.asarray(Image)` is read-only** (Pillow ≥ ~11): build numpy
  arrays directly or `.copy()`; `Image.fromarray(..., mode="RGBA")` for pastes.
- **Block grouping reshape**: `idx.reshape(A, b, A, b).reshape(A*A, b*b)` mixes
  pixels of different blocks (layout is (a, x, b, y), not (a, b, x, y)).
  Transpose to (A, B, b, b) first: `.transpose(0, 2, 1, 3)`.
- Node 13 (ResolutionSelector) in t2i.json wants `aspect_ratio: "1:1 (Square)"`
  (string!) + `megapixels` float ((size/1024)²) + `multiple: 8`; node 488's
  width/height are *links* from 13 — override 13, don't break the links.
- edit.json `508` switch=true uses `505` (fixed-size latent) — set 505's
  width/height for write_size; leave the switch alone.

## Stage plan (agreed with Felix, 2026-10-17)

Stage 1 = **engine core first**: headless modules in `applet/engine/` +
`applet/rule/` with unit tests, no remote generation yet. ✅ done.
Next: experiments E1–E7 (SPEC §6) — E1 (VAE alignment probe) first, needs
`workflows/t2i_ca.json` copy. Then GUI (toolkit still unconfirmed, PySide6
suggested).
