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

**Built copies (2026-10-17, see OBSERVATIONS revision):**
- `workflows/t2i_ca.json` = byte-identical copy (nothing to change for direct t2i).
- `workflows/edit_ca.json` = edit.json minus node `475` and `510.images.image_2`
  (single-reference form); slot `image_1` → `470`.
- `workflows/edit_neigh_ca.json` = edit.json plus new `LoadImage` nodes `471`/`473`/`474`
  wired as `510.images.image_3`/`4`/`5` (five-reference form);
  slots: `image_1→470, image_2→475, image_3→471, image_4→473, image_5→474`
  (`472` is taken by ImageCompare, so slot ids are not contiguous).

## Environment

- **Rule LLM = the dev model's multimodal variant.** `PI_MODEL=
  qwen3.8-27b-ud-q6-k-xl`, `PI_PROVIDER=llama.cpp` (env vars in the container).
  Felix (2026-10-17): rule calls should target this same model; it's RAM-cached
  and cheap, but **cannot run in parallel with the dev agent's own work** →
  when `rule_backend: pi_agent`, use `concurrency: 1` (one rule call at a time,
  sequenced, not overlapped with other model work).
- **llama.cpp router at `http://127.0.0.1:8080`** (checked 2026-10-17 ~17:00 UTC,
  `GET /v1/models`, nonstandard `status.loaded` field per model):
  - `qwen3.8-27b-ud-q6-k-xl` — LOADED; this is the dev model. It **refuses image
    input** (text-only deployment; a test `pi -p` with an @image asked Felix to
    switch models). So `pi_agent` with the default model CANNOT read tiles.
  - `qwen3.8-27b-ud-q6-k-xl-multimodal` — **LOADED since 2026-10-17 ~18:30 UTC**
    (Felix loads it on demand; it is slower and has a smaller context than the
    text deployment — keep calls minimal). No auto-load and no load endpoint
    from inside the container (400 "model is not loaded"; probed /api/models,
    /models/load, ...). `llamacpp` backend config:
    `llm_url: http://127.0.0.1:8080`, `llm_model: qwen3.8-27b-ud-q6-k-xl-multimodal`.
  - **It is a *thinking* model.** Responses carry `reasoning_content`; without
    `"chat_template_kwargs": {"enable_thinking": false}` in the chat payload
    the entire max_tokens budget goes to reasoning and the visible `content`
    is empty (finish_reason=length). The engine `llamacpp` backend sets this
    (E3, 2026-10-17; regression-tested). `pi_agent` unaffected.
  - **Image cap: 4096 visual tokens (2048×2048) PER IMAGE.** Larger images
    are downscaled server-side (verified via prompt_tokens in E3: a 3072²
    composite is billed as exactly 4096 visual tokens). So read_size 1024
    *composites* (3072²) are served at ~683 px/tile — but the *separate*
    render mode (five one-tile images) sends five 1024² images, each under
    the cap: true 1024 px/tile at 5×1024 = 5120 visual tokens total (E3
    separate probe).
  - pi's model catalog (`/root/.pi/agent/models.json`) already knows the
    multimodal id (input: text+image, temp 0.7), so `pi_agent` could use
    `--model qwen3.8-27b-ud-q6-k-xl-multimodal` once it is loaded.
  - Also listed (unloaded): gemma-4-31b-it variants.
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
  - **Dual render mode (2026-10-17, Felix's original design):**
    `rule_render: composite | separate` (SPEC §2.6). composite = one labeled
    cross image (3×read_size)²; separate = five one-tile images, order fixed
    in text (1=C, 2=N, 3=E, 4=S, 5=W — `SEPARATE_ORDER`/`SEPARATE_INTRO` in
    `rule/prompts.py`, the single source of truth for both backends). E3:
    both 100% on probe questions; separate is cheaper at 512 (1280 vs 2304
    visual tokens — the composite pays 1024 for its four empty corners) and
    the only uncapped path to 1024. Default stays `composite` pending the
    behavioural comparison (E6). pi_agent separate mode: five @files; pi's
    docs don't document exact part ordering, but the system prompt fixes the
    image order so either layout is unambiguous (untested live — the
    multimodal model is only loaded on demand).
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

## Palette: the dev64 stand-in is far from the model's color space (E4 finding, 2026-10-17)

Felix (2026-10-17): **this finding is very important — keep it in mind for
E5 and for reading every fidelity number until then.**

- Quantize loss (max-channel RGB distance to the nearest `palette_dev64.json`
  color) is **~41 mean / 51 p95 at every write size** (E4), and essentially
  **no output pixel is within 16 RGB of a dev-palette color**. The model's
  natural flat colors sit systematically far from the random stand-in.
- Consequence: post-quantize "exact match" scores (E2) partly measure palette
  mismatch, not just generation quality; the **near-match (within 64) column
  is the more honest metric** until the real palette exists.
- **E5 must build the real 64-color palette from the model's own outputs**
  (kmeans over a corpus of E1–E4-style generations — the SPEC's
  `kmeans(init)` source option), not by hand. Expected effects: quantize loss
  collapses toward ~0 for flat blocks, and per-seed color-identity drift
  (E2 follow-up: green `[22,198,44]` → adjacent `[49,216,18]`) maps onto the
  same/adjacent swatch instead of a "wrong color".
- **Mechanism (E2 splotch diagnosis, 2026-10-17):** the damage is not a
  color shift but *instability*. t2i_1024's green `[35,153,42]` (std ≈ 2,
  visually flat) sits 46.9/46.9/48.1 from three dev-palette entries — the
  argmin boundary passes through it, so per-pixel nearest flips 63/31/6%
  among them and the block-mode ground truth becomes a 3-color art-px
  patchwork (the "splotch" Felix saw in `t2i_1024/quantized_32.png`).
  Swapping one slot to the model's actual green → 99.8% uniform. The edit
  arms don't show it (they copy palette-made references → 100% single
  index). The quantize algorithm itself (SPEC §2.4: nearest-then-mode) is
  correct. → E5 candidate-palette scoring must include **quantize stability**
  (per-pixel nearest-vs-second-nearest margin over a real-output corpus),
  not just mean loss.

## Stage plan (agreed with Felix, 2026-10-17)

Stage 1 = **engine core first**: headless modules in `applet/engine/` +
`applet/rule/` with unit tests, no remote generation yet. ✅ done.
Stage 2 = **experiments E1–E7 (SPEC §6), approved 2026-10-17**.
Status 2026-10-17: **E1 ✅** (16× VAE factor confirmed), **E2 ✅** (write-size
matrix; edit arms ≈ lossless copy channel; 512 sweet spot), **E3 ✅** (rule
reads composites 28/28 at all read sizes; composite 1024 capped at ~683
px/tile by the per-image 4096-visual-token cap; **separate reads — Felix's
original design — 14/14 at 512, cheaper and uncapped at 1024**; both modes
now in the engine via `rule_render`), **E4 ✅** (frame holds; 512 flattest;
dev64 palette far from model color space). **E5 pending** (palette from model
outputs, incl. a quantize-stability metric), **E6 pending** (collapse watch),
**E7 pending** (pixel-art LoRA A/B). Felix runs experiments one at a time with
explicit go-ahead per experiment (2026-10-17: "don't proceed with any other
experiment yet" until told). Experiment scripts live in `experiments/`;
results + evidence in `notes/E*.md`. Then GUI (toolkit still unconfirmed,
PySide6 suggested).
