# SD-CA — Spec

A hybrid **LLM / diffusion / cellular-automaton** system: a mosaic field of small
pixel-art cells that evolves in lockstep. Each cell's state is a tiny image; the
CA update rule is an LLM that reads a cell's neighborhood and writes a prompt;
the Qwen-Image-2.1 diffusion model (via remote ComfyUI) renders the cell's new
state.

Status: planning complete, build phase. This document is the authoritative design.
`base_workflows/OBSERVATIONS.md` holds the empirical model/workflow notes the
design relies on.

---

## 1. Concept

- **Space:** a grid of `N×N` cells (default `N = 16`). The *art field* is
  `N × art_size` px (default 16 × 32 = **512×512**); the *display field* is the
  art field scaled by `render_scale` (default 2× → **1024×1024**).
  (Grid 32×32 of 32-art cells is an open variant — same display field, 4× the
  cells; keep grid size in config.)
- **State:** each cell's state is a **32×32 pixel-art image** (default; 64 and
  16 are experiment values), quantized to a **global 64-color palette**.
  The 32×32 art image *is* the ground truth — not a render artifact.
- **Topology:** 4-neighborhood (N/E/S/W) + self = 5 images. **Borders wrap.**
- **Dynamics:** lockstep steps. Each step:
  1. schedule policy selects a **subset** of cells (size `K`, config; policy
     module TBD — see §2.2),
  2. a snapshot of the field is frozen,
  3. for each selected cell, the **rule** (LLM) reads the 5 neighborhood tiles
     from the snapshot and emits one generation prompt,
  4. **ComfyUI** renders the new tile (one of the generation arms, §2.4),
  5. the output is block-quantized down to the art grid and written back;
  6. step complete → renderer shows the patch, video writer emits a frame,
     metrics logged.
- **Governor:** a local Python **applet** owns the grid, the schedule, all
  ComfyUI plumbing, the state log, the renderer, and the video. The LLM is a
  *pure stateless rule function* invoked per cell; it holds no state and owns
  no plumbing (modular backends, §2.3 — swappable so an agent-governed variant
  can be revived later).

---

## 2. Components

### 2.1 Applet (Python, runs on the user's local machine)

- **Engine** (GUI-agnostic, importable, headless-testable in the container):
  grid state, schedule policy, neighborhood render, upscale/downscale +
  quantize, ComfyUI client, rule client, state log, metrics.
  `python -m applet.run --headless ...` must work with no display.
- **GUI** (desktop, local machine; toolkit suggestion: PySide6 — confirm with
  user): mosaic view patched cell-by-cell as updates land within a step;
  controls: start / pause / single-step, arm, write_size, read_size, K,
  seed mode, schedule, per-run instruction text, palette; metrics panel
  (sparklines, collapse warning).
- **Video writer** (§2.7).

Remote dependencies (all by config URL): ComfyUI server(s), rule-LLM server.
Everything else is local.

### 2.2 Schedule policy

Pluggable module implementing `select(field, step) -> list[(row, col)]`.
**TBD by user** — spec provides the interface plus simple defaults to build
against: `random_uniform(K)`, `ring_scan(K)` (moving band), `all` (K = N²,
slow). K is the main speed dial (step time ≈ K × (rule + generation)).

### 2.3 Rule backends (modular)

Contract:

```
RuleRequest:
  tiles: {center, north, east, south, west}  # image paths, nearest-upscaled to read_size
  step: int
  cell: (row, col)
  arm: "t2i" | "edit_self" | "edit_neighborhood"
  frame: str        # pixel-art framing text, injected by the applet
  instruction: str  # optional per-run evolution instruction from the GUI
RuleResult:
  prompt: str       # exactly one prompt string (~<=500 chars)
  reasoning: str    # optional; logged only
```

Backends:
- **`llamacpp`** — direct HTTP to a llama.cpp server (OpenAI-compatible
  `/v1/chat/completions`, multimodal Qwen3.8-27B; endpoint in config, confirm
  with user). Default target.
- **`pi_agent`** — headless pi invocation, constrained: single multimodal
  model call, fixed system prompt, no tools, strict output format.
- **`template`** — deterministic canned prompts; for pipeline tests without an
  LLM.

The **system prompt** encodes the CA's character and is the main place
dynamics are tuned. It must contain:
- the pixel-art frame: "you are describing a {art_size}×{art_size} pixel-art
  tile: chunky shapes on a coarse grid, limited {palette_n}-color palette,
  flat colors, no gradients, no fine detail" (+ exact palette when fixed),
- the neighborhood semantics: which tile is self vs. which is N/E/S/W, and
  how the cell should relate to them (this is the actual rule behavior —
  evolve-with-neighbors vs. contrast-vs-neighbors etc.),
- arm-specific prompt style: `t2i` → full descriptive prompt of the new tile;
  `edit_self` / `edit_neighborhood` → edit *instruction* (what changes, what
  is kept).

### 2.4 Generation arms (all three are developed)

All arms share: write at `write_size` (default **512**; test 1024, 256),
turbo-8-step sampling exactly as in the base workflows (do not "fix"), then
**block-quantize** to the art grid (below).

| Arm | Diffusion sees | Prompt type | Notes |
|---|---|---|---|
| `t2i` | nothing (text only) | full description | lineage lives only in the LLM's words; strongest evolution |
| `edit_self` | the cell (upscaled) | edit instruction | `image_1` = nearest-upscale(self → write_size); continuous local transitions |
| `edit_neighborhood` | cell + 4 neighbors | edit instruction | `image_1` = self, `image_2..5` = N/E/S/W; `TextEncodeQwenImage21` supports up to 16 references; rule becomes partly sub-linguistic |

**Upscale for editing:** nearest-neighbor art → write_size. A 32×32 tile
upscaled 32× is exactly "32×32 pixel art at 1:1 art-pixel scale", i.e.
on-distribution for the pixel-art frame in the prompt.

**Downscale (write → ground truth):** output write_size² → art_size² by
taking, per `write_size/art_size` px block, the **mode color over quantized
colors** (quantize block pixels to the global palette first; median as
fallback; **never a plain mean** — means blend edges into mud).

**Workflow copies** (in `workflows/`, never the originals):
- `t2i_ca.json` ← `base_workflows/t2i.json`; prompt override node `497`;
  output size via node `13` (or `488` directly).
- `edit_ca.json` ← `base_workflows/edit.json`; single-reference form (per
  OBSERVATIONS: for one image, delete `475` + `515`, drop `images.image_2`
  from `510`, drop `image` input from `516` — the with-PE variant is not
  needed since the CA agent replaces the enhancer). Output latent: set node
  `505` (or switch `508` → false) to write_size.
- `edit_neigh_ca.json` ← as above but wire **five** reference slots
  (`image_1..image_5` on `510`) and five `LoadImage` nodes.

### 2.5 State & persistence

- **Ground truth:** per-cell art image, stored palette-indexed (or PNG) at
  art_size².
- **State log (the cell-delta store, authoritative record):** one entry
  `(step, cell, png)` **only when a cell changes**. This is the lossless,
  disk-efficient archive and the source for video reconstruction.
- **History policy (config):** `none` | `ring(N)` per cell | `full`. Default:
  state log + current snapshot (ring 16 suggested once disk is observed).
- **Resumable:** current field snapshot + log ⇒ applet restarts cleanly.

### 2.6 Neighborhood render (for the rule LLM)

The 5 tiles are nearest-upscaled to `read_size` (default **512**; test 256,
1024 — must be a multiple of 32) and composed as a **labeled cross layout**
(N/E/S/W + C, labels drawn into the composite so the LLM can't misassign
positions). Per-tile visual-token cost at read_size (Qwen3.8-27B: 32×32 input
px per visual token): 256² → 64, 512² → 256, 1024² → 1024. A 512² composite
≈ 1280 visual tokens per rule call.

Reading is more robust than writing at a given size (per our model notes), so
read_size and write_size are independent config values.

### 2.7 Video output

- The **state log is the master** (lossless, cell-delta). Video is a *derived*
  artifact: stitch display-field frames from the log → encode with ffmpeg.
  - preview: HEVC (or H.264), CRF ~20, `fps` = 1 frame per step (or stretched);
  - optional lossless master: FFV1, only on demand.
- Frame resolution = display field (1024×1024 at render_scale 2).

### 2.8 Metrics & collapse watch

Per step, logged to CSV + GUI panel:
- `changed_cells` (count),
- `neighbor_dist`: mean color-distance over all adjacent pairs,
- `palette_entropy`: Shannon entropy of the global palette-index histogram,
- `uniformity`: fraction of cells equal to the most common cell.

Attractor collapse is the main qualitative risk (the model has known
attractors — see OBSERVATIONS — and a CA can homogenize onto one). Warning at
configurable `uniformity` threshold. The rule's system prompt can carry
local-diversity pressure ("stay visually distinct from the four neighbors")
as a mitigation dial.

---

## 3. Configuration (everything a knob)

| Key | Default | Notes |
|---|---|---|
| `grid_n` | 16 | cells per side; 32 is an open variant |
| `art_size` | 32 | ground-truth tile size; test 64, 16 |
| `render_scale` | 2 | display = art × scale (default display 1024²) |
| `palette_n` | 64 | global fixed palette; source: fixed file (dev picks, E5) or `kmeans(init)`. E4 (2026-10-17): the random dev stand-in is far from the model's color space (quantize loss ~41/51) — build the initial palette **from the model's outputs**, not by hand |
| `write_size` | 512 | generation size; default set 2026-10-17 from E2/E4 (flatness, alignment, fidelity, latency all peak at 512); test 1024, 256 |
| `read_size` | 512 | neighborhood-render size for the rule LLM; multiple of 32 |
| `arm` | `edit_self` | `t2i` \| `edit_self` \| `edit_neighborhood`; can be per-cell for experiments |
| `schedule` | `random_uniform` | policy module name |
| `K` | 16 | cells per step |
| `seed` | `random` | or `deterministic(cell, step)` for reproducible experiments |
| `rule_backend` | `llamacpp` | `llamacpp` \| `pi_agent` \| `template` |
| `history` | `log+snapshot` | `none` \| `ring(N)` \| `full` |
| `concurrency` | 4 | parallel rule/generation dispatch (server serializes anyway) |
| `comfy_url` | from skill | 8188 (5090/32GB) or 8189 (4080/16GB); health-check `GET /models/diffusion_models` each run |
| `llm_url` | TBD (confirm) | llama.cpp endpoint for Qwen3.8-27B |

---

## 4. Model facts (researched 2026-10; verify E1 before relying on alignment)

- **Qwen-Image-2.1 VAE: 16× spatial compression** (secondary source:
  vllm-omni recipe; probe in E1). Latent = size/16; 2×2 token packing ⇒
  tokens = (size/32)².
- **Effective art ceiling = write_size / 16** (one art pixel ≈ one latent
  cell is the resolution limit): 1024 → 64 art, 512 → 32 art, 256 → 16 art.
  For a 32-art target: **512² is alignment-matched (1 art px = 1 latent),
  1024² is 2× oversampled, 256² caps at 16-art** (the model really paints
  16×16 and interpolates).
- No hard local minimum; dimensions in multiples of 32. The hosted-API floor
  (512×512 total pixels) is the vendor's de facto supported minimum.
- `TextEncodeQwenImage21`: `resolution` 0..4096 step 32, `0` = keep reference
  size (rounded to /32); **up to 16 reference images** (`image_1..16`), seen
  by the VL encoder and spliced in as VAE latents; output latent = first
  reference's size (1024² if none) — "any other size shifts the edit".
- Base Qwen-Image comfort band reported as ~1.5M–17M px; 1MP (1024²) is the
  well-trodden point; the base workflows run turbo-8-step (8 steps, cfg 1,
  euler/simple) — do not change sampling.
- **Rule-LLM (Qwen3.8-27B multimodal, llama.cpp):** 32×32 input px per visual
  token (16-px patches, 2×2 merge); min image 256×256 (64 tokens); smaller
  inputs get bicubic-upscaled by the processor (blurry) — always
  nearest-upscale tiles ourselves to a multiple of 32. Reading is more
  reliable than writing at a given size ⇒ read_size may exceed write_size
  capability (independent knobs).
- Latency budget (16GB server, turbo 8-step): ~10–25 s per 1024² gen
  (4MP ≈ 50 s measured), plus rule-LLM time. Step time ≈ K × (rule + gen).

---

## 5. Repository layout

```
sd-ca/
  AGENTS.md            # build-phase operating instructions
  SPEC.md              # this document
  base_workflows/      # READ-ONLY (t2i.json, edit.json, *_with_pe.json, OBSERVATIONS.md)
  workflows/           # working copies: t2i_ca.json, edit_ca.json, edit_neigh_ca.json
  comfy/               # comfy.py (reference client) -> generalized client module
  applet/
    engine/            # grid, state, schedule, pipeline, metrics  (headless, testable)
    rule/              # backends: llamacpp.py, pi_agent.py, template.py
    gui/               # desktop renderer (PySide6 suggested)
    video/             # state log -> frames -> ffmpeg
  config/              # yaml/json configs per run
  state/               # logs, snapshots (gitignored)
  images_out/          # generations, experiment outputs
  notes/               # experiment results (E1–E7) with linked images
```

---

## 6. Development experiments (run early, record in `notes/`)

| # | Experiment | Question / acceptance |
|---|---|---|
| E1 | **VAE alignment probe** | checkerboard prompt at 1024²: do block edges land on 16-px multiples? Confirms the 16× factor and the alignment argument. 5-minute test. |
| E2 | **Write-size matrix** | {1024, 512, 256} × {t2i, edit_self, edit_neighborhood}: post-quantize 32-art fidelity, style adherence, latency. Expectation: 512 ≥ 1024 quality-per-second; 256 capped at 16-art. |
| E3 | **Read-size matrix** | {256, 512, 1024} neighborhood renders: does the rule correctly identify and describe neighbor relations? (Have it answer fixed probe questions about known 5-tile composites.) |
| E4 | **Pixel-art frame adherence** | does the model actually emit blocky grid-aligned art at each write size with the frame prompt? Measure blockiness before quantize. |
| E5 | **Palette** | candidate 64-color palettes (fixed files + kmeans from init): aesthetic coherence across 256 cells; state-log compression. Pick default. |
| E6 | **Collapse watch** | 20–50 lockstep steps, K=16, each arm: track §2.8 metrics; does uniformity rise? Test diversity-pressure prompt dial. |
| E7 | **Pixel-art LoRA** | does one exist for Qwen-Image-2.1? If so, A/B in E2. |

---

## 7. Non-goals (for now)

Real-time operation; multi-server load balancing; rule self-modification;
in-GUI manual cell steering (post-MVP); transparency/RGBA cells; non-square
cells.

---

## 8. Open decisions (flagged to user, defaults proposed)

1. **Schedule policy** — TBD by user; build against `random_uniform(K=16)` default.
2. **Grid 16×16 vs 32×32** (at 32-art) — default 16; 32 gives 4× cells, same display field.
3. **Palette source** — fixed 64-color file (dev picks a candidate, E5) vs kmeans from initial field.
4. **Initial field** — default: one coherent pixel-art generation covering the
   whole 512² art field, tiled; alternatives: N² independent t2i tiles, user-supplied.
5. **Rule-LLM endpoint** — llama.cpp URL (confirm with user).
6. **GUI toolkit** — PySide6 suggested (confirm).
