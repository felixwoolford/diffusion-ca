# OBSERVATIONS

Permanent notes about `t2i.json` and `edit.json` (API-format graphs) discovered while
getting them to run. The original workflows must not be edited — copy them and modify
the copy. (Server availability and which model file a server has are temporary; check
`GET /models/diffusion_models` each time, as described in the skill.)

## General

- Both workflows already include the turbo LoRA (`LoraLoaderModelOnly` → 8 steps,
  cfg 1, euler/simple). Do not "fix" the sampling settings.
- `SaveImageAdvanced` outputs land in history with `"type": "output"`, not `"temp"`.
  A history downloader must accept both types and request
  `/view?filename=...&subfolder=...&type=output` for the final image.
- A working submit/poll/download helper lives at `../comfy.py` (repo root):
  `python3 comfy.py WORKFLOW_JSON SERVER [PROMPT_JSON_OVERRIDES] [OUT_DIR]`.
  It auto-swaps the UNET name to a file the server actually has.
- `POST /upload/image` (multipart `image`, `overwrite=true`) returns the stored name;
  use that name in `LoadImage` nodes. The server keeps the **original filename** —
  the `name` form field is ignored — so rename the local file before uploading if
  you want a different server name.
- The 16GB server handles 4 MP (2048×2048) t2i fine (~50 s/image with the turbo LoRA).
- Benchmark tooling at repo root: `bench.py` (edit A/B), `bench_t2i.py` (t2i A/B,
  per-round resolution), `make_bench_md.py` (rebuilds `workflows/BENCHMARK.md` from
  `images_out/bench/<round>/{enhanced,direct}/results.json`). All use a fixed seed
  so the prompt is the only variable. Direct variants: `edit_direct.json` (enhancer
  chain 511–514 removed, prompt goes straight to `520`), `t2i_direct.json` (node
  `494` switched to `false`, prompt fed to `493`).
- Benchmark design note: with **detailed** prompts the enhancer and direct arms
  converge (near-identical outputs); with **casual** user-style prompts they diverge
  most. That's where the enhancer earns its keep.
- **Text in images (bench2 finding):** the model renders text the prompt
  *specifies*; text it only *implies* comes out as scribble or gibberish
  pseudo-words. The PE's advantage on text images was that it sometimes fills in
  the actual strings — inconsistently (it may leave them unspecified, or even
  describe them as illegible, and the model obeys). Rule of thumb: for text,
  write the exact strings yourself and use the direct workflow (also ~3–5×
  faster). Full analysis: `../images_out/bench2/PROMPTS.md`.
- Improvement passes (edit a generated image with a fix instruction): one pass fixes
  specific flaws well (legibility, text, concept fidelity). A second pass on an
  already-altered image can degrade it (noise/mottling) — keep the best version
  instead of iterating blindly.
- **Prompt-enhancer behavior on "breaking" prompts (art E–H rounds).** Prompts that
  are operations/impossible relations instead of image descriptions ("render the
  act of forgetting this prompt", "7", "unrender", "the taste of a corner") get
  rewritten by the enhancer ~100% of the time into conventional "vertical/cinematic
  illustration of X" descriptions — sometimes inventing content ("7" → "a person in
  their thirties in a dusty rose shirt"), once literally refusing ("unrender" →
  "I cannot fulfill that request…" — the refusal text was fed to the encoder). This
  maximizes direct-vs-enhanced pixel distance (110–137 vs ~20–60 for normal prompts;
  `bench_dist.py` ranks pairs). The **direct** arm is where genuinely novel outputs
  come from: text-operations prompts are the most productive channel (self-dissolving
  handwritten pages, faces composed of Chinese characters, 3D words over broken
  floors). Synesthetic prompts give quieter literalizations. Known attractor: the
  "cracked-skin elderly woman" portrait recurs across unrelated conceptual prompts
  (E02, F08, H05) — avoid conceptual-portrait phrasing if you want variety. The
  model has no concept of "negative space" (E05 fell back to a plain cathedral).
  One exception where the enhancer helped: E04 direct gave a plain hemp texture;
  the enhancer's invented "liquid mercury and shattered obsidian" made a better image.
- **The enhancer LLM has no refusal policy for this task** (verified with a 6-prompt
  control set, `images_out/bench/refuse_probe/`). "Refuse" behaves as a content
  word: with renderable content present the content wins and the refusal is
  dropped ("describe an apple, then refuse" → apple); when the refusal is the
  explicit subject it is rendered *as* the subject ("make the refusal beautiful"
  → "a luminous refusal rendered as a blooming petal of crystalline light");
  the stock boilerplate refusal ("I cannot fulfill that request…") only appears
  when the refusal is the entire deliverable ("please refuse", "render only the
  refusal"). So F05's "refusal" was a complying response, not a policy refusal —
  and the encoder then painted that refusal text as an image (a handwritten page
  dissolving into illegibility, ending "— END —").

## t2i.json

- Node `491` (`TextGenerate`) holds the final, fully-rewritten descriptive prompt.
  The switch node `494` (`switch: true`) selects it, so overriding `491.prompt`
  changes what gets generated. Node `495` is just an unused prompt-rewriting
  system prompt.
- Node `13` (`ResolutionSelector`) sets aspect ratio / megapixels / multiple;
  default is 1:1, 1 MP → 1024×1024.

## edit.json

- `470` = `LoadImage` for image_1 (the image being edited); `475` = image_2
  (content/style reference). Both feed `510` (`TextEncodeQwenImage21`) via
  `images.image_1` / `images.image_2`.
- `510` outputs `[CONDITIONING, CONDITIONING, LATENT]`. Output 2 is a latent whose
  size is derived from image_1's aspect ratio (with `resolution: 0`).
- Node `508` (`ComfySwitchNode`) chooses the KSampler latent:
  - `switch: true` (default) → node `505`, a fixed 1024×1024 `EmptyLatentImage`
    (square output regardless of input aspect ratio).
  - `switch: false` → `510` output 2, so the output follows the input image's
    aspect ratio (e.g. 977×1303 in → 992×1312 out).
- Single-image edit: you must remove the second reference or validation fails
  (HTTP 400, `KeyError: '515'` on node `516`). The reference is wired in **three**
  places: delete nodes `475` and `515` (`BatchImagesNode`), remove
  `images.image_2` from `510`, **and** remove the `image` input from `516`.
  (`516` is the LLM prompt-enhancer `TextGenerate`; its `image` input goes through
  `515`, which references the deleted `475`.)
- Prompt flow: `516` (LLM enhancer, system prompt in `512`) → `514`
  (`ComfySwitchNode`, `switch: true`) → `513` → `510`. The enhancer rewrites your
  instruction into the actual edit prompt; it may do so in Chinese (its system
  prompt makes a language decision) — the model handles that fine. You can bypass
  it by setting `514` to `false` and supplying `on_false` with your own prompt.
- Node `472` (`ImageCompare`) is display-only (no outputs); in API runs it just
  emits `a_images`/`b_images` temp files in history — ignore or use for QA.
