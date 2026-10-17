---
name: comfyui
description: Generate or edit images with Qwen-Image-2.1 on the remote ComfyUI server. Use for any image generation, editing, style transfer or transparent-image task.
---
API: POST /prompt with API-format graphs; poll GET /history/<id>; fetch via GET /view; upload references via POST /upload/image; node schemas at GET /object_info; free VRAM via POST /free.
Known-good graphs: base_workflows/t2i.json, base_workflows/edit.json (API format). Modify these rather than building from scratch.

Model: Qwen-Image-2.1 (one model for generation and editing).
- Files: qwen_image_2.1_bf16 / qwen3vl_8b_int8_convrot (CLIPLoader type qwen_image) / qwen_image_2.1_vae_bf16
- Sampling: euler, simple, cfg 1, ~40 steps. Turbo LoRA (LoraLoaderModelOnly between UNETLoader and KSampler): 8 steps, cfg 1.
- Editing: image_1 = image being edited (sets composition and output size); image_2+ = content/style references. Refer to them in prompts as <image1>, <image2>. State what to keep from each and what to take from each.
- Transparency: wrap the prompt as "This is an RGBA format image with transparency. [...]. The image has an alpha channel and a transparent background." Save as PNG.


## Servers and diffusion model

Servers (each reached over the user's SSH tunnel):
- http://127.0.0.1:8188 — (5090 - 32GB): qwen_image_2.1_bf16.safetensors
- http://127.0.0.1:8189 — (4080 - 16GB): qwen_image_2.1_int8_convrot.safetensors
At any time, it is likely that only one of these will be available.

The workflow JSONs reference qwen_image_2.1_bf16.safetensors. Before submitting to any server:
1. GET /models/diffusion_models on that server.
2. If the UNETLoader's unet_name isn't in that list, replace it with the qwen_image_2.1 file the server does have (prefer bf16, then int8_convrot).
Change nothing else in the graph; the text encoder, VAE and LoRA filenames are identical on all servers.

