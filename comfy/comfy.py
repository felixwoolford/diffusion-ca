#!/usr/bin/env python3
"""Submit a ComfyUI API-format workflow, poll for completion, download outputs.

Usage:
  python3 comfy.py WORKFLOW_JSON SERVER [PROMPT_JSON] [OUTPUT_DIR]

PROMPT_JSON: optional JSON mapping of node_id -> {input_name: value} overrides,
e.g. '{"491": {"prompt": "a red fox in snow"}}'.

Model handling: if the UNETLoader's unet_name is not available on the server,
it is replaced with the qwen_image_2.1 file the server does have (bf16 preferred).
"""
import json, sys, time, urllib.request, os

def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)

def post(url, payload, ctype="application/json"):
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"[http {e.code}] {body[:2000]}")
        raise

def main():
    wf_path, server = sys.argv[1], sys.argv[2].rstrip("/")
    overrides = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    out_dir = sys.argv[4] if len(sys.argv) > 4 else "images_out"
    os.makedirs(out_dir, exist_ok=True)

    graph = json.load(open(wf_path))

    # ensure UNET matches what the server has
    models = get(f"{server}/models/diffusion_models")
    for node in graph.values():
        if node["class_type"] == "UNETLoader":
            name = node["inputs"]["unet_name"]
            if name not in models:
                cand = sorted([m for m in models if "qwen_image_2.1" in m],
                              key=lambda m: (m != "qwen_image_2.1_bf16.safetensors", m))
                node["inputs"]["unet_name"] = cand[0]
                print(f"[note] unet_name -> {cand[0]} (server has {models})")

    for nid, inputs in overrides.items():
        graph[str(nid)]["inputs"].update(inputs)

    res = post(f"{server}/prompt", {"prompt": graph})
    pid = res["prompt_id"]
    print(f"[submitted] {pid}")

    deadline = time.time() + 1800
    while time.time() < deadline:
        h = get(f"{server}/history/{pid}")
        if pid in h:
            outputs = h[pid].get("outputs", {})
            for nid, out in outputs.items():
                for img in out.get("images", []):
                    if img.get("type") in ("temp", "output"):
                        fn = img["filename"]
                        url = f"{server}/view?filename={fn}&subfolder={img.get('subfolder','')}&type={img['type']}"
                        local = os.path.join(out_dir, fn)
                        with urllib.request.urlopen(url, timeout=60) as r, open(local, "wb") as f:
                            f.write(r.read())
                        print(f"[saved] {local}")
            status = h[pid].get("status", {})
            print(f"[done] status: {status.get('status_str')}")
            if status.get("status_str") != "success":
                print(json.dumps(h[pid], indent=2)[:3000])
            return
        time.sleep(2)
    print("[timeout]")

if __name__ == "__main__":
    main()
