"""Generalized ComfyUI client (from comfy/comfy.py reference, which stays as-is).

Covers everything the engine needs: health-check + UNET auto-swap, submit,
poll, download (accepting both `temp` and `output` image types — see
base_workflows/OBSERVATIONS.md), and reference-image upload. No workflow
structure is hardcoded here; per-arm node wiring lives in Config.arms.

Servers: http://127.0.0.1:8188 (5090/32GB) / http://127.0.0.1:8189 (4080/16GB)
via SSH tunnels; usually only one is up — check `models()` each run.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


class ComfyError(RuntimeError):
    pass


def _multipart(fields: dict[str, str], file_field: str, filename: str,
               file_bytes: bytes, file_type: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
             f"filename=\"{filename}\"\r\nContent-Type: {file_type}\r\n\r\n").encode()
    body += file_bytes + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


class ComfyClient:
    def __init__(self, url: str, timeout: int = 30):
        self.url = url.rstrip("/")
        self.timeout = timeout

    # -- low-level ------------------------------------------------------- #
    def _get(self, path: str):
        with urllib.request.urlopen(f"{self.url}{path}", timeout=self.timeout) as r:
            return json.load(r)

    def _post(self, path: str, payload, ctype: str = "application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.url}{path}", data=data,
                                     headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise ComfyError(f"POST {path} -> HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}") from None

    # -- model / graph --------------------------------------------------- #
    def models(self) -> list[str]:
        """Health check: diffusion models on the server (call each run)."""
        return self._get("/models/diffusion_models")

    def swap_unet(self, graph: dict, models: list[str] | None = None) -> dict:
        """Replace UNETLoader names the server lacks with an available
        qwen_image_2.1 file (bf16 preferred, then int8_convrot)."""
        models = models if models is not None else self.models()
        g = json.loads(json.dumps(graph))
        for node in g.values():
            if node["class_type"] == "UNETLoader":
                name = node["inputs"]["unet_name"]
                if name not in models:
                    cand = sorted((m for m in models if "qwen_image_2.1" in m),
                                  key=lambda m: (m != "qwen_image_2.1_bf16.safetensors", m))
                    if not cand:
                        raise ComfyError(f"no qwen_image_2.1 model on server (has {models})")
                    node["inputs"]["unet_name"] = cand[0]
        return g

    @staticmethod
    def load_graph(path: str | Path) -> dict:
        return json.loads(Path(path).read_text())

    # -- submit / poll / download ---------------------------------------- #
    def submit(self, graph: dict, overrides: dict[str, dict] | None = None) -> str:
        """POST /prompt. overrides: {node_id: {input_name: value}} applied on a copy."""
        g = json.loads(json.dumps(graph))
        for nid, inputs in (overrides or {}).items():
            if str(nid) not in g:
                raise ComfyError(f"override node {nid} not in graph (has {sorted(g, key=int)})")
            g[str(nid)]["inputs"].update(inputs)
        res = self._post("/prompt", {"prompt": g})
        if "prompt_id" not in res:
            raise ComfyError(f"submit failed: {json.dumps(res)[:500]}")
        return res["prompt_id"]

    def poll(self, pid: str, timeout: float = 1800, interval: float = 2.0) -> dict:
        """Wait for the prompt to land in history; return the history entry."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            h = self._get(f"/history/{pid}")
            if pid in h:
                return h[pid]
            time.sleep(interval)
        raise ComfyError(f"prompt {pid} did not finish within {timeout}s")

    @staticmethod
    def entry_images(entry: dict) -> list[dict]:
        """All images in a history entry — both 'temp' and 'output' types
        (SaveImageAdvanced outputs are 'output'; ImageCompare emits 'temp')."""
        out = []
        for nid, o in entry.get("outputs", {}).items():
            for img in o.get("images", []):
                if img.get("type") in ("temp", "output"):
                    out.append({"node": nid, **img})
        return out

    def view_bytes(self, filename: str, subfolder: str = "", type: str = "output") -> bytes:
        q = f"filename={filename}&subfolder={urllib.request.quote(subfolder)}&type={type}"
        with urllib.request.urlopen(f"{self.url}/view?{q}", timeout=120) as r:
            return r.read()

    def download_entry(self, entry: dict, out_dir: str | Path) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for img in self.entry_images(entry):
            p = out_dir / img["filename"]
            p.write_bytes(self.view_bytes(img["filename"], img.get("subfolder", ""), img["type"]))
            paths.append(p)
        return paths

    def run(self, workflow_path: str | Path, overrides: dict[str, dict] | None = None,
            out_dir: str | Path = "images_out") -> list[Path]:
        """Full cycle: UNET swap, submit, poll, download. Returns saved paths."""
        graph = self.load_graph(workflow_path)
        graph = self.swap_unet(graph)
        pid = self.submit(graph, overrides)
        entry = self.poll(pid)
        status = entry.get("status", {}).get("status_str")
        if status != "success":
            raise ComfyError(f"prompt {pid} status={status}: {json.dumps(entry)[:1500]}")
        return self.download_entry(entry, out_dir)

    # -- upload ------------------------------------------------------------ #
    def upload(self, local: str | Path, as_name: str | None = None) -> str:
        """POST /upload/image (multipart, overwrite=true) -> stored name.

        The server keeps the *original* filename (the `name` form field is
        ignored) — pass a local file already named as you want it, or use
        as_name to copy it to a temp file under that name first.

        The file part itself carries the field name `image`; do NOT also send
        an empty text field `image` — ComfyUI rejects that with a bare 400.
        """
        local = Path(local)
        tmp: Path | None = None
        if as_name and local.name != as_name:
            # the temp file must be named EXACTLY as_name: the server keeps
            # the original filename and ignores the `name` form field
            tmp = Path(os.environ.get("TMPDIR", "/tmp")) / as_name
            shutil.copyfile(local, tmp)
            src = tmp
        else:
            src = local
        try:
            body, ctype = _multipart({"overwrite": "true"},
                                     "image", src.name, src.read_bytes(),
                                     "application/octet-stream")
            name = self._post("/upload/image", body, ctype).get("name")
            if not name:
                raise ComfyError(f"upload of {local} returned no name")
            return name
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
