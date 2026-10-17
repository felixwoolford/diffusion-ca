"""ComfyUI client tests against a local fake HTTP server (no remote needed)."""
import http.server
import json
import threading

import pytest

from applet.engine.comfy import ComfyClient, _multipart


class FakeComfy(http.server.BaseHTTPRequestHandler):
    server_version = "FakeComfy/1.0"
    uploads: list[dict] = []
    prompts: list[dict] = []

    def log_message(self, *a):  # quiet
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/models/diffusion_models":
            self._send(["qwen_image_2.1_int8_convrot.safetensors"])
        elif self.path == "/history/fakepid":
            self._send({"fakepid": {
                "status": {"status_str": "success"},
                "outputs": {"461": {"images": [
                    {"filename": "out.png", "subfolder": "", "type": "output"}]}}}})
        elif self.path.startswith("/view?"):
            body = b"\x89PNG-fake"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send({"error": self.path}, 404)

    def do_POST(self):  # noqa: N802
        if self.path == "/upload/image":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            ctype = self.headers.get("Content-Type", "")
            assert ctype.startswith("multipart/form-data"), "expected multipart"
            boundary = ctype.split("boundary=")[1].encode()
            # every part must have a name= field; file parts carry filename=
            parts = body.split(b"--" + boundary)
            file_parts = [p for p in parts if b"filename=" in p]
            text_parts = [p for p in parts if b"filename=" not in p and b"Content-Disposition" in p]
            # regression: an extra empty text field named 'image' must not exist
            for p in text_parts:
                assert b'name="image"' not in p.split(b"\r\n\r\n")[0], \
                    "stray text field named 'image' in multipart body"
            self.__class__.uploads.append({"boundary": boundary.decode(),
                                           "file_parts": len(file_parts),
                                           "body": body})
            name = file_parts[0].split(b"filename=")[1].split(b'"')[1].decode()
            self._send({"name": name, "subfolder": "", "type": "input"})
        elif self.path == "/prompt":
            length = int(self.headers.get("Content-Length", 0))
            self.__class__.prompts.append(json.loads(self.rfile.read(length)))
            self._send({"prompt_id": "fakepid"})
        else:
            self._send({"error": self.path}, 404)


@pytest.fixture()
def client():
    FakeComfy.uploads = []
    FakeComfy.prompts = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    c = ComfyClient(f"http://127.0.0.1:{httpd.server_address[1]}")
    yield c
    httpd.shutdown()


def test_upload(client, tmp_path):
    p = tmp_path / "myref.png"
    p.write_bytes(b"\x89PNG-fake-bytes")
    name = client.upload(p, as_name="renamed_ref.png")
    assert name == "renamed_ref.png"
    up = FakeComfy.uploads[-1]
    assert up["file_parts"] == 1
    assert b"renamed_ref.png" in up["body"]
    assert b'name="overwrite"' in up["body"]


def test_models_and_run_cycle(client, tmp_path):
    assert client.models() == ["qwen_image_2.1_int8_convrot.safetensors"]
    graph = {"484": {"class_type": "UNETLoader",
                     "inputs": {"unet_name": "qwen_image_2.1_bf16.safetensors"}},
             "490": {"class_type": "KSampler", "inputs": {"seed": 1}}}
    # unet swap: server lacks the bf16 file
    g = client.swap_unet(graph)
    assert g["484"]["inputs"]["unet_name"] == "qwen_image_2.1_int8_convrot.safetensors"
    pid = client.submit(g, {"490": {"seed": 42}})
    assert pid == "fakepid"
    sent = FakeComfy.prompts[-1]["prompt"]["490"]["inputs"]["seed"]
    assert sent == 42
    entry = client.poll(pid, timeout=10)
    assert entry["status"]["status_str"] == "success"
    assert client.entry_images(entry)[0]["filename"] == "out.png"
    paths = client.download_entry(entry, tmp_path)
    assert paths and paths[0].read_bytes() == b"\x89PNG-fake"


def test_multipart_shape():
    body, ctype = _multipart({"overwrite": "true"}, "image", "a.png", b"1234", "image/png")
    b = ctype.split("boundary=")[1]
    assert body.startswith(f"--{b}\r\n".encode())
    assert body.endswith(f"--{b}--\r\n".encode())
    assert b'name="image"; filename="a.png"' in body
    assert b"1234" in body
