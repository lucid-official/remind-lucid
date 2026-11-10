"""The torch-free ``torch.save`` reader, the remote-file shim and the ``.npz`` reader."""

import io
import os
import pickle
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer

import lucid
import pytest

from conftest import FIXTURES, load_fixture
from remind_lucid import hub
from remind_lucid.io.npy import load_npz
from remind_lucid.io.torch_zip import read_state_dict, to_safetensors
from remind_lucid.presets import PRESET_DIR

ARCHIVE = FIXTURES / "tiny_state.pth"


def _same(a: dict, b: dict) -> None:
    assert sorted(a) == sorted(b)
    for key in a:
        assert a[key].dtype == b[key].dtype and tuple(a[key].shape) == tuple(
            b[key].shape
        ), key
        assert (a[key].to(lucid.float64) == b[key].to(lucid.float64)).all().item(), key


def test_archive_to_safetensors(tmp_path):
    if not ARCHIVE.exists():
        pytest.skip("run tests/ref/tiny_modules.py in .venv-ref")
    to_safetensors(ARCHIVE, tmp_path / "out.safetensors")
    _same(
        lucid.load_safetensors(str(tmp_path / "out.safetensors")),
        load_fixture("tiny_state.safetensors"),
    )


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned > /dev/null",))


def test_reader_refuses_code(tmp_path):
    path = tmp_path / "evil.pth"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps({"w": _Evil()}))
    with pytest.raises(pickle.UnpicklingError, match="refusing to load"):
        read_state_dict(path)


class _RangeHandler(BaseHTTPRequestHandler):
    payload = b""
    requests = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        type(self).requests += 1
        lo, _, hi = self.headers["Range"].removeprefix("bytes=").partition("-")
        lo, hi = int(lo), int(hi) if hi else len(self.payload) - 1
        body = self.payload[lo : hi + 1]
        self.send_response(206)
        self.send_header(
            "Content-Range", f"bytes {lo}-{lo + len(body) - 1}/{len(self.payload)}"
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_remote_archive_by_range_requests(tmp_path):
    """A ``torch.save`` archive converted straight from HTTP range requests, never stored whole."""
    if not ARCHIVE.exists():
        pytest.skip("run tests/ref/tiny_modules.py in .venv-ref")
    _RangeHandler.payload = ARCHIVE.read_bytes()
    server = HTTPServer(("127.0.0.1", 0), _RangeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/tiny_state.pth"
        seen = []
        remote = hub.RangeFile(
            url, "tiny", lambda name, done, total: seen.append((done, total))
        )
        assert remote.size == len(_RangeHandler.payload)
        to_safetensors(remote, tmp_path / "remote.safetensors")
        assert hub.read_range(url, 3, 11) == _RangeHandler.payload[3:11]
    finally:
        server.shutdown()
    _same(
        lucid.load_safetensors(str(tmp_path / "remote.safetensors")),
        load_fixture("tiny_state.safetensors"),
    )
    assert seen and seen[-1][1] == len(_RangeHandler.payload)


def test_npz_trajectory():
    arrays = load_npz(PRESET_DIR / "assets" / "whisking_batter_camera.npz")
    assert tuple(arrays["extrinsics"].shape) == (
        81,
        3,
        4,
    )  # 3x4 camera-to-world; the loader adds the bottom row
    assert tuple(arrays["intrinsics"].shape) == (81, 3, 3)
    assert arrays["extrinsics"].dtype == lucid.float32
