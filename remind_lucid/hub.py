"""Hugging Face downloads straight into Lucid-ready SafeTensors.

Nothing is fetched that the pipeline does not read. Three ways in:

* :func:`fetch_file` downloads a file whole, resuming a ``.part``, and can
  check its SHA-256 (the two ReMind files, against the release's
  ``SHA256SUMS``).
* :class:`RemoteSafetensors` reads a remote SafeTensors file by HTTP range
  requests — the header first, then only selected tensors
  (:func:`fetch_passthrough` copies bytes, :func:`fetch_cast` casts on the
  CPU with round-to-nearest-even, exactly like ``Tensor.to`` in the
  reference).
* :class:`RangeFile` is a seekable read-only file over range requests, so
  :mod:`zipfile` can open a remote ``torch.save`` archive and
  :func:`remind_lucid.io.torch_zip.to_safetensors` can convert it tensor by
  tensor without the archive ever touching the disk.

Every function takes an optional ``progress(name, done, total)`` callback;
the CLI prints with it and the web demo streams it over SSE.

Only the standard library and Lucid are used.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import lucid

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
_CHUNK = 48 << 20
_ST_DTYPE = {"F32": lucid.float32, "F16": lucid.float16, "BF16": lucid.bfloat16}

Progress = Callable[[str, int, int], None]


def hf_url(repo: str, filename: str, revision: str = "main") -> str:
    return f"{HF_ENDPOINT}/{repo}/resolve/{revision}/{filename}"


_resolved: dict[str, str] = {}  # resolve URL -> signed CDN URL, so a redirect is paid once per file


def _request(
    url: str, start: int | None = None, end: int | None = None, retries: int = 8
):
    headers = {"User-Agent": "remind-lucid"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end - 1}"
    for attempt in range(retries):
        target = _resolved.get(url, url)
        request_headers = dict(headers)
        token = os.environ.get("HF_TOKEN")
        if token and target == url:  # never forward the token to the CDN host
            request_headers["Authorization"] = f"Bearer {token}"
        try:
            response = urllib.request.urlopen(
                urllib.request.Request(target, headers=request_headers), timeout=60
            )
            if response.geturl() != url:
                _resolved[url] = response.geturl()
            return response
        except urllib.error.HTTPError as exc:
            if target != url and exc.code in (400, 401, 403, 404):
                _resolved.pop(url, None)  # the signed URL expired: resolve again
                continue
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries - 1:
                raise
            wait = min(60, 2**attempt)
            print(f"  retry in {wait}s after {exc}", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise urllib.error.URLError(f"giving up on {url}")


def read_range(url: str, start: int, end: int, on_block=None) -> bytes:
    """Bytes ``[start, end)`` of ``url``, re-requesting the remainder on a dropped connection."""
    out = bytearray()
    while len(out) < end - start:
        try:
            with _request(url, start + len(out), end) as response:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out += block
                    if on_block is not None:
                        on_block(len(block))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            print(f"  resume after {exc}", file=sys.stderr, flush=True)
            time.sleep(2)
    return bytes(out)


_PARALLEL = int(os.environ.get("REMIND_LUCID_CONNECTIONS", "6"))
_PIECE = 8 << 20
_pool: ThreadPoolExecutor | None = None


def read_parallel(url: str, start: int, end: int) -> bytes:
    """Bytes ``[start, end)`` over several connections (the Hub's CDN throttles each one)."""
    global _pool
    if end - start <= _PIECE or _PARALLEL <= 1:
        return read_range(url, start, end)
    if _pool is None:
        _pool = ThreadPoolExecutor(_PARALLEL)
    cuts = list(range(start, end, _PIECE)) + [end]
    parts = _pool.map(lambda ab: read_range(url, ab[0], ab[1]), zip(cuts[:-1], cuts[1:]))
    return b"".join(parts)


def remote_size(url: str) -> int:
    with _request(url, 0, 1) as response:
        return int(response.headers["Content-Range"].rsplit("/", 1)[1])


def print_progress() -> Progress:
    """A :data:`Progress` callback that draws one status line per file."""
    state = {"t0": time.time(), "first": None, "name": None}

    def show(name: str, done: int, total: int) -> None:
        if state["name"] != name:
            state.update(name=name, t0=time.time(), first=done)
        rate = (done - state["first"]) / max(time.time() - state["t0"], 1e-6) / 1e6
        sys.stdout.write(
            f"\r  {name}: {done / 1e9:6.2f}/{total / 1e9:.2f} GB  {rate:5.1f} MB/s"
        )
        if done >= total:
            sys.stdout.write("\n")
        sys.stdout.flush()

    return show


# --------------------------------------------------------------------------- #
# Whole files
# --------------------------------------------------------------------------- #


def sha256_file(path: Path, progress: Progress | None = None) -> str:
    digest, done, total = hashlib.sha256(), 0, path.stat().st_size
    with open(path, "rb") as fh:
        while block := fh.read(_CHUNK):
            digest.update(block)
            done += len(block)
            if progress is not None:
                progress(f"sha256 {path.name}", done, total)
    return digest.hexdigest()


def fetch_file(
    repo: str,
    filename: str,
    dest: Path,
    *,
    sha256: str | None = None,
    revision: str = "main",
    progress: Progress | None = None,
) -> None:
    """Download a whole file, resuming a ``.part``; verify ``sha256`` when given."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    url = hf_url(repo, filename, revision)
    total = remote_size(url)
    done = part.stat().st_size if part.exists() else 0
    name = f"{repo}/{filename}"
    with open(part, "ab") as fh:
        while done < total:
            block = read_parallel(url, done, min(total, done + _CHUNK))
            fh.write(block)
            done += len(block)
            if progress is not None:
                progress(name, done, total)
    if sha256 is not None:
        got = sha256_file(part, progress)
        if got != sha256:
            part.unlink()
            raise RuntimeError(f"{name}: sha256 {got} != expected {sha256}")
        dest.with_suffix(dest.suffix + ".sha256").write_text(got + "\n")
    os.replace(part, dest)


def fetch_text(repo: str, filename: str, revision: str = "main") -> str:
    with _request(hf_url(repo, filename, revision)) as response:
        return response.read().decode("utf-8")


# --------------------------------------------------------------------------- #
# Remote SafeTensors
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


class RemoteSafetensors:
    """A SafeTensors file on the Hub, read by range requests."""

    def __init__(self, repo: str, filename: str, revision: str = "main") -> None:
        self.url = hf_url(repo, filename, revision)
        self.name = f"{repo}/{filename}"
        (length,) = struct.unpack("<Q", read_range(self.url, 0, 8))
        header = json.loads(read_range(self.url, 8, 8 + length))
        self.metadata = header.pop("__metadata__", None) or {}
        self.base = 8 + length
        self.tensors = sorted(
            (
                TensorInfo(k, v["dtype"], tuple(v["shape"]), *v["data_offsets"])
                for k, v in header.items()
            ),
            key=lambda t: t.start,
        )

    def read(self, info: TensorInfo) -> bytes:
        return read_range(self.url, self.base + info.start, self.base + info.end)


def write_header(fh, entries: list[tuple[str, str, tuple[int, ...], int]]) -> int:
    header, offset = {}, 0
    for name, dtype, shape, nbytes in entries:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    fh.write(struct.pack("<Q", len(encoded)))
    fh.write(encoded)
    return 8 + len(encoded)


def fetch_passthrough(
    remote: RemoteSafetensors,
    dest: Path,
    select: Callable[[str], bool] = lambda name: True,
    progress: Progress | None = None,
) -> None:
    """Copy the selected tensors byte for byte. Resumes an interrupted ``.part`` file."""
    if dest.exists():
        return
    picked = [t for t in remote.tensors if select(t.name)]
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = sum(t.nbytes for t in picked)
    mode = "r+b" if part.exists() else "wb"
    with open(part, mode) as fh:
        data_start = write_header(
            fh, [(t.name, t.dtype, t.shape, t.nbytes) for t in picked]
        )
        done = max(0, os.path.getsize(part) - data_start)
        fh.seek(data_start + done)
        runs: list[list[TensorInfo]] = []
        for t in picked:
            if runs and runs[-1][-1].end == t.start:
                runs[-1].append(t)
            else:
                runs.append([t])
        written = 0
        for run in runs:
            size = run[-1].end - run[0].start
            if written + size <= done:
                written += size
                continue
            skip = max(0, done - written)
            pos = remote.base + run[0].start + skip
            stop = remote.base + run[-1].end
            while pos < stop:
                block = read_parallel(remote.url, pos, min(stop, pos + _CHUNK))
                fh.write(block)
                pos += len(block)
                if progress is not None:
                    progress(remote.name, fh.tell() - data_start, total)
            written += size
    os.replace(part, dest)


def fetch_cast(
    remote: RemoteSafetensors,
    dest: Path,
    dtype: str = "BF16",
    select: Callable[[str], bool] = lambda name: True,
    progress: Progress | None = None,
) -> None:
    """Download the selected tensors one at a time, cast floating ones to ``dtype`` on the CPU, save.

    At most one source tensor is in memory besides the (half-size) results.
    """
    if dest.exists():
        return
    target = _ST_DTYPE[dtype]
    picked = [t for t in remote.tensors if select(t.name)]
    total = sum(t.nbytes for t in picked)
    out: dict[str, lucid.Tensor] = {}
    got = 0
    for t in picked:
        raw = remote.read(t)
        got += len(raw)
        if t.dtype == "F32":
            value = lucid.tensor(memoryview(raw).cast("f")).reshape(t.shape)
        else:
            raise ValueError(f"{t.name}: unexpected dtype {t.dtype}")
        out[t.name] = value.to(target)
        if progress is not None:
            progress(remote.name, got, total)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    lucid.save_safetensors(out, str(part))
    os.replace(part, dest)


# --------------------------------------------------------------------------- #
# Remote torch.save archives
# --------------------------------------------------------------------------- #


class RangeFile(io.RawIOBase):
    """A read-only seekable file over HTTP range requests, with read-ahead.

    ``zipfile`` seeks to the end for the central directory, then reads each
    stored entry front to back; a 48 MB read-ahead window, fetched over several connections, turns that into a
    few hundred sequential requests.
    """

    def __init__(self, url: str, name: str, progress: Progress | None = None) -> None:
        self.url, self.name, self.progress = url, name, progress
        self.size = remote_size(url)
        self.pos = 0
        self.fetched = 0
        self._buf_start, self._buf = 0, b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self.pos, os.SEEK_END: self.size}[whence]
        self.pos = max(0, min(self.size, base + offset))
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        end = min(self.size, self.pos + n)
        out = bytearray()
        while self.pos < end:
            lo = self._buf_start
            if not (lo <= self.pos < lo + len(self._buf)):
                # Small reads near the end are the zip directory: fetch a tight window.
                # The zip directory is read first (small reads near the end): a tight
                # window there, then read-ahead windows, so progress moves during one
                # multi-GB entry.
                window = 1 << 20 if self.fetched == 0 else _CHUNK
                stop = min(self.size, self.pos + window)
                self._buf = read_parallel(self.url, self.pos, stop)
                self._buf_start = lo = self.pos
                self.fetched += len(self._buf)
                if self.progress is not None:
                    self.progress(self.name, min(self.fetched, self.size), self.size)
            take = self._buf[self.pos - lo : end - lo]
            out += take
            self.pos += len(take)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote(
    repo: str, filename: str, progress: Progress | None = None
) -> RangeFile:
    return RangeFile(hf_url(repo, filename), f"{repo}/{filename}", progress)
