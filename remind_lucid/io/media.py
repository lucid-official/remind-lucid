"""Images and videos in, videos out.

Decoding and encoding go through the ``ffmpeg`` binary (raw RGB24 over a
pipe); everything in between is Lucid. The reference resizes its inputs with
``PIL.Image.resize((W, H), LANCZOS)``, Pillow's two-pass filter in 8-bit
fixed point (a no-op for the bundled presets, which are already 832x480).
:func:`pil_resize` reproduces it: the fixed-point coefficients are integers,
and so are the sums, which float64 matrix products represent without
rounding.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import lucid

Tensor = lucid.Tensor
_PRECISION_BITS = 32 - 8 - 2


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(
            "ffmpeg is required for image/video I/O (brew install ffmpeg)"
        )
    return exe


def _probe_size(path: str) -> tuple[int, int]:
    out = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-i", path], capture_output=True, text=True
    ).stderr
    for token in out.replace(",", " ").split():
        if "x" in token:
            w, _, h = token.partition("x")
            if w.isdigit() and h.isdigit() and int(w) > 1 and int(h) > 1:
                return int(w), int(h)
    raise ValueError(f"cannot read the size of {path}")


def read_image(path: str | Path) -> Tensor:
    """Decode an image to ``(H, W, 3)`` int16 pixel values ``0..255`` (alpha dropped, like ``convert('RGB')``)."""
    path = str(path)
    w, h = _probe_size(path)
    raw = subprocess.run(
        [
            _ffmpeg(),
            "-v",
            "error",
            "-i",
            path,
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    if len(raw) != w * h * 3:
        raise ValueError(f"decoded {len(raw)} bytes for a {w}x{h} image")
    return lucid.tensor(memoryview(raw)).reshape(h, w, 3)


def image_from_bytes(data: bytes) -> Tensor:
    """Decode image bytes (any format ffmpeg reads) to ``(H, W, 3)``."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".img") as fh:
        fh.write(data)
        fh.flush()
        return read_image(fh.name)


def _lanczos(x: float) -> float:
    if not -3.0 <= x < 3.0:
        return 0.0
    if x == 0.0:
        return 1.0
    a, c = math.pi * x, math.pi * x / 3.0
    return (math.sin(a) / a) * (math.sin(c) / c)


def _pil_coefficients(in_size: int, out_size: int) -> Tensor:
    """Pillow's ``precompute_coeffs`` (Lanczos, support 3) + ``normalize_coeffs_8bpc``, as an ``(out, in)`` matrix."""
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 3.0 * filterscale
    rows = []
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        ss = 1.0 / filterscale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size) - xmin
        weights = [_lanczos((x + xmin - center + 0.5) * ss) for x in range(xmax)]
        total = sum(weights)
        row = [0.0] * in_size
        for x, wgt in enumerate(weights):
            k = wgt / total if total != 0.0 else wgt
            fixed = (
                math.trunc(-0.5 + k * (1 << _PRECISION_BITS))
                if k < 0
                else math.trunc(0.5 + k * (1 << _PRECISION_BITS))
            )
            row[x + xmin] = float(fixed)
        rows.append(row)
    return lucid.tensor(rows, dtype=lucid.float64)


def _fixed_point_pass(x: Tensor, coeffs: Tensor) -> Tensor:
    """``clip8((2^21 + sum in * k) >> 22)`` along the last axis."""
    acc = x @ coeffs.swapaxes(0, 1) + float(1 << (_PRECISION_BITS - 1))
    return lucid.floor(acc / float(1 << _PRECISION_BITS)).clamp(0.0, 255.0)


def pil_resize(image: Tensor, height: int, width: int) -> Tensor:
    """``PIL.Image.resize((width, height), LANCZOS)`` of an ``(H, W, 3)`` 8-bit image; identity at equal size."""
    h, w, c = image.shape
    if (h, w) == (height, width):
        return image
    x = image.to(lucid.float64)
    if w != width:  # horizontal pass first, as Pillow does
        x = _fixed_point_pass(x.permute(0, 2, 1), _pil_coefficients(w, width)).permute(
            0, 2, 1
        )
    if h != height:
        x = _fixed_point_pass(x.permute(1, 2, 0), _pil_coefficients(h, height)).permute(
            2, 0, 1
        )
    return x.to(lucid.int16)


def frames_to_model(frames: Tensor, dtype) -> Tensor:
    """``(T, H, W, 3)`` 8-bit frames -> ``(1, 3, T, H, W)`` in ``[-1, 1]``.

    The reference divides by 255 in fp32, casts to the run's dtype and only
    then maps to ``[-1, 1]`` (``.to(dtype).mul(2.0).sub(1.0)``); the order of
    the cast matters in half precision and is kept.
    """
    pixels = (frames.float() / 255.0).to(dtype) * 2.0 - 1.0
    return pixels.permute(3, 0, 1, 2).unsqueeze(0)


def read_video(path: str | Path, max_frames: int, height: int, width: int) -> Tensor:
    """The first ``max_frames`` frames of a video as ``(T, H, W, 3)`` 8-bit values, each resized like an image."""
    path = str(path)
    w, h = _probe_size(path)
    raw = subprocess.run(
        [
            _ffmpeg(),
            "-v",
            "error",
            "-i",
            path,
            "-frames:v",
            str(max_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    n = len(raw) // (w * h * 3)
    frames = lucid.tensor(memoryview(raw[: n * w * h * 3])).reshape(n, h, w, 3)
    if (h, w) != (height, width):
        frames = lucid.stack([pil_resize(frames[i], height, width) for i in range(n)])
    return frames


def to_uint8_frames(video: Tensor) -> list[bytes]:
    """Decoder output ``(3, T, H, W)`` in ``[-1, 1]`` -> RGB24 frames.

    ``(v * 0.5 + 0.5).clamp(0, 1) * 255`` truncated to integers, as
    ``utils.video_io.write_mp4`` does with the reference's float video.
    """
    v = ((video.float() * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).to("cpu")
    v = lucid.floor(v).to(lucid.int16).permute(1, 2, 3, 0)  # (T, H, W, 3)
    return [bytes(frame.reshape(-1).tolist()) for frame in v]


class VideoWriter:
    """Append RGB24 frames to an H.264 MP4 through an ffmpeg pipe."""

    def __init__(
        self, path: str | Path, width: int, height: int, fps: int = 16, crf: int = 18
    ) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [
                _ffmpeg(),
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(crf),
                "-movflags",
                "+faststart",
                self.path,
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frames: list[bytes]) -> None:
        for frame in frames:
            self.proc.stdin.write(frame)

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")


def encode_jpeg(frame: bytes, width: int, height: int, quality: int = 4) -> bytes:
    """One RGB24 frame as JPEG bytes (for streaming previews)."""
    return subprocess.run(
        [
            _ffmpeg(),
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-i",
            "-",
            "-frames:v",
            "1",
            "-q:v",
            str(quality),
            "-f",
            "mjpeg",
            "-",
        ],
        input=frame,
        capture_output=True,
        check=True,
    ).stdout


def make_gif(
    video: str | Path, out: str | Path, width: int = 416, fps: int = 8, colors: int = 96
) -> None:
    """A palette GIF of a video (for README media)."""
    flt = f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"  # fmt: skip
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-i", str(video), "-filter_complex", flt, str(out)], check=True)  # fmt: skip


def make_strip(
    videos: list[str | Path], out: str | Path, frames: list[int], width: int = 208
) -> None:
    """A contact sheet: one row per video, one column per frame index (for README media)."""
    select = "+".join(f"eq(n\\,{f})" for f in frames)
    inputs, rows = [], []
    for i, video in enumerate(videos):
        inputs += ["-i", str(video)]
        rows.append(
            f"[{i}:v]select='{select}',scale={width}:-1,tile={len(frames)}x1[r{i}]"
        )
    stack = "".join(f"[r{i}]" for i in range(len(videos)))
    graph = ";".join(rows) + (f";{stack}vstack=inputs={len(videos)}[o]" if len(videos) > 1 else ";[r0]copy[o]")  # fmt: skip
    subprocess.run([_ffmpeg(), "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[o]", "-frames:v", "1", "-update", "1", "-q:v", "3", str(out)], check=True)  # fmt: skip
