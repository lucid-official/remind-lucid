"""Render an attention-mass report as a PNG heat map with the standard library (zlib)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

# A perceptually ordered ramp (dark blue -> teal -> yellow), 5 stops.
_RAMP = [(13, 8, 135), (84, 2, 163), (185, 50, 137), (240, 124, 79), (240, 249, 33)]


def _color(value: float) -> tuple[int, int, int]:
    value = min(1.0, max(0.0, value)) * (len(_RAMP) - 1)
    i = min(int(value), len(_RAMP) - 2)
    t = value - i
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(_RAMP[i], _RAMP[i + 1]))


def write_png(path: str | Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    height, width = len(pixels), len(pixels[0])
    raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in pixels)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))  # fmt: skip

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))  # fmt: skip
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def layer_by_key(report: dict, chunk: int) -> tuple[list[int], list[int], list[list[float]]]:
    """Head-averaged mass of one generated chunk: rows = logged layers, columns = key chunks."""
    layers = report["attention_mass"][str(chunk)]
    keys = report["attention_keys"][str(chunk)]
    order = sorted(int(l) for l in layers)
    rows = [[sum(head[c] for head in layers[str(l)]) / len(layers[str(l)]) for c in range(len(keys))] for l in order]  # fmt: skip
    return order, keys, rows


def render(report: dict, chunk: int, path: str | Path, cell: int = 28, normalise: bool = True) -> None:  # fmt: skip
    """Heat map for one generated chunk (rows: layers, top = first; columns: key chunks, last = itself)."""
    _, _, rows = layer_by_key(report, chunk)
    top = max(max(r) for r in rows) if normalise else 1.0
    pixels = []
    for row in rows:
        line = [px for v in row for px in [_color(v / top if top else 0.0)] * (cell - 1) + [(20, 20, 20)]]  # fmt: skip
        pixels += [line] * (cell - 1) + [[(20, 20, 20)] * len(line)]
    write_png(path, pixels)
