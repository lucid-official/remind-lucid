"""Eager vs ``lucid.compile`` for the streaming Wan2.2 VAE decoder (README: speed, Lucid findings).

    python tests/bench/compile_vae.py [fp32|bf16]

Decodes six latent frames (21 video frames at 832x480) eagerly and through
``remind_lucid.accel.CompiledDecoder``; prints seconds per latent frame and
peak memory. A watchdog aborts the compiled run if memory passes 20 GB.
"""

import sys
import time

import lucid

from _guard import guard
from remind_lucid import presets
from remind_lucid.accel import CompiledDecoder
from remind_lucid.io import media
from remind_lucid.pipeline import StreamingDecoder, load_vae

dtype = {"fp32": lucid.float32, "bf16": lucid.bfloat16}[
    sys.argv[1] if len(sys.argv) > 1 else "fp32"
]
guard(20.0)
vae = load_vae("metal", dtype)
_, preset = presets.load_preset("08")
frames = media.read_video(preset["_input_video_path"], 21, 480, 832)
with lucid.no_grad():
    z = (
        vae.encode(media.frames_to_model(frames, lucid.bfloat16).to("metal"))
        .float()
        .permute(0, 2, 1, 3, 4)
    )
    lucid.eval(z)
lucid.metal.empty_cache()


def run(decoder, label):
    for rep in range(2):
        decoder.reset()
        lucid.metal.reset_peak_memory_stats()
        times, outs = [], []
        for i in range(z.shape[1]):
            t0 = time.time()
            outs.append(decoder(z[:, i : i + 1]).to("cpu"))
            times.append(round(time.time() - t0, 2))
        print(
            f"{label} pass {rep}: s/latent frame {times}, peak {lucid.metal.max_memory_allocated() / 1e9:.1f} GB",
            flush=True,
        )
    return lucid.cat(outs, dim=1)


eager = run(
    StreamingDecoder(vae), f"{sys.argv[1] if len(sys.argv) > 1 else 'fp32'} eager"
)
compiled = run(CompiledDecoder(vae), "compiled")
print("compiled vs eager max abs diff", (compiled - eager).abs().max().item())
