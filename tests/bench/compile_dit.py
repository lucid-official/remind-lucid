"""Eager vs ``lucid.compile`` for one real-size DiT block (README: speed, Lucid findings).

    python tests/bench/compile_dit.py [bf16|fp16|fp32]

Random weights at the generator's real widths (3072 / 14336, 24 heads), one
chunk of 1,170 query tokens, history of 0 .. 7,020 cached tokens. Reports
milliseconds per block step for eager, per-signature static programs and a
``dynamic=True`` program, the first-call (compile) time, and whether the
compiled result equals the eager one. A watchdog aborts past 20 GB.
"""

import sys
import time

import lucid

from _guard import guard
from remind_lucid.accel import _BlockFresh, _BlockHistory
from remind_lucid.models.wan_dit import CausalWanModel, DiTConfig
from remind_lucid.pipeline import build_on

name = sys.argv[1] if len(sys.argv) > 1 else "bf16"
dtype = {"bf16": lucid.bfloat16, "fp16": lucid.float16, "fp32": lucid.float32}[name]
guard(20.0)
model = build_on("metal", dtype, lambda: CausalWanModel(DiTConfig(num_layers=1))).eval()
block = model.blocks[0]
L, REPS = 1170, 10


def rand(*shape, dt=dtype):
    return (lucid.randn(*shape, device="metal") * 0.5).to(dt)


x, e = rand(1, L, 3072), rand(1, 3, 6, 3072) * 0.1
tk, tv = rand(1, 24, 512, 128), rand(1, 24, 512, 128)
angle = lucid.randn(L, 64, device="metal")
cos = lucid.cos(angle).reshape(L, 64, 1).expand(L, 64, 2).reshape(L, 128)
sin = lucid.sin(angle).reshape(L, 64, 1).expand(L, 64, 2).reshape(L, 128)
eye = lucid.eye(4, device="metal").reshape(1, 1, 1, 1, 4, 4).expand(1, 1, 3, 1, 4, 4) + 0.01 * lucid.randn(1, 1, 3, 1, 4, 4, device="metal")
base = (x, e, tk, tv, cos, sin, eye, eye)
lucid.eval(*base)


def timed(fn, args):
    t0 = time.perf_counter()
    out = fn(*args)
    lucid.eval(*out)
    first = (time.perf_counter() - t0) * 1e3
    for _ in range(2):
        lucid.eval(*fn(*args))
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = fn(*args)
        lucid.eval(*out)
    return first, (time.perf_counter() - t0) / REPS * 1e3, out[0]


print(f"dtype {name}, lucid {lucid.__version__}")
static_fresh = lucid.compile(_BlockFresh(block).eval(), dynamic=False)
static_hist = lucid.compile(_BlockHistory(block).eval(), dynamic=False)
dyn_hist = lucid.compile(_BlockHistory(block).eval(), dynamic=True)
with lucid.no_grad():
    for hist in (0, 1170, 3510, 7020):
        if hist == 0:
            args, eager_fn, programs = base, block.step, {"static": static_fresh}
        else:
            hk, hv = rand(1, 24, hist, 128), rand(1, 24, hist, 128)
            lucid.eval(hk, hv)
            args, eager_fn, programs = (*base, hk, hv), block.step, {"static": static_hist, "dynamic": dyn_hist}
        _, eager_ms, ref = timed(eager_fn, args)
        line = f"history {hist:5d}: eager {eager_ms:7.1f} ms"
        for label, program in programs.items():
            try:
                first, ms, out = timed(program, args)
                diff = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
                line += f" | {label} {ms:7.1f} ms (first call {first:8.0f} ms, max rel diff {diff:.1e})"
            except Exception as exc:
                line += f" | {label} failed: {type(exc).__name__}: {str(exc)[:70]}"
        print(line + f" | in use {lucid.metal.memory_allocated() / 1e9:.1f} GB", flush=True)
for label, program in (("static fresh", static_fresh), ("static history", static_hist), ("dynamic history", dyn_hist)):
    try:
        print(label, "cache entries:", program.cache_info()["entries"])
    except Exception as exc:
        print(label, "cache_info failed:", exc)
