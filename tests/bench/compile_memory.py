"""Memory retained by ``lucid.compile`` per input signature (README: Lucid findings).

    python tests/bench/compile_memory.py

One real-size DiT block (0.37 GB of bf16 weights), compiled once and called
with four history lengths. Each new signature adds a cached executable; the
script prints the memory in use after each and after ``clear_cache()``.
"""

import lucid

from _guard import guard
from remind_lucid.accel import _BlockHistory
from remind_lucid.models.wan_dit import CausalWanModel, DiTConfig
from remind_lucid.pipeline import build_on

guard(22.0)
dtype = lucid.bfloat16
model = build_on("metal", dtype, lambda: CausalWanModel(DiTConfig(num_layers=1))).eval()
block = model.blocks[0]
L = 1170


def rand(*shape):
    return (lucid.randn(*shape, device="metal") * 0.5).to(dtype)


def in_use() -> float:
    return lucid.metal.memory_allocated() / 1e9


angle = lucid.randn(L, 64, device="metal")
cos = lucid.cos(angle).reshape(L, 64, 1).expand(L, 64, 2).reshape(L, 128)
sin = lucid.sin(angle).reshape(L, 64, 1).expand(L, 64, 2).reshape(L, 128)
eye = lucid.eye(4, device="metal").reshape(1, 1, 1, 1, 4, 4).expand(1, 1, 3, 1, 4, 4) + 0.0
base = (rand(1, L, 3072), rand(1, 3, 6, 3072) * 0.1, rand(1, 24, 512, 128), rand(1, 24, 512, 128), cos, sin, eye, eye)
lucid.eval(*base)
weights = sum(p.numel() for p in block.parameters()) * 2 / 1e9
with lucid.no_grad():
    lucid.eval(*block.step(*base))
    lucid.metal.empty_cache()
    print(f"block weights (bf16): {weights:.2f} GB; in use before compiling: {in_use():.2f} GB")
    program = lucid.compile(_BlockHistory(block).eval(), dynamic=False)
    for hist in (1170, 2340, 3510, 4680):
        hk, hv = rand(1, 24, hist, 128), rand(1, 24, hist, 128)
        lucid.eval(hk, hv)
        before = in_use()
        out = program(*base, hk, hv)
        lucid.eval(*out)
        del out, hk, hv
        lucid.metal.empty_cache()
        print(f"signature history={hist}: in use {in_use():.2f} GB (+{in_use() - before:.2f}), cache entries {program.cache_info()['entries']}")
    program.clear_cache()
    lucid.metal.empty_cache()
    print(f"after clear_cache(): in use {in_use():.2f} GB, cache entries {program.cache_info()['entries']}")
