"""Lucid (Metal) timings of the generator's dominant ops at its real shapes (README op table).

    python tests/bench/ops_lucid.py

Same shapes and repetition count as ``tests/ref/ops_mps.py``.
"""

import json
import time

import lucid
import lucid.nn.functional as F

REPS = 20


def bench(fn):
    for _ in range(3):
        lucid.eval(fn())
    t0 = time.perf_counter()
    for _ in range(REPS):
        lucid.eval(fn())
    return (time.perf_counter() - t0) / REPS * 1e3


rows = {}
for name, dtype in (
    ("bf16", lucid.bfloat16),
    ("fp16", lucid.float16),
    ("fp32", lucid.float32),
):
    x = lucid.randn(1, 1170, 3072, device="metal").to(dtype)
    w1, b1 = lucid.randn(3072, 3072, device="metal").to(dtype), lucid.randn(
        3072, device="metal"
    ).to(dtype)
    w2, b2 = lucid.randn(14336, 3072, device="metal").to(dtype), lucid.randn(
        14336, device="metal"
    ).to(dtype)
    q = lucid.randn(1, 24, 1170, 128, device="metal").to(dtype)
    k = lucid.randn(1, 24, 8190, 128, device="metal").to(dtype)
    k0 = lucid.randn(1, 24, 1170, 128, device="metal").to(dtype)
    lucid.eval(x, w1, b1, w2, b2, q, k, k0)
    rows[f"linear 1170x3072->3072 {name}"] = bench(lambda: F.linear(x, w1, b1))
    rows[f"linear 1170x3072->14336 {name}"] = bench(lambda: F.linear(x, w2, b2))
    rows[f"attention q1170 kv1170 {name}"] = bench(
        lambda: F.scaled_dot_product_attention(q, k0, k0)
    )
    rows[f"attention q1170 kv8190 {name}"] = bench(
        lambda: F.scaled_dot_product_attention(q, k, k)
    )
print(json.dumps({"lucid": lucid.__version__, "ms": rows}, indent=1))
