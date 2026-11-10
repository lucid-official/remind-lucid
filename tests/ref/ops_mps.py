"""torch-MPS timings of the generator's dominant ops at its real shapes (README op table).

    .venv-ref/bin/python tests/ref/ops_mps.py

Same shapes and repetition count as ``tests/bench/ops_lucid.py``.
"""

import json
import time

import torch

dev = torch.device("mps")
REPS = 20


def bench(fn):
    for _ in range(3):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = fn()
    torch.mps.synchronize()
    del out
    return (time.perf_counter() - t0) / REPS * 1e3


rows = {}
for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
    try:
        x = torch.randn(1, 1170, 3072, device=dev).to(dtype)
        w1 = torch.randn(3072, 3072, device=dev).to(dtype); b1 = torch.randn(3072, device=dev).to(dtype)
        w2 = torch.randn(14336, 3072, device=dev).to(dtype); b2 = torch.randn(14336, device=dev).to(dtype)
        q = torch.randn(1, 24, 1170, 128, device=dev).to(dtype)
        k = torch.randn(1, 24, 8190, 128, device=dev).to(dtype)
        k0 = torch.randn(1, 24, 1170, 128, device=dev).to(dtype)
        rows[f"linear 1170x3072->3072 {name}"] = bench(lambda: torch.nn.functional.linear(x, w1, b1))
        rows[f"linear 1170x3072->14336 {name}"] = bench(lambda: torch.nn.functional.linear(x, w2, b2))
        rows[f"attention q1170 kv1170 {name}"] = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k0, k0))
        rows[f"attention q1170 kv8190 {name}"] = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, k))
    except Exception as exc:  # e.g. a dtype MPS does not support
        rows[f"{name}"] = f"failed: {type(exc).__name__}: {str(exc)[:80]}"
print(json.dumps({"torch": torch.__version__, "ms": rows}, indent=1))
