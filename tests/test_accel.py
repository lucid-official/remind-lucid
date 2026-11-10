"""``lucid.compile`` block programs give the eager rollout (toy generator, fp32 on Metal)."""

import lucid

from remind_lucid.accel import BlockPrograms
from test_probes import _run, _setup


def test_compiled_blocks_match_eager():
    model, job, features, source = _setup("metal")
    eager = _run(model, job, features, source, None, "metal")
    model.runner = BlockPrograms(model)
    try:
        compiled = _run(model, job, features, source, None, "metal")
        again = _run(model, job, features, source, None, "metal")  # second pass: cached executables
    finally:
        model.runner = None
    scale = eager.abs().max().item()
    assert (compiled - eager).abs().max().item() / scale < 1e-4
    assert (again == compiled).all().item()
