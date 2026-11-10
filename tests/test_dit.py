"""The causal DiT against ``wan/modules/causal_model.py`` at toy size (random fp32 weights).

Covers a denoising step without history, the cache-commit pass, a denoising
step over cached history, and a chunk placed at non-contiguous temporal
positions; per-frame timesteps, a moving camera and chunk-local text.
"""

import lucid
import pytest

from conftest import DEVICES, load_fixture, max_rel
from remind_lucid import camera
from remind_lucid.models.wan_dit import CausalWanModel, DiTConfig

CFG = DiTConfig(
    dim=192, ffn_dim=256, text_dim=64, text_len=32, num_heads=6, num_layers=2
)
F_, GRID = 3, (4, 6)


def _model(ref, device):
    model = CausalWanModel(CFG).eval()
    model.load_state_dict(
        {k[6:]: v for k, v in ref.items() if k.startswith("state.")}, strict=True
    )
    return model.to(device)


def _geom(model, ref, chunk, positions=None):
    lo = chunk * F_
    cam = camera.prepare_chunk(
        ref["viewmats"][:, lo : lo + F_], ref["Ks"][:, lo : lo + F_], GRID[1] * 2 * 16.0
    )
    return cam, model.geometry(positions or list(range(lo, lo + F_)), cam, GRID)


def test_camera_and_rope_tables():
    ref = load_fixture("tiny_dit.safetensors")
    model = _model(ref, "cpu")
    cam, geom = _geom(model, ref, 1)
    assert max_rel(cam.pose, ref["c1_denoise.pose_c"]) < 1e-6
    # freqs_x is the camera-free base table (complex128 as [cos, sin]); the phasor is applied per block.
    assert geom.cos[0].shape == (F_ * GRID[0] * GRID[1], 32)


@pytest.mark.parametrize("device", DEVICES)
def test_rollout_calls(device):
    ref = load_fixture("tiny_dit.safetensors")
    model = _model(ref, device)
    cache = model.new_cache()
    with lucid.no_grad():
        embedded = model.embed_text(ref["context"][0].to(device))
        assert max_rel(embedded[0:1], ref["c0_denoise.context"][0]) < 1e-5
        ctx = [
            model.text_context(ref["context"][0][i : i + 1].to(device))
            for i in range(2)
        ]
        x0, x1 = ref["x0"].to(device), ref["x1"].to(device)
        _, g0 = _geom(model, ref, 0)
        _, g1 = _geom(model, ref, 1)
        _, g_gap = _geom(model, ref, 1, positions=[40, 41, 42])

        taps = {}
        flow = model(x0, [0, 938, 938], ctx[0], g0, cache, taps=taps)
        for i in range(CFG.num_layers):
            assert max_rel(taps[f"block_{i}"], ref[f"c0_denoise.block_{i}"]) < 1e-4, i
        assert max_rel(flow, ref["c0_denoise.flow"]) < 1e-4

        flow = model(x0, [0, 0, 0], ctx[0], g0, cache, write_key=0)
        assert max_rel(flow, ref["c0_commit.flow"]) < 1e-4
        for i in range(CFG.num_layers):
            k, v = cache.read(i)
            assert max_rel(k.permute(0, 2, 1, 3), ref[f"c0_commit.k_{i}"]) < 1e-4
            assert max_rel(v.permute(0, 2, 1, 3), ref[f"c0_commit.v_{i}"]) < 1e-4

        taps = {}
        flow = model(x1, [625, 625, 625], ctx[1], g1, cache, taps=taps)
        for i in range(CFG.num_layers):
            assert max_rel(taps[f"block_{i}"], ref[f"c1_denoise.block_{i}"]) < 1e-4, i
        assert max_rel(flow, ref["c1_denoise.flow"]) < 1e-4

        flow = model(x1, [625, 625, 625], ctx[1], g_gap, cache)
        assert max_rel(flow, ref["c1_gap.flow"]) < 1e-4
