"""Probes are no-ops when off: the rollout is bit-identical with and without the probe machinery.

Run on a toy generator with random weights (no released weights needed).
"""

import lucid
import pytest

from conftest import DEVICES
from remind_lucid import presets
from remind_lucid.models.wan_dit import CausalWanModel, DiTConfig
from remind_lucid.pipeline import Job, NoiseSource, ReMindPipeline
from remind_lucid.probes import ProbeSet
from remind_lucid.probes.sweeps import (
    interruption_phases,
    interruption_sweep,
    seed_sweep,
)

CFG = DiTConfig(
    dim=192, ffn_dim=256, text_dim=64, text_len=16, num_heads=6, num_layers=3
)


def _setup(device):
    lucid.manual_seed(0)
    model = CausalWanModel(CFG).eval()
    for p in model.parameters():  # zero-initialised branches would hide differences
        if float(p.abs().sum().item()) == 0.0:
            p.data = lucid.randn(*p.shape) * 0.05
    model.to(device)
    _, preset = presets.load_preset("03")
    data = {k: v for k, v in preset.items() if not k.startswith("_")}
    data.update(
        height=64, width=96, num_frames=33
    )  # 9 latent frames = 3 chunks of 2x3 tokens
    job = Job.from_preset("toy", presets.validate(data, presets.PRESET_DIR), seed=3)
    features = lucid.randn(1, job.num_chunks, 16, 64).to(device)
    source = lucid.randn(1, 1, 48, 4, 6).to(device)
    return model, job, features, source


def _run(model, job, features, source, probes, device):
    pipe = ReMindPipeline(device=device, dtype=lucid.float32)
    pipe.generator = model
    pipe.probes = probes
    chunks = list(
        pipe.rollout(job, features, source, NoiseSource(job.seed, lucid.float32))
    )
    return lucid.cat([c.latents for c in chunks], dim=1).to("cpu")


@pytest.mark.parametrize("device", DEVICES)
def test_probes_off_is_bit_identical(device):
    model, job, features, source = _setup(device)
    plain = _run(model, job, features, source, None, device)
    assert (
        ProbeSet.from_dict({}) is None and ProbeSet.from_dict({"gap_frames": 0}) is None
    )
    assert (
        (_run(model, job, features, source, ProbeSet(), device) == plain).all().item()
    )
    # Attention logging observes; it must not change the output either.
    logged = ProbeSet(attention_layers=[0, 2])
    assert (_run(model, job, features, source, logged, device) == plain).all().item()
    report = logged.report()
    assert sorted(report["attention_mass"]) == ["1", "2"]  # chunk 0 has no history
    for chunk, layers in report["attention_mass"].items():
        for rows in layers.values():
            for head in rows:  # the mass over all keys (history + own chunk) sums to 1
                assert sum(head) == pytest.approx(1.0, abs=1e-3)
    assert all(model.blocks[i].self_attn.probe is None for i in range(CFG.num_layers))


def test_hooks_change_the_output():
    model, job, features, source = _setup("cpu")
    plain = _run(model, job, features, source, None, "cpu")
    for probe in (
        ProbeSet(gap_from_chunk=1, gap_frames=5),
        ProbeSet(knockout={2: [0]}),
        ProbeSet(replace={0: "noise"}),
        ProbeSet(replace={1: "chunk:0"}),
    ):
        out = _run(model, job, features, source, probe, "cpu")
        assert (
            (out[:, :3] == plain[:, :3]).all().item()
        )  # chunk 0 is generated before any hook acts
        assert not (out == plain).all().item(), probe


def test_sweeps():
    assert interruption_phases(7, 2, 1) == [
        "none",
        "none",
        "enter",
        "hold",
        "exit",
        "recover",
        "recover",
    ]
    assert interruption_phases(7, 2, -1) == ["none"] * 7
    _, preset = presets.load_preset("01")
    variants = interruption_sweep(preset, [-1, 0, 1, 2])
    assert [v["sweep"]["hold_chunks"] for v in variants] == [-1, 0, 1, 2]
    # hold = 1 is the released preset's own schedule (enter 24-30, hold 30-42, exit 42-48).
    job = Job.from_preset("p", preset)
    assert (
        presets.validate(variants[2], presets.PRESET_DIR)["control"]["prompts"]
        == job.prompts
    )
    assert [v["seed"] for v in seed_sweep(preset, [1, 2])] == [1, 2]
