"""UMT5 encoder against ``wan/modules/t5.py`` at toy size (random fp32 weights)."""

import lucid
import pytest

from conftest import DEVICES, load_fixture, max_rel
from remind_lucid.models.umt5 import UMT5Config, UMT5Encoder, relative_position_buckets


def test_relative_buckets_match_torch():
    ref = load_fixture("tiny_umt5.safetensors")
    ours = lucid.tensor(relative_position_buckets(512), dtype=lucid.int64)
    assert (ours == ref["buckets"]).all().item()


@pytest.mark.parametrize("device", DEVICES)
def test_tiny_encoder(device):
    ref = load_fixture("tiny_umt5.safetensors")
    cfg = UMT5Config(vocab_size=100, dim=64, dim_attn=64, dim_ffn=128, num_heads=4, num_layers=2)
    model = UMT5Encoder(cfg).eval()
    model.load_state_dict({k[6:]: v for k, v in ref.items() if k.startswith("state.")}, strict=True)
    model.to(device)
    with lucid.no_grad():
        out = model(ref["ids"], ref["mask"])
    assert max_rel(out, ref["out"]) < 1e-4
