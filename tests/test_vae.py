"""Wan2.2 VAE against ``wan/modules/vae_wan22.py`` at toy size (random fp32 weights)."""

import lucid
import pytest

from conftest import DEVICES, load_fixture, max_rel
from remind_lucid.models.wan22_vae import VAEConfig, WanVAE, patchify, unpatchify


def _tiny(ref, device):
    vae = WanVAE(VAEConfig(dim=16, dec_dim=16), scale_dtype=lucid.float32).eval()
    vae.load_state_dict(
        {k[6:]: v for k, v in ref.items() if k.startswith("state.")}, strict=True
    )
    return vae.to(device)


def test_patchify_roundtrip():
    x = lucid.arange(2 * 3 * 2 * 4 * 6, dtype=lucid.float32).reshape(2, 3, 2, 4, 6)
    assert (unpatchify(patchify(x, 2), 2) == x).all().item()


@pytest.mark.parametrize("device", DEVICES)
def test_encode(device):
    ref = load_fixture("tiny_vae.safetensors")
    vae = _tiny(ref, device)
    with lucid.no_grad():
        z = vae.encode(ref["video"].to(device))
        z1 = vae.encode(ref["video"][:, :, :1].to(device))
    assert max_rel(z, ref["latent"]) < 1e-4
    assert max_rel(z1, ref["latent_first"]) < 1e-4
    # Causality: the first latent frame of the full encode is the single-frame encode.
    assert max_rel(z[:, :, :1], z1) < 1e-5


@pytest.mark.parametrize("device", DEVICES)
def test_decode_full_and_streamed(device):
    ref = load_fixture("tiny_vae.safetensors")
    vae = _tiny(ref, device)
    z = ref["latent"].to(device)
    with lucid.no_grad():
        full = vae.decode(z)
        state = vae.new_state()
        parts = [
            vae.decode_stream(z[:, :, i : i + 1], state) for i in range(z.shape[2])
        ]
        streamed = lucid.cat(parts, dim=2)
    assert max_rel(full, ref["pixels"].clamp(-1, 1)) < 1e-4
    assert [p.shape[2] for p in parts] == [1, 4, 4]
    assert (
        (streamed == full).all().item()
    )  # one frame at a time is the same computation
