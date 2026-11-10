"""The Wan2.2 VAE with the released weights against upstream's (``tests/ref/vae_upstream.py``).

Measures fp32 and bf16 on Metal against the fp32 reference at full
resolution, and bf16 against upstream's bf16 on a crop. The numbers are
written to ``outputs/vae_report.json``; they are what the VAE dtype was
chosen from (README, "Lucid findings").
"""

import json
import math

import lucid
import pytest

from conftest import FIXTURES, ROOT, cosine, max_rel
from remind_lucid import catalog, presets
from remind_lucid.io import media
from remind_lucid.pipeline import load_vae

FIXTURE = FIXTURES / "vae_upstream_large.safetensors"
pytestmark = pytest.mark.skipif(not (FIXTURE.exists() and catalog.vae_path().exists()), reason="needs the VAE weights and tests/ref/vae_upstream.py")  # fmt: skip
REPORT: dict = {}


def psnr(a, b) -> float:
    """PSNR of two [-1, 1] tensors on a [0, 1] scale."""
    mse = ((a.float().to("cpu") - b.float().to("cpu")) * 0.5).square().mean().item()
    return 10 * math.log10(1.0 / mse) if mse > 0 else float("inf")


@pytest.fixture(scope="module")
def ref():
    return lucid.load_safetensors(str(FIXTURE))


@pytest.fixture(scope="module", autouse=True)
def write_report():
    yield
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "vae_report.json").write_text(
        json.dumps(REPORT, indent=2) + "\n"
    )


def test_inputs_are_decoded_identically(ref):
    """ffmpeg here, PIL / imageio upstream: the 8-bit inputs must be the same pixels."""
    _, p1 = presets.load_preset("01")
    image = media.pil_resize(media.read_image(p1["_input_image_path"]), 480, 832)
    assert (image.to(lucid.int32) == ref["image_u8"].to(lucid.int32)).all().item()
    _, p8 = presets.load_preset("08")
    video = media.read_video(p8["_input_video_path"], 21, 480, 832)
    diff = (video.to(lucid.int32) - ref["video_u8"].to(lucid.int32)).abs()
    REPORT["video_prefix_decode"] = {"max_abs_diff_8bit": int(diff.max().item()), "mean_abs_diff_8bit": round(diff.float().mean().item(), 5)}  # fmt: skip
    assert diff.max().item() <= 2


def test_upstream_encoder_is_causal(ref):
    """Upstream's own latents: frame 0 of a 5-frame encode equals the 1-frame encode."""
    r = max_rel(ref["fp32.image5_latent"][:, :1], ref["fp32.image_latent"])
    REPORT["upstream_causality_max_rel"] = r
    assert r < 1e-5


@pytest.mark.parametrize("name", ["fp32", "bf16"])
def test_full_resolution(ref, name):
    dtype = {"fp32": lucid.float32, "bf16": lucid.bfloat16}[name]
    vae = load_vae("metal", dtype)
    # The fp32 reference builds mean and 1/std in fp32 (upstream builds them in the
    # dtype of its input); match that here so only the network's arithmetic is compared.
    vae.scale_dtype = lucid.float32
    with lucid.no_grad():
        image = media.frames_to_model(ref["image_u8"].unsqueeze(0), lucid.bfloat16 if name == "bf16" else lucid.float32).to("metal")  # fmt: skip
        z_image = vae.encode(image).float().permute(0, 2, 1, 3, 4)
        video = media.frames_to_model(ref["video_u8"], lucid.bfloat16).to("metal")
        z_video = vae.encode(video).float().permute(0, 2, 1, 3, 4)
        pixels = vae.decode(ref["fp32.video_latent"][:, :3].permute(0, 2, 1, 3, 4).to("metal").to(dtype))  # fmt: skip
        pixels = pixels.permute(0, 2, 1, 3, 4)
        lucid.eval(z_image, z_video, pixels)
    REPORT[f"{name}_vs_upstream_fp32"] = {
        "image_latent_cosine": round(cosine(z_image, ref["fp32.image_latent"]), 6),
        "image_latent_max_rel": round(max_rel(z_image, ref["fp32.image_latent"]), 6),
        "video_latent_cosine": round(cosine(z_video, ref["fp32.video_latent"]), 6),
        "video_latent_max_rel": round(max_rel(z_video, ref["fp32.video_latent"]), 6),
        "decode_psnr_db": round(psnr(pixels, ref["fp32.pixels"].clamp(-1, 1)), 2),
    }
    del vae
    lucid.metal.empty_cache()
    if name == "fp32":
        # The clip's input is rounded to bf16 on both sides; the image is not (fp32 reference input).
        assert max_rel(z_video, ref["fp32.video_latent"]) < 1e-4
        assert psnr(pixels, ref["fp32.pixels"].clamp(-1, 1)) > 60


def test_crop_bf16(ref):
    if "crop.bf16.pixels" not in ref:
        pytest.skip("the bf16 crop of vae_upstream.py has not finished")
    crop = media.frames_to_model(ref["crop_u8"], lucid.bfloat16).to("metal")
    z_ref = ref["crop.fp32.latent"].permute(0, 2, 1, 3, 4).to("metal")
    out = {
        "upstream_bf16_vs_upstream_fp32": {
            "latent_cosine": round(
                cosine(ref["crop.bf16.latent"], ref["crop.fp32.latent"]), 6
            ),
            "decode_psnr_db": round(
                psnr(ref["crop.bf16.pixels"], ref["crop.fp32.pixels"]), 2
            ),
        }
    }
    for name, dtype in (("fp32", lucid.float32), ("bf16", lucid.bfloat16)):
        vae = load_vae("metal", dtype)
        vae.scale_dtype = (
            dtype  # as upstream: the constants in the dtype the VAE runs in
        )
        with lucid.no_grad():
            z = vae.encode(crop).float().permute(0, 2, 1, 3, 4)
            pix = vae.decode(z_ref.to(dtype)).permute(0, 2, 1, 3, 4)
            lucid.eval(z, pix)
        for other in ("fp32", "bf16"):
            out[f"lucid_{name}_vs_upstream_{other}"] = {
                "latent_cosine": round(cosine(z, ref[f"crop.{other}.latent"]), 6),
                "decode_psnr_db": round(
                    psnr(pix, ref[f"crop.{other}.pixels"].clamp(-1, 1)), 2
                ),
            }
        del vae
        lucid.metal.empty_cache()
    REPORT["crop_64x96"] = out
