"""Reference Wan2.2 VAE with the released weights: latents and pixels in fp32 and bf16.

    .venv-ref/bin/python tests/ref/vae_upstream.py

Upstream runs its VAE in bf16 (``pipeline.to(dtype=bf16)``). torch's CPU
bf16 convolutions are about three orders of magnitude slower than fp32 (one
832x480 frame did not finish encoding in 25 minutes; fp32 takes about a
second), so the full-resolution reference is fp32 and bf16 is run on a
64x96 crop — enough to compare bf16 with bf16 and to choose the VAE dtype
on Metal with numbers. Inputs are the first frame of preset 01 and the
21-frame prefix of preset 08, prepared exactly as ``remind_inference.py``
does (PIL / imageio). Writes ``tests/fixtures/vae_upstream_large.safetensors``
(not in git).
"""

import _upstream  # noqa: F401
import time

import imageio
import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file

from utils.wan_wrapper import WanVAEWrapper
from wan.modules.vae_wan22 import WanVideoVAE38

ASSETS = _upstream.REFERENCE / "examples" / "assets"
out = {}

wrapper = WanVAEWrapper.__new__(WanVAEWrapper)
torch.nn.Module.__init__(wrapper)
vae = WanVideoVAE38()
state = load_file(str(_upstream.CHECKPOINTS / "Wan2.2-TI2V-5B" / "Wan2.2_VAE.safetensors"))
vae.model.load_state_dict(state, strict=True)  # shim: the port stores the .pth as SafeTensors
wrapper.mean, wrapper.std = vae.mean.float(), vae.std.float()
wrapper.model = vae.model.eval().requires_grad_(False)
wrapper.z_dim, wrapper.upsampling_factor = 48, 16

image = ImageOps.exif_transpose(Image.open(ASSETS / "latte_art_espresso.png")).convert("RGB").resize((832, 480), Image.Resampling.LANCZOS)
frame = torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255.0).permute(2, 0, 1)
out["image_u8"] = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())

reader = imageio.get_reader(str(ASSETS / "cake_decorating_prefix_21f.mp4"))
frames = [np.asarray(Image.fromarray(f.astype(np.uint8)).convert("RGB").resize((832, 480), Image.Resampling.LANCZOS), dtype=np.uint8).copy() for f in reader]
reader.close()
out["video_u8"] = torch.from_numpy(np.stack(frames[:21]))
clip = torch.from_numpy(np.stack(frames[:21]).astype(np.float32) / 255.0).permute(3, 0, 1, 2)

with torch.no_grad():
    # fp32, full resolution: image, 21-frame clip, and the decode of the clip's first 3 latent frames.
    t0 = time.time()
    out["fp32.image_latent"] = wrapper.encode_to_latent(frame.unsqueeze(1).unsqueeze(0).mul(2.0).sub(1.0))
    out["fp32.image5_latent"] = wrapper.encode_to_latent(frame.unsqueeze(1).repeat(1, 5, 1, 1).unsqueeze(0).mul(2.0).sub(1.0))
    # The inputs upstream feeds are rounded to bf16 before the [-1, 1] mapping; do the same here.
    vid = clip.unsqueeze(0).to(torch.bfloat16).mul(2.0).sub(1.0).float()
    out["fp32.video_latent"] = wrapper.encode_to_latent(vid)  # (1, 6, 48, 30, 52)
    print("fp32 encode", round(time.time() - t0, 1), "s", flush=True)
    t0 = time.time()
    out["fp32.pixels"] = wrapper.decode_to_pixel(out["fp32.video_latent"][:, :3], use_cache=False).half()
    print("fp32 decode", round(time.time() - t0, 1), "s", flush=True)

    save_file({k: v.contiguous() for k, v in out.items()}, str(_upstream.FIXTURES / "vae_upstream_large.safetensors"))
    print("saved the fp32 part", flush=True)

    # fp32 and bf16 on a 64x96 crop (5 frames -> 2 latent frames -> 5 frames).
    crop = clip[:, :5, 200:264, 380:476].unsqueeze(0)
    out["crop_u8"] = out["video_u8"][:5, 200:264, 380:476]
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        wrapper.model.to(dtype)
        t0 = time.time()
        x = crop.to(torch.bfloat16).mul(2.0).sub(1.0).to(dtype)
        out[f"crop.{name}.latent"] = wrapper.encode_to_latent(x)
        print(name, "crop encode", round(time.time() - t0, 1), "s", flush=True)
        save_file({k: v.contiguous() for k, v in out.items()}, str(_upstream.FIXTURES / "vae_upstream_large.safetensors"))
        z = out["crop.fp32.latent"].to(dtype)  # the same latents for both dtypes
        out[f"crop.{name}.pixels"] = wrapper.decode_to_pixel(z, use_cache=False).float()
        print(name, "crop encode + decode", round(time.time() - t0, 1), "s", flush=True)

save_file({k: v.contiguous() for k, v in out.items()}, str(_upstream.FIXTURES / "vae_upstream_large.safetensors"))
print({k: tuple(v.shape) for k, v in out.items()})
