"""Reference Wan2.2 VAE with the released weights: latents and pixels in fp32 and bf16.

    .venv-ref/bin/python tests/ref/vae_upstream.py

Upstream runs its VAE in bf16 (``pipeline.to(dtype=bf16)``); both are dumped
so the port can decide fp32 vs bf16 on Metal with numbers. Inputs are the
first frame of preset 01 and the 21-frame prefix of preset 08, prepared
exactly as ``remind_inference.py`` does (PIL / imageio). Writes
``tests/fixtures/vae_upstream_large.safetensors`` (not in git).
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
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        wrapper.model.to(dtype)
        t0 = time.time()
        img = frame.unsqueeze(1).unsqueeze(0).to(dtype).mul(2.0).sub(1.0)  # (1, 3, 1, H, W)
        out[f"{name}.image_latent"] = wrapper.encode_to_latent(img)
        vid = clip.unsqueeze(0).to(dtype).mul(2.0).sub(1.0)
        if name == "fp32":
            lat = wrapper.encode_to_latent(vid)  # (1, 6, 48, 30, 52), float
            out["fp32.video_latent"] = lat
        print(name, "encode", round(time.time() - t0, 1), "s", flush=True)
        t0 = time.time()
        z = out["fp32.video_latent"][:, :3].to(dtype)  # the same latents for both dtypes
        out[f"{name}.pixels"] = wrapper.decode_to_pixel(z, use_cache=False).half()  # (1, 9, 3, 480, 832)
        print(name, "decode", round(time.time() - t0, 1), "s", flush=True)

save_file({k: v.contiguous() for k, v in out.items()}, str(_upstream.FIXTURES / "vae_upstream_large.safetensors"))
print({k: tuple(v.shape) for k, v in out.items()})
