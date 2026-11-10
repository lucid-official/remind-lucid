"""The primary parity reference: upstream ReMind itself, on the CPU, with the released weights.

    .venv-ref/bin/python tests/ref/remind_upstream.py [--preset 01_latte_occluder_recovery] [--chunks 2]

Runs ``load_dmd_ema_pipeline`` and ``DMDInferenceRollout`` of ``reference/``
unmodified in bf16, with the two shims of ``_upstream.py`` (``cuda -> cpu``;
no flash-attn, so upstream's own SDPA fallback). What else differs from
``inference.py``, none of it numerical:

* the model folder is ``checkpoints/_ref_model``: links to the files the
  port downloaded, with the VAE re-saved as the ``.pth`` upstream opens. It
  holds no Wan DiT shards — the ReMind generator overwrites all 975 tensors
  (upstream's own coverage check reports 100 %);
* the text encoder is dropped after the prompts are encoded (36 GB machine);
* only the frames the known prefix depends on are VAE-encoded (the encoder
  is causal; ``vae_upstream.py`` checks that on the released weights);
* **the VAE runs in fp32, not bf16.** This one is numerical. torch's CPU
  bf16 convolutions are far slower than fp32 (``vae_upstream.py``: 40x on a
  small crop; a single 832x480 frame did not finish encoding in 25 minutes),
  which puts a bf16 decode of 21 frames at hours. The VAE's input is still rounded
  to bf16 as upstream does, and its latents are cast to bf16 for the
  generator. bf16-vs-bf16 VAE agreement is measured separately on a crop;
* ``--chunks N`` rolls out the first N chunks only (later chunks do not
  influence earlier ones).

Dumps, into ``tests/fixtures/remind_upstream_<preset>_large.safetensors``:
the initial noise and every re-noising draw, UMT5 features of the unique
chunk prompts, the encoded first-frame latent, every DiT block output of
chunk 0 / step 0, K/V of three blocks after chunk 0, the final latents of
each chunk, the decoded frames, and samples of the LoRA-merged weights.
"""

import _upstream  # noqa: F401
import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="01_latte_occluder_recovery")
parser.add_argument("--chunks", type=int, default=2)
parser.add_argument("--no-decode", action="store_true")
args = parser.parse_args()

from pipeline.cache_adapter import InferencePipelineAdapter  # noqa: E402
from pipeline.dmd_rollout import DMDInferenceRollout  # noqa: E402
from pipeline.known_context import build_known_context  # noqa: E402
from pipeline import remind_inference as R  # noqa: E402

CKPT = _upstream.CHECKPOINTS
WAN = CKPT / "Wan2.2-TI2V-5B"
REF_MODEL = CKPT / "_ref_model"
REF_MODEL.mkdir(exist_ok=True)
for name in ("config.json", "google", "models_t5_umt5-xxl-enc-bf16.safetensors"):
    if not (REF_MODEL / name).exists():
        os.symlink(WAN / name, REF_MODEL / name)
if not (REF_MODEL / "Wan2.2_VAE.pth").exists():
    torch.save(
        load_file(str(WAN / "Wan2.2_VAE.safetensors")),
        str(REF_MODEL / "Wan2.2_VAE.pth"),
    )

out, log = {}, {}
t_all = time.time()


def stamp(name, t0):
    log[name] = round(time.time() - t0, 1)
    print(f"[ref] {name}: {log[name]} s", flush=True)


device, dtype = torch.device("cpu"), torch.bfloat16
preset_path, preset = R.load_preset(
    _upstream.REFERENCE / "examples" / "presets" / f"{args.preset}.yaml"
)
frames, height, width = (
    int(preset.get("num_frames", 81)),
    int(preset.get("height", 480)),
    int(preset.get("width", 832)),
)
chunk_size, seed = int(preset.get("chunk_size", 3)), int(preset["seed"])

t0 = time.time()
pipeline = R.load_dmd_ema_pipeline(
    config_path=_upstream.REFERENCE / "configs" / "model_5b.yaml",
    model_folder=REF_MODEL,
    base_checkpoint=CKPT / "ReMind" / "ReMind-5B.safetensors",
    ema_checkpoint=CKPT / "ReMind" / "ReMind-5b-dmd-ema.safetensors",
    device=device,
    dtype=dtype,
    lora_rank=128,
    lora_alpha=128,
)
stamp("load", t0)

model = pipeline.generator.model
for block in (0, 29):
    for name, module in model.blocks[block].named_modules():
        if isinstance(module, torch.nn.Linear) and not name.endswith(
            "camera_phase_mlp.proj"
        ):
            out[f"merged.blocks.{block}.{name}.slice"] = module.weight[
                :256, :256
            ].clone()
            out[f"merged.blocks.{block}.{name}.sum"] = (
                module.weight.double().sum().reshape(1)
            )

# -- inputs ------------------------------------------------------------------
task, seed_latent_frames = str(preset["task"]), int(preset["seed_latent_frames"])
seed_pixel_frames = 1 + 4 * (seed_latent_frames - 1)
t0 = time.time()
if task == "i2v":
    image = R._resize_input(preset["_input_image_path"], width, height)
    source_video = R.image_to_repeated_video(
        image, frames=seed_pixel_frames, device=device, dtype=dtype
    )
else:
    source_video, _ = R.video_prefix_to_padded_video(
        preset["_input_video_path"],
        seed_latent_frames=seed_latent_frames,
        frames=seed_pixel_frames,
        width=width,
        height=height,
        device=device,
        dtype=dtype,
    )
pipeline.vae.model.float()
prefix = pipeline.vae.encode_to_latent(source_video.float()).to(
    device=device, dtype=dtype
)
stamp("vae_encode", t0)
out["source_prefix"] = prefix
latent_frames = (frames - 1) // 4 + 1
latent_frames -= latent_frames % chunk_size
source_latent = torch.zeros(1, latent_frames, *prefix.shape[2:], dtype=dtype)
source_latent[:, :seed_latent_frames] = prefix[:, :seed_latent_frames]

pixel_extrinsics, pixel_intrinsics = R.build_pixel_cameras(
    preset["control"], frames=frames, height=height, width=width
)
viewmats, intrinsics = R.pixel_to_latent_cameras(
    pixel_extrinsics, pixel_intrinsics, latent_frames
)
out["viewmats"], out["intrinsics"] = viewmats, intrinsics
prompt = R.build_prompt(
    str(preset["prompt"]),
    preset["control"],
    pixel_frames=frames,
    latent_frames=latent_frames,
    chunk_size=chunk_size,
)

t0 = time.time()
with torch.no_grad():
    conditional = pipeline.text_encoder(text_prompts=[prompt])
conditional["prompt_chunk_size"] = chunk_size
stamp("text_encoder", t0)
unique = list(dict.fromkeys(prompt))
for i, text in enumerate(unique):
    out[f"t5_{i}"] = conditional["prompt_embeds"][0, prompt.index(text)].clone()
pipeline.text_encoder = None

# -- rollout -----------------------------------------------------------------
denoising_steps = [int(v) for v in pipeline.denoising_step_list.tolist() if int(v) > 0]
adapter = InferencePipelineAdapter(pipeline, chunk_size=chunk_size, context_timestep=0)
rollout = DMDInferenceRollout(
    adapter,
    pipeline.scheduler,
    denoising_steps,
    chunk_size=chunk_size,
    context_timestep=0,
)
known_latents, known_mask = build_known_context(source_latent, seed_latent_frames)
torch.manual_seed(seed)
noise = torch.randn_like(source_latent)
out["noise.initial"] = noise

n = min(args.chunks, latent_frames // chunk_size) * chunk_size
state = {"calls": 0, "draws": 0}
real_randn_like = torch.randn_like


def recording_randn_like(x, *a, **k):
    value = real_randn_like(x, *a, **k)
    chunk, step = divmod(state["draws"], len(denoising_steps) - 1)
    out[f"noise.renoise_{chunk}_{step}"] = value.clone()
    state["draws"] += 1
    return value


torch.randn_like = recording_randn_like


def block_hook(index):
    def hook(module, inputs, output):
        if state["calls"] == 0:
            out[f"c0s0.block_{index}"] = output.detach().clone()
        if index == len(model.blocks) - 1:
            print(
                f"[ref] forward {state['calls']} done at {time.time() - t_all:.0f} s",
                flush=True,
            )
            state["calls"] += 1

    return hook


for i, blk in enumerate(model.blocks):
    blk.register_forward_hook(block_hook(i))

real_encode = adapter.encode_chunk_to_cache


def encode_and_dump(chunk_x0, *, chunk_index, **kwargs):
    real_encode(chunk_x0, chunk_index=chunk_index, **kwargs)
    if chunk_index == 0:
        tokens = adapter.tokens_per_frame * chunk_size
        for b in (0, 14, 29):
            out[f"kv.k_{b}"] = adapter.kv_cache[b]["k"][:, :tokens].clone()
            out[f"kv.v_{b}"] = adapter.kv_cache[b]["v"][:, :tokens].clone()


adapter.encode_chunk_to_cache = encode_and_dump

t0 = time.time()
generated = rollout.rollout(
    noise[:, :n],
    conditional,
    {"viewmats": viewmats[:, :n], "Ks": intrinsics[:, :n]},
    known_latents=known_latents[:, :n],
    known_mask=known_mask[:, :n],
    exit_indices=[len(denoising_steps) - 1] * (n // chunk_size),
)
stamp("rollout", t0)
torch.randn_like = real_randn_like
adapter.clear_cache()
out["latents"] = generated

if not args.no_decode:
    t0 = time.time()
    with torch.no_grad():
        decoded = pipeline.vae.decode_to_pixel(generated.float(), use_cache=False)
    out["pixels"] = (
        decoded.float().mul(0.5).add(0.5).clamp(0, 1)[0].half()
    )  # (T, 3, H, W) in [0, 1]
    stamp("vae_decode", t0)

path = _upstream.FIXTURES / f"remind_upstream_{args.preset}_large.safetensors"
save_file(
    {k: v.contiguous() for k, v in out.items()},
    str(path),
    metadata={
        "prompts": json.dumps(unique),
        "schedule": json.dumps(prompt),
        "timings": json.dumps(log),
        "chunks": str(n // chunk_size),
        "seed": str(seed),
        "torch": torch.__version__,
    },
)
print("[ref] wrote", path, json.dumps(log))
