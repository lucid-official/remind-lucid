"""ReMind 5B DMD-EMA inference: ``pipeline/remind_inference.py`` + ``dmd_rollout.py`` + ``cache_adapter.py``.

:meth:`ReMindPipeline.rollout` is ``DMDInferenceRollout.rollout`` turned into
a generator: it yields each chunk of three latent frames the moment it is
final, so a caller can decode and show frames while later chunks are still
being denoised. The arithmetic and the order of random draws follow the
reference:

1. the input image (I2V) or 21-frame video prefix (V2V) is VAE-encoded; its
   first 1 (or 6) latent frames are *known*: they overwrite the noise, the
   prediction and the re-noised sample at every step, and enter the network
   at timestep 0 beside the noisy frames of their chunk;
2. each chunk is denoised at ``[1000, 938, 833, 625]`` with the chunk's own
   text context and cameras: the flow becomes ``x0``, and between steps
   ``x0`` is re-noised with fresh noise to the next timestep. There is no
   classifier-free guidance on this path — ``guidance_scale`` and
   ``negative_prompt`` of the config are never read;
3. the clean chunk is run once more at timestep 0 with its K/V written to
   the cache (``encode_chunk_to_cache``), where every later chunk attends to
   it. Nothing is ever evicted: chunk 6 reads 18 frames = 7,020 tokens.

Staged residency keeps the peak far below the sum of the parts: UMT5
(11 GB) is loaded only when a prompt is not in the disk cache and is
released before the generator (11 GB) is loaded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import lucid

from remind_lucid import camera, catalog, presets
from remind_lucid.io import media
from remind_lucid.models.wan22_vae import StreamState, WanVAE
from remind_lucid.models.wan_dit import CausalWanModel, ChunkGeometry, KVCache
from remind_lucid.scheduler import FlowMatchSchedule
from remind_lucid.text.encoder import PromptEncoder

Tensor = lucid.Tensor
DTYPES = {"bf16": lucid.bfloat16, "fp16": lucid.float16, "fp32": lucid.float32}
# The VAE runs in fp32 here although the reference runs it in bf16: see the
# README's "Lucid findings" (half-precision conv on Metal) for the measurement.
VAE_DTYPE = lucid.float32


def build_on(device: str, dtype, factory: Callable[[], lucid.nn.Module]) -> lucid.nn.Module:  # fmt: skip
    """Construct a module directly on ``device``/``dtype``; its random init is never evaluated."""
    prev = lucid.get_default_device(), lucid.get_default_dtype()
    lucid.set_default_device(device)
    lucid.set_default_dtype(dtype)
    try:
        return factory()
    finally:
        lucid.set_default_device(prev[0])
        lucid.set_default_dtype(prev[1])


def memory_gb() -> float:
    return lucid.metal.memory_allocated() / 1e9 if lucid.metal.is_available() else 0.0


def peak_memory_gb() -> float:
    return lucid.metal.max_memory_allocated() / 1e9 if lucid.metal.is_available() else 0.0  # fmt: skip


def load_vae(device: str = "metal", dtype=VAE_DTYPE, path: Path | None = None) -> WanVAE:  # fmt: skip
    model = build_on(device, dtype, WanVAE)
    weights = lucid.load_safetensors(str(path or catalog.vae_path()), device=device)
    weights = {k.removeprefix("model."): (v if v.dtype == dtype else v.to(dtype)) for k, v in weights.items()}  # fmt: skip
    model.load_state_dict(weights, strict=True, assign=True)
    return model.eval()


def load_generator(device: str = "metal", dtype=lucid.bfloat16, lora: bool = True, log=None) -> CausalWanModel:  # fmt: skip
    """``load_dmd_ema_pipeline``: the ReMind-5B generator, cast to ``dtype``, *then* the DMD EMA LoRA merged.

    The released generator file supplies every tensor of the model (strict
    load), so the Wan2.2 base weights the reference loads first are entirely
    overwritten and are not read here.
    """
    model = build_on(device, dtype, CausalWanModel)
    state = lucid.load_safetensors(str(catalog.generator_path()), device=device)
    state = {catalog.clean_key(k): (v if v.dtype == dtype else v.to(dtype)) for k, v in state.items()}  # fmt: skip
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.eval()
    if lora:
        adapter = lucid.load_safetensors(str(catalog.lora_path()), device=device)
        merged = model.merge_lora(adapter, alpha=catalog.LORA_ALPHA, rank=catalog.LORA_RANK)  # fmt: skip
        del adapter
        if log is not None:
            log(f"merged DMD EMA adapter into {merged} linear layers (rank {catalog.LORA_RANK})")  # fmt: skip
    if device == "metal":
        lucid.metal.empty_cache()
    return model


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #


class NoiseSource:
    """The random draws of one rollout, in the reference's order.

    ``initial`` is ``torch.randn_like(source_latent)``; ``renoise`` is the
    ``randn_like`` between two denoising steps of a chunk. Lucid's generator
    is not torch's, so a seed gives a different (equally valid) video than
    upstream; :class:`InjectedNoise` replays recorded tensors for parity.
    """

    def __init__(self, seed: int, dtype=lucid.bfloat16) -> None:
        self.generator = lucid.manual_seed(int(seed))
        self.dtype = dtype

    def _draw(self, shape) -> Tensor:
        return lucid.randn(*shape, dtype=lucid.float32, device="cpu", generator=self.generator).to(self.dtype)  # fmt: skip

    def initial(self, shape) -> Tensor:
        return self._draw(shape)

    def renoise(self, chunk: int, step: int, shape) -> Tensor:
        return self._draw(shape)


class InjectedNoise:
    """Replays noise recorded from the reference (``initial``, ``renoise_<chunk>_<step>``)."""

    def __init__(self, tensors: dict[str, Tensor]) -> None:
        self.tensors = tensors

    def initial(self, shape) -> Tensor:
        return self.tensors["initial"]

    def renoise(self, chunk: int, step: int, shape) -> Tensor:
        return self.tensors[f"renoise_{chunk}_{step}"]


# --------------------------------------------------------------------------- #
# Job
# --------------------------------------------------------------------------- #


@dataclass
class Job:
    """One resolved generation: everything derived from a preset before any weight is loaded."""

    preset: dict
    preset_path: str
    prompts: list[str]
    seed: int
    frames: int
    height: int
    width: int
    fps: int
    chunk_size: int
    latent_frames: int
    seed_latent_frames: int
    pixel_extrinsics: Tensor
    viewmats: Tensor  # (1, latent_frames, 4, 4)
    intrinsics: Tensor  # (1, latent_frames, 3, 3)

    @property
    def num_chunks(self) -> int:
        return self.latent_frames // self.chunk_size

    @property
    def seed_pixel_frames(self) -> int:
        return 1 + 4 * (self.seed_latent_frames - 1)

    @classmethod
    def from_preset(cls, preset_path, preset: dict, seed: int | None = None) -> "Job":
        g = presets.geometry(preset)
        if g["latent_frames"] <= 1:
            raise RuntimeError(f"invalid latent length {g['latent_frames']}")
        if g["height"] % 32 or g["width"] % 32:
            raise ValueError("height and width must be multiples of 32 (VAE x16, patch x2)")  # fmt: skip
        prompts = presets.build_prompt(str(preset["prompt"]), preset["control"], pixel_frames=g["frames"], latent_frames=g["latent_frames"], chunk_size=g["chunk_size"])  # fmt: skip
        ext, ks = camera.build_pixel_cameras(preset["control"], frames=g["frames"], height=g["height"], width=g["width"])  # fmt: skip
        view, k_lat = camera.pixel_to_latent(ext, ks, g["latent_frames"])
        return cls(
            preset=preset,
            preset_path=str(preset_path),
            prompts=prompts,
            seed=int(preset.get("seed", 20260711) if seed is None else seed),
            frames=g["frames"],
            height=g["height"],
            width=g["width"],
            fps=g["fps"],
            chunk_size=g["chunk_size"],
            latent_frames=g["latent_frames"],
            seed_latent_frames=int(preset["seed_latent_frames"]),
            pixel_extrinsics=ext,
            viewmats=view,
            intrinsics=k_lat,
        )

    def metadata(self) -> dict:
        """The fields ``generate_from_preset`` writes next to its video."""
        task = str(self.preset["task"])
        input_key = "input_image" if task == "i2v" else "input_video"
        return {
            "name": self.preset["name"],
            "preset": self.preset_path,
            "task": task,
            input_key: str(self.preset[f"_{input_key}_path"]),
            "control_kind": self.preset["control"]["kind"],
            "seed": self.seed,
            "num_frames": self.frames,
            "latent_frames": self.latent_frames,
            "seed_latent_frames": self.seed_latent_frames,
            "seed_pixel_frames": self.seed_pixel_frames,
            "rollout_steps": list(catalog.DENOISING_STEPS),
            "prompt_mode": "chunk_local",
            "prompt_schedule": self.prompts,
            "camera": camera.camera_metrics(self.pixel_extrinsics),
            "website_case": self.preset.get("website_case"),
        }


@dataclass
class Chunk:
    """One finished chunk."""

    index: int
    latents: Tensor  # (1, chunk_size, 48, h, w)
    seconds: float
    kv_tokens: int  # history tokens this chunk attended to
    known: bool  # every frame came from the input (V2V prefix): no network call was needed


@dataclass
class Timings:
    stages: dict[str, float] = field(default_factory=dict)
    memory: dict[str, float] = field(default_factory=dict)
    chunks: list[float] = field(default_factory=list)

    def mark(self, name: str, t0: float) -> None:
        self.stages[name] = round(time.time() - t0, 3)
        self.memory[name] = round(memory_gb(), 3)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


class ReMindPipeline:
    """Generator + VAE + prompt encoder with staged residency.

    Parameters
    ----------
    device : str
        ``"metal"`` (or ``"cpu"`` for tests at toy sizes).
    dtype
        Generator and text-encoder dtype; the reference offers bf16 (default) and fp16.
    accelerate : bool
        Use the ``lucid.compile`` programs of :mod:`remind_lucid.accel` where they exist.
    """

    def __init__(self, device: str = "metal", dtype=lucid.bfloat16, accelerate: bool = False, log: Callable[[str], None] | None = None) -> None:  # fmt: skip
        self.device, self.dtype, self.accelerate = device, dtype, accelerate
        self.log = log or (lambda message: None)
        self.schedule = FlowMatchSchedule(catalog.TIMESTEP_SHIFT, catalog.NUM_TRAIN_TIMESTEPS)  # fmt: skip
        self.steps = [int(s) for s in catalog.DENOISING_STEPS if int(s) > 0]
        self.prompt_encoder = PromptEncoder(device, dtype)
        self.generator: CausalWanModel | None = None
        self.vae: WanVAE | None = None
        self.timings = Timings()
        self.probes = None  # remind_lucid.probes.ProbeSet; None = the reference path
        self._decoder = None

    # -- stages ------------------------------------------------------------- #

    def encode_prompts(self, prompts: list[str], on_prompt=None) -> Tensor:
        """``(1, chunks, 512, 4096)`` UMT5 features; UMT5 is loaded only on a cache miss and released after."""
        t0 = time.time()
        features = self.prompt_encoder.encode(prompts, on_prompt)
        loaded = self.prompt_encoder.model is not None
        self.prompt_encoder.release()
        self.timings.mark("prompt_features" if loaded else "prompt_features_cached", t0)
        self.log(f"prompt features {tuple(features.shape)} ({'encoded' if loaded else 'cached'}), in use {memory_gb():.1f} GB")  # fmt: skip
        return features

    def ensure_vae(self) -> WanVAE:
        if self.vae is None:
            t0 = time.time()
            self.vae = load_vae(self.device)
            self.timings.mark("load_vae", t0)
        return self.vae

    def ensure_generator(self) -> CausalWanModel:
        if self.generator is None:
            t0 = time.time()
            self.generator = load_generator(self.device, self.dtype, log=self.log)
            self.timings.mark("load_generator", t0)
            self.log(f"generator ready, in use {memory_gb():.1f} GB")
        return self.generator

    def encode_source(self, job: Job) -> Tensor:
        """The known latent prefix ``(1, seed_latent_frames, 48, h, w)`` in the run dtype.

        The reference encodes all 81 frames (the image repeated, or the prefix
        padded with its last frame) and keeps the prefix. The encoder is
        causal, so encoding only the ``1 + 4 (n - 1)`` frames the prefix
        depends on gives the same latents (``tests/test_vae.py``).
        """
        t0 = time.time()
        vae = self.ensure_vae()
        if job.preset["task"] == "i2v":
            image = media.pil_resize(media.read_image(job.preset["_input_image_path"]), job.height, job.width)  # fmt: skip
            frames = image.unsqueeze(0)
        else:
            frames = media.read_video(job.preset["_input_video_path"], job.seed_pixel_frames, job.height, job.width)  # fmt: skip
            if frames.shape[0] < job.seed_pixel_frames:
                raise ValueError(f"V2V input has {frames.shape[0]} decoded frames; {job.seed_pixel_frames} are required for {job.seed_latent_frames} seed latents")  # fmt: skip
        with lucid.no_grad():
            video = media.frames_to_model(frames, self.dtype).to(self.device)
            latent = vae.encode(video).float().to(self.dtype)  # (1, 48, n, h, w)
            latent = latent.permute(0, 2, 1, 3, 4)
            lucid.eval(latent)
        self.timings.mark("vae_encode", t0)
        return latent

    # -- rollout ------------------------------------------------------------ #

    def rollout(self, job: Job, features: Tensor, source: Tensor, noise, taps: dict | None = None) -> Iterator[Chunk]:  # fmt: skip
        """``DMDInferenceRollout.rollout``, one :class:`Chunk` per iteration.

        Parameters
        ----------
        features : Tensor
            ``(1, chunks, 512, 4096)`` from :meth:`encode_prompts`.
        source : Tensor
            ``(1, seed_latent_frames, 48, h, w)`` known latents.
        noise
            A :class:`NoiseSource` (or :class:`InjectedNoise`).
        taps : dict, optional
            Receives intermediate tensors of chunk 0 / step 0 for parity tests.
        """
        model = self.ensure_generator()
        cs, steps = job.chunk_size, self.steps
        h, w = job.height // 16, job.width // 16
        grid = (h // 2, w // 2)
        shape = (1, job.latent_frames, 48, h, w)
        probes = self.probes
        cache = model.new_cache()
        self.cache = cache
        if probes is not None:
            probes.begin(self, job, cache)
        with lucid.no_grad():
            noise0 = noise.initial(shape).to(self.device)
            contexts: dict[str, Tensor] = {}
            for chunk in range(job.num_chunks):
                t0 = time.time()
                lo, hi = chunk * cs, (chunk + 1) * cs
                known = [f < job.seed_latent_frames for f in range(lo, hi)]
                text = job.prompts[chunk]
                if text not in contexts:
                    contexts[text] = model.embed_text(features[0, chunk : chunk + 1])
                    lucid.eval(contexts[text])
                context = contexts[text]
                positions = list(range(lo, hi))
                if probes is not None:
                    positions = probes.positions(chunk, positions)
                cam = camera.prepare_chunk(job.viewmats[:, lo:hi], job.intrinsics[:, lo:hi], float(job.width))  # fmt: skip
                geom = model.geometry(positions, cam, grid)
                if probes is not None:
                    probes.before_chunk(chunk)

                def overwrite(value: Tensor) -> Tensor:
                    parts = [source[:, lo + i : lo + i + 1] if known[i] else value[:, i : i + 1] for i in range(cs)]  # fmt: skip
                    return lucid.cat(parts, dim=1) if any(known) else value

                current = overwrite(noise0[:, lo:hi])
                predicted = None
                for step, tval in enumerate(steps):
                    timesteps = [0 if known[i] else tval for i in range(cs)]
                    if all(known):
                        # Every frame is overwritten by the input: the network's output is unused.
                        predicted = overwrite(current)
                    else:
                        tap = taps if (taps is not None and chunk == 0 and step == 0) else None  # fmt: skip
                        flow = self._denoise(model, current, timesteps, context, geom, cache, tap)  # fmt: skip
                        predicted = overwrite(self.schedule.x0_from_flow(flow, current, timesteps))  # fmt: skip
                    if step == len(steps) - 1:
                        break
                    fresh = noise.renoise(chunk, step, (cs, 48, h, w)).to(self.device).reshape(1, cs, 48, h, w)  # fmt: skip
                    current = overwrite(self.schedule.add_noise(predicted, fresh, [steps[step + 1]] * cs))  # fmt: skip
                    lucid.eval(current)
                lucid.eval(predicted)
                kv_tokens = cache.tokens()
                # encode_chunk_to_cache: one pass at context_timestep = 0 writes this chunk's K/V.
                model(predicted.permute(0, 2, 1, 3, 4), [0.0] * cs, context, geom, cache, write_key=chunk)  # fmt: skip
                if probes is not None:
                    probes.remember(chunk, predicted, context, geom)
                    probes.after_commit(chunk)
                seconds = time.time() - t0
                self.timings.chunks.append(round(seconds, 3))
                if taps is not None:
                    taps[f"latent_{chunk}"] = predicted
                yield Chunk(chunk, predicted, seconds, kv_tokens, all(known))
        if probes is not None:
            probes.end()

    def _denoise(self, model, current, timesteps, context, geom: ChunkGeometry, cache: KVCache, taps) -> Tensor:  # fmt: skip
        flow = model(current.permute(0, 2, 1, 3, 4), [float(t) for t in timesteps], context, geom, cache, taps=taps)  # fmt: skip
        flow = flow.permute(0, 2, 1, 3, 4)
        lucid.eval(flow)
        return flow

    # -- decode ------------------------------------------------------------- #

    def decoder(self) -> "StreamingDecoder":
        vae = self.ensure_vae()
        if self.accelerate:
            from remind_lucid.accel import CompiledDecoder

            return CompiledDecoder(vae)
        return StreamingDecoder(vae)

    def release_generator(self) -> None:
        self.generator = None
        self.cache = None
        if self.device == "metal":
            lucid.metal.empty_cache()


class StreamingDecoder:
    """Decodes latent chunks as they arrive; the state carries the VAE's causal histories."""

    def __init__(self, vae: WanVAE) -> None:
        self.vae = vae
        self.state: StreamState = vae.new_state()

    def reset(self) -> None:
        self.state = self.vae.new_state()

    def __call__(self, latents: Tensor) -> Tensor:
        """``(1, T, 48, h, w)`` latents -> ``(3, T', H, W)`` pixels in ``[-1, 1]``."""
        with lucid.no_grad():
            pixels = self.vae.decode_stream(latents.permute(0, 2, 1, 3, 4), self.state)
            lucid.eval(pixels)
        return pixels[0]
