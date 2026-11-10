# Porting plan: ReMind-5B on Lucid

What was read in the upstream repository
([Applied-Intuition-Open-Source/ReMind](https://github.com/Applied-Intuition-Open-Source/ReMind),
commit `df1d52f`), what each piece became here, and every place where the
inference path depends on a constant, a dtype, a random draw or mutable
state. Statements about upstream behaviour are read from its code, not from
the paper.

Runtime: `lucid-dl==3.16.4` from PyPI, Python 3.14, Metal. Reference
environment: `.venv-ref` (torch 2.15.0, CPU).

## 1. Module map

| Upstream | Here | Notes |
|---|---|---|
| `inference.py` | `remind_lucid/generate.py` | Same flags (`--preset`, `--seed`, `--dtype`, `--output`, `--dry-run`) plus custom inputs and `--compile`. |
| `pipeline/remind_inference.py`: `load_preset`, `build_prompt`, `dry_run_summary`, event defaults | `remind_lucid/presets.py` | JSON instead of YAML (§7). |
| …: cameras (`build_pixel_cameras`, `_camera_npz_to_tensors`, `pixel_to_latent_cameras`, `camera_metrics`) | `remind_lucid/camera.py` | |
| …: `load_dmd_ema_pipeline`, `generate_from_preset` | `remind_lucid/pipeline.py` (`load_generator`, `ReMindPipeline`, `Job`) | |
| `pipeline/dmd_rollout.py` | `pipeline.py::ReMindPipeline.rollout` | A generator: one chunk per iteration. |
| `pipeline/cache_adapter.py` | `pipeline.py` (commit pass) + `models/wan_dit.py::KVCache` | |
| `pipeline/known_context.py` | `pipeline.py` (`known`, `overwrite`) | |
| `pipeline/prompt_schedule.py` | `remind_lucid/text/schedule.py` | Clause texts verbatim. |
| `pipeline/checkpoints.py` | `pipeline.py::load_generator`, `wan_dit.py::merge_lora`, `catalog.clean_key` | |
| `pipeline/causal_inference.py` | — | Only its constructor is used upstream (it builds the three models); its own `inference()` is a different, unused path. |
| `utils/wan_wrapper.py`: `WanDiffusionWrapper` | `wan_dit.py::CausalWanModel.forward` + `scheduler.py::x0_from_flow` | |
| …: `_select_chunk_prompt_embeds` | `pipeline.py` (one context per chunk) | |
| …: `WanTextEncoder` | `text/encoder.py` | Unique prompts encoded once; disk cache added. |
| …: `WanVAEWrapper` | `models/wan22_vae.py::WanVAE` | |
| `utils/scheduler.py::FlowMatchScheduler` | `scheduler.py::FlowMatchSchedule` | |
| `wan/modules/causal_model.py` | `models/wan_dit.py` | Inference (KV-cache) path only. |
| `wan/modules/model.py` (norms, cross-attention, sinusoid, RoPE) | `models/wan_dit.py` | |
| `prope/camera_rope.py` | `camera.py::prepare_chunk`, `wan_dit.py::_tiled_projection` | |
| `wan/modules/vae_wan22.py` | `models/wan22_vae.py` | `*_38` classes only. |
| `wan/modules/t5.py` | `models/umt5.py` | Encoder only. |
| `wan/modules/tokenizers.py` | `text/tokenizer.py` | |
| `utils/video_io.py` | `io/media.py` | ffmpeg subprocess. |
| `prompt_engineering/` (Qwen prompt rewriter) | — | Not ported: an optional authoring tool, not part of inference. Its 12 camera trajectories are bundled. |
| `wan/distributed/`, flex-attention masks, `dl3dv_inference.py`, `clip.py`, `vae.py` (Wan2.1) | — | Training or unused on this path. |

Other modules here: `hub.py`, `download.py`, `catalog.py` (weights),
`io/torch_zip.py`, `io/npy.py` (readers), `accel.py` (`lucid.compile`),
`server/` (web demo), `probes/` (research hooks).

## 2. Dataflow at 81 frames, 832×480

```
preset ──► 7 chunk prompts ──► UMT5-XXL (bf16) ──► (1, 7, 512, 4096)      padded rows zeroed
image  (480, 832, 3) u8 ─► /255 ─► bf16 ─► *2-1 ─► (1, 3, 1, 480, 832)
        patchify 2x2 ─► (1, 12, 1, 240, 416) ─► encoder (x8 space, x4 time) ─► (1, 96, 1, 30, 52)
        conv1, first 48 channels, (mu - mean) / std ─► known latent (1, 1, 48, 30, 52)
V2V:    21 frames ─► (1, 6, 48, 30, 52)
cameras (1, 81, 4, 4), (1, 81, 3, 3) ─► linspace(0, 80, 21).round() ─► (1, 21, 4, 4), (1, 21, 3, 3)
noise   (1, 21, 48, 30, 52) bf16

per chunk c (frames 3c .. 3c+2):
  x (1, 48, 3, 30, 52) ─► patch_embedding 1x2x2 ─► (1, 1170, 3072)        390 tokens per frame
  timesteps (3,) ─► sinusoid (3, 256) float64 ─► time_embedding (3, 3072) ─► time_projection (1, 3, 6, 3072)
  context (1, 512, 4096) ─► text_embedding ─► (1, 512, 3072)
  30 blocks: self-attention  q (1, 24, 1170, 128)  x  kv (1, 24, 1170·c + 1170, 128)
             cross-attention q (1, 24, 1170, 128)  x  kv (1, 24, 512, 128)
             feed-forward 3072 ─► 14336 ─► 3072
  head ─► (1, 1170, 192) ─► unpatchify ─► flow (1, 48, 3, 30, 52)
  x0 = xt - sigma_t * flow (float64) ─► re-noise ─► ... 4 steps ─► commit pass writes K/V (1170 tokens x 30 blocks)

KV history: 0, 1170, 2340, ... 7020 tokens (chunk 6); with the chunk's own 1170 keys, up to 8190.
decode: latent frame (1, 48, 1, 30, 52) ─► decoder ─► (1, 12, 1|4, 240, 416) ─► unpatchify ─► (1, 3, 1|4, 480, 832)
        21 latent frames ─► 1 + 4·20 = 81 frames
```

`head_dim = 3072 / 24 = 128`, one of the head sizes for which
`scaled_dot_product_attention` takes Lucid's fused MLX kernel.

## 3. Config constants and where they are read

| Constant | Value | Read at | Here |
|---|---|---|---|
| `denoising_step_list` | `[1000, 938, 833, 625]` | `CausalInferencePipeline.__init__` → `DMDInferenceRollout.steps` | `catalog.DENOISING_STEPS` |
| `warp_denoising_step` | `false` | same; the list is used as written | not needed |
| `timestep_shift` | `5.0` | `FlowMatchScheduler(shift=...)` | `catalog.TIMESTEP_SHIFT` |
| `num_train_timestep` | `1000` | **never read**: `FlowMatchScheduler` uses its own default (1000) | `catalog.NUM_TRAIN_TIMESTEPS` |
| `num_frame_per_block` | `3` | sets `generator.model.num_frame_per_block`, which the KV-cache forward never reads; the chunk size comes from the preset's `chunk_size` (3) | `Job.chunk_size` |
| `guidance_scale` | `3.0` | **never read** on the inference path | `catalog.GUIDANCE_SCALE` (unused) |
| `negative_prompt` | `'blurry, …'` | **never read**; it is not even encoded | `catalog.NEGATIVE_PROMPT` (unused) |
| `in_dim = out_dim` | `48` | patch embedding and head; `out_dim == 48` also selects VAE stride 16 for the intrinsics normalisation | `DiTConfig` |
| `model_type` | `ti2v` | selects `t2v_cross_attn` (no image branch) | `CrossAttention` |
| `cc_rope_mode` | `cc_full` | `CausalWanSelfAttention`: phase MLP on all 64 slots + `value_proj` + `prope_proj` | `CausalSelfAttention` |
| `prope_temporal_dim` | `8` | passed to `prope_qkv`, which only the `dual_prope` mode calls — **unused in `cc_full`** | not needed |
| `cc_phase_slots` | `16` | only the `cc_dual_*` modes mask slots — **unused in `cc_full`** (all 64 slots get a phase) | not needed |
| `degradation_control_dim` | `0` | no control embedding is built | not needed |
| `filter_list` | `img_emb.`, `cross_attn.k_img`, … | filters Wan base keys; matches nothing in Wan2.2-TI2V-5B | `catalog.FILTER_LIST` |
| `lora_rank`, `lora_alpha` | `128`, `128` | `configure_inference_lora` (scale 1.0) | `catalog.LORA_RANK/ALPHA` |
| `context_timestep` | `0` (preset default) | commit pass timestep; no noise is added at 0 | `rollout` |
| `text_len` | `512` | tokenizer `seq_len`; context padded/cut to it | `text/encoder.py`, `embed_text` |
| `frame_seq_length = 1560`, `max_num_context_frames = 6`, `seq_len = 1560·24` | — | attributes of the unused `inference()` path / an assert that is far from binding | not needed |

**Does the DMD path apply classifier-free guidance? No.** `DMDInferenceRollout`
calls the generator once per step with the positive context only;
`guidance_scale` and `negative_prompt` appear nowhere outside the YAML.

## 4. Checkpoint key coverage

| File | Tensors | Role |
|---|---|---|
| `ReMind-5B.safetensors` (bf16) | 975 | All of the generator: the 825 Wan2.2-TI2V-5B keys + per block `camera_phase_mlp.proj.weight` (64×14), `prope_proj.{weight,bias}`, `value_proj.{weight,bias}` (30 × 5 = 150). |
| Wan2.2 `diffusion_pytorch_model-0000{1,2,3}` (fp32, 20 GB) | 825 | Loaded first upstream, then **every one is overwritten** by the ReMind file (same names, same shapes). `filter_list` removes none. Needed here: **0 tensors**. `download.py::fetch_base_extra` re-derives this from the remote headers. |
| `ReMind-5b-dmd-ema.safetensors` (fp32) | 720 | `base_model.model.blocks.N.<m>.lora_{A,B}.weight` for the 12 linears of each block: `self_attn.{q,k,v,o,prope_proj,value_proj}`, `cross_attn.{q,k,v,o}`, `ffn.{0,2}`. `camera_phase_mlp.proj` is excluded. |
| `models_t5_umt5-xxl-enc-bf16.pth` | 242 | UMT5 encoder, strict. |
| `Wan2.2_VAE.pth` | 196 | VAE encoder + decoder, strict. |

Load order (`load_dmd_ema_pipeline`): Wan base → zero `camera_phase_mlp`,
`prope_proj`, `value_proj` → overlay ReMind-5B (coverage must be ≥ 90 %; it
is 100 %) → cast the whole pipeline to bf16 → cast LoRA A/B to bf16 → merge.
With full coverage the first two steps have no effect, so the port starts at
the overlay.

LoRA merge (`peft` 0.19, `merge_and_unload`): `W += (B @ A) * (alpha / r)`
with A, B already in the weight dtype. On CPU peft computes the product in
fp32 and casts; on CUDA the product is a bf16 matmul. The port does the
former (it is what the CPU reference does).

## 5. Float64, RNG and cache mutation

**Float64 sites**

| Site | Upstream | Here |
|---|---|---|
| `sinusoidal_embedding_1d` | float64 outer product, `cos`/`sin`, cast to the model dtype | Lucid CPU float64, cast, moved to Metal |
| `rope_params` | float64 angles, `polar` → complex128 | Lucid CPU float64 `cos`/`sin` tables |
| camera phasor × RoPE | complex128 table × complex64 phasor, cast to complex64 | float64 products on the CPU, cast to fp32 |
| `_convert_flow_pred_to_x0` | `xt - sigma * flow` in float64 | Lucid CPU float64 |
| `camera_metrics` | float64 | Lucid CPU float64 |

fp32 inside a bf16 model: RoPE rotation, `_apply_tiled_projmat`, `WanRMSNorm`,
`T5LayerNorm`'s scale, T5's softmax, `add_noise`, all camera matrices.

**Random draws** (in order): `torch.randn_like(source_latent)` once after
`torch.manual_seed(seed)`; then, per chunk, one `randn_like` after each of
the first three denoising steps (also for chunks that are entirely known).
`context_timestep = 0`, so the commit pass draws nothing. 1 + 7·3 = 22 draws.
The LoRA's random init (`fork_rng`) is overwritten by the checkpoint.

**Cache mutation**: `reset_cache` (allocate zeros), `encode_chunk_to_cache`
(write K/V of one chunk at `hist_tokens`, then `hist_tokens += 1170`),
`clear_cache`. `replace_chunk_in_cache` exists but the release path never
calls it. Denoising steps only read `[0, hist_tokens)`. The VAE's
`feat_cache` lists are the other mutable state.

## 6. Things in upstream that are easy to get wrong

* **Known frames ride along.** In chunk 0 of an I2V run, frame 0 is the
  encoded image at timestep 0 and frames 1–2 are noise at the step's
  timestep; all three go through the network together.
* **Camera translations are normalised per call**, over the 3 frames of the
  chunk (mean-centred, scaled by the largest distance, clamped at 0.01) — not
  over the trajectory.
* **Both intrinsics rows are divided by the image width** (832), and the
  principal point is dropped when `P` is built.
* **`viewmats` are camera-to-world** although `camera_rope.py` documents
  `P = lift(K) @ viewmats` for world-to-camera matrices. Used as they are.
* **Keys are cached after rotation**; values after the `value_proj`
  residual. Re-rotating history would be wrong.
* **Cross-attention has no padding mask**: all 512 positions are attended;
  padded positions hold `text_embedding(0)`.
* **The trajectory NPZ's own `resolution` is ignored** unless the preset has
  `source_resolution` (none does), so intrinsics are used unscaled.
* **Attention always runs in bf16**, whatever the model dtype
  (`attention(..., dtype=torch.bfloat16)`).
* **The VAE runs in bf16** upstream (`pipeline.to(dtype=bf16)`), and
  `1 / std` is computed in bf16.
* **Pixels are cast to bf16 before `*2 - 1`.**

## 7. Presets and camera trajectories

Presets are YAML upstream. Chosen here: **a development-time conversion to
JSON** (`tests/ref/presets_to_json.py`, PyYAML in `.venv-ref`), shipped in
`remind_lucid/presets/` with their inputs (7 images, 4 video prefixes, 4
trajectories) and the 12 trajectories of `prompt_engineering/`. The
alternative — a minimal YAML parser for the subset the presets use (block
mappings, `>-` folded scalars, numbers, booleans) — would be a second YAML
dialect to maintain and to trust; JSON loads with the standard library and is
exact. `tests/test_presets.py` compares every resolved schedule, camera
metric and count with upstream's `--dry-run`, including the analytic-camera
and default-event fallbacks.

Trajectories are `.npz` with `extrinsics (81, 3, 4)` and `intrinsics (81, 3,
3)` in float32 (plus `fps`, `resolution`, unused); read by `io/npy.py`.

## 8. Lucid operations: missing or questionable, and the composition used

| Need | Status in 3.16.4 | Composition |
|---|---|---|
| float64 on Metal | not available (MLX GPU) | tables and the x0 conversion on the CPU backend |
| complex tensors for RoPE | available, but not needed | real `cos`/`sin` tables with `F.apply_rotary_emb(interleaved=True)` |
| `einsum('bcij,bncpkj->bncpki')` | — | broadcast `matmul` with the transposed 4×4 matrix |
| in-place slice writes into a preallocated KV buffer | possible, not used | one contiguous entry per chunk, concatenated on read |
| `nearest-exact` upsampling | `nearest` only | identical for an integer scale factor of 2 |
| `repeat_interleave`, `F.normalize`, `F.pad` on 5-D | available | used as they are |
| bf16 `conv2d`/`conv3d` accuracy on Metal | questionable (cmd-lucid finding) | measured in §README "Lucid findings"; the VAE dtype is chosen from that measurement |
| `randn` matching torch | different generator by design | noise injection for parity; documented |
| YAML | no parser in the stdlib | JSON presets (§7) |
| `ftfy.fix_text` | not available | the non-mojibake subset re-implemented; verified on the prompt set |

## 9. Lucid components: as-is, subclassed, or new

Checked against the installed `lucid-dl==3.16.4`.

| Lucid component | Use | Detail |
|---|---|---|
| `nn.Linear`, `nn.Embedding`, `nn.Conv2d`, `nn.SiLU`, `nn.GELU(approximate="tanh")`, `nn.Dropout`, `nn.Identity`, `nn.Upsample`, `nn.Sequential`, `nn.ModuleList`, `nn.Parameter` | as-is | |
| `F.scaled_dot_product_attention` | as-is | DiT self/cross attention and the VAE attention block (fused MLX kernel) |
| `F.apply_rotary_emb(interleaved=True)` | as-is | fp32 rotation of queries and keys with per-block camera-modulated tables |
| `F.normalize`, `F.pad`, `F.softmax` | as-is | |
| `lucid.load_safetensors` / `save_safetensors` | as-is | bf16 and fp32, loaded straight onto the device |
| `lucid.utils.tokenizer.UnigramTokenizerFast` | **subclassed** → `UMT5Tokenizer` | adds `load`, `encode_clean`, `batch` (cleaning, truncation to 511 + `</s>`, padding); Viterbi and the `tokenizer.json` pipeline unchanged |
| `nn.Conv3d` | **subclassed** → `CausalConv3d` | `__init__` (no temporal padding), `stream`/`_stream` (history as causal padding); `forward` unchanged |
| `nn.RMSNorm` | **subclassed** → `WanRMSNorm`, `T5LayerNorm` | `forward` (normalise in fp32, cast, scale — the reference's rounding points) |
| `nn.LayerNorm` | **subclassed** → `WanLayerNorm` | `forward` (result cast to the input dtype) |
| `nn.Linear` | **subclassed** → `LoRALinear` | adds `merge_lora`; `forward` unchanged |
| `nn.Module` → VAE `RMSNorm` | new | `nn.RMSNorm` normalises the last axis by the root *mean* square with eps; Wan's `RMS_norm` L2-normalises axis 1 and scales by `sqrt(C)·gamma` |
| `nn.Module` → `T5Attention`, `CausalSelfAttention`, `CrossAttention` | new | `nn.MultiheadAttention` packs q/k/v in one weight, always scales, has no per-head additive bias, no q/k norms and no hook for cached rotated keys or projective residuals |
| `nn.Module` → `AvgDown3D`, `DupUp3D` | new | parameter-free; `nn.PixelUnshuffle`/`PixelShuffle` are 2-D and have no group mean / repeat |
| `nn.Module` → `Resample`, residual blocks, encoder/decoder, `CausalWanModel`, `UMT5Encoder` | new | containers that mirror the checkpoints' parameter names |
| `lucid.compile` | as-is | `accel.py` |
| `lucid.metal.{memory_allocated, max_memory_allocated, reset_peak_memory_stats, empty_cache}` | as-is | stage logging and staged residency |

Nothing in Lucid is patched, copied or reached through a private name;
`tests/test_public_api.py` fails on any `lucid._…` import or attribute and on
`lucid.models`.

## 10. Parity plan

1. **Toy-size architecture checks** (`tests/ref/tiny_modules.py`): upstream
   modules with random fp32 weights at small widths, compared on CPU and
   Metal at ≤ 1e-4 relative. No released weights needed.
2. **Component checks with released weights**: tokenizer ids against
   upstream's `HuggingfaceTokenizer`; the VAE in fp32 and bf16; the
   flow-matching table bit for bit.
3. **Upstream itself, end to end** (`tests/ref/remind_upstream.py`): preset
   01 on the CPU in bf16 with the initial noise and every re-noising draw
   recorded and injected into the Lucid run: UMT5 features, first-frame
   latent, all 30 block outputs of chunk 0 / step 0, K/V after chunk 0,
   chunk latents, decoded frames, merged LoRA weights.

Targets: fp32 component checks ≤ ~1e-4 relative; bf16 end to end per-block
cosine ≥ 0.999, final latent cosine ≥ 0.995, decoded PSNR ≥ 35 dB. Measured
results are in the README.
