<h1 align="center">ReMind on Lucid</h1>

<p align="center">
  <b>Video generation that remembers what it cannot see, natively on Apple Silicon.</b><br>
  A torch-free re-implementation of
  <a href="https://arxiv.org/abs/2605.25333">ReMind: Teaching Video Generators to Remember</a>
  on <a href="https://github.com/lucid-official/lucid">Lucid</a>. The released ReMind-5B checkpoint runs unchanged
  on Metal and streams an 81-frame video three latent frames at a time.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.25333"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2605.25333-b31b1b.svg"></a>
  <a href="https://github.com/Applied-Intuition-Open-Source/ReMind"><img alt="Upstream" src="https://img.shields.io/badge/upstream-Applied--Intuition%2FReMind-black.svg"></a>
  <a href="https://huggingface.co/AppliedIntuitionResearch/ReMind"><img alt="Weights" src="https://img.shields.io/badge/%F0%9F%A4%97%20weights-ReMind--5B-orange.svg"></a>
  <a href="https://pypi.org/project/lucid-dl/3.16.4/"><img alt="Lucid" src="https://img.shields.io/badge/Lucid-3.16.4-4c8bf5.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.14-blue.svg">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-green.svg"></a>
</p>

<p align="center">
  <img src="assets/latte-occluder.gif" width="60%">
</p>

<p align="center">
  <img src="assets/latte-occluder-strip.jpg" width="100%">
</p>

<p align="center">
  <sub>Upstream's preset <code>01_latte_occluder_recovery</code>, run here: one input image, 81 frames at 832×480.
  A cardboard box covers the whole view for 0.75 s. When it leaves, the foam has risen: the pour went on while
  nothing of it was visible. Generated in 177 s on an M4 Max. The seed does not reproduce upstream's video, whose
  noise comes from CUDA's generator.</sub>
</p>

---

## Highlights

- **The authors' weights, unchanged.** All 975 tensors of `ReMind-5B.safetensors` load with `strict=True`, and the
  rank-128 DMD EMA LoRA is merged into its 360 linear layers the way `peft` does on the CPU. The 20 GB of Wan2.2
  base shards upstream loads first are never downloaded: the ReMind file overwrites every one of them.
- **All eleven upstream presets.** Seven image-to-video and four video-to-video cases, with their inputs, camera
  trajectories and seeds. Each resolves to exactly the prompt and camera schedule upstream's `--dry-run` prints.
- **Streaming, causal generation.** Each chunk of three latent frames is denoised in four steps against a KV cache
  of every earlier chunk, then decoded by a streaming VAE while the next one starts.
- **Research probes for the memory itself.** Shift the temporal positions to open a gap between the history and
  the present, hide or replace individual cached chunks, and log how much attention each layer puts on each chunk.
  All of them are off by default, and off is bit-identical to the reference path.
- **Written as a Lucid user would write it.** Models and pipeline use only Lucid's public API and the standard
  library. There is no torch and no numpy, and [`tests/test_public_api.py`](tests/test_public_api.py) enforces it.

## What this is

A causal video model generates frame by frame and keeps earlier frames as context. That context is usually a
static memory: when the camera turns away from a pot of boiling water and back, the water is as it was. ReMind
(Xu et al., 2026) trains the model so that state hidden by an occluder or by camera motion keeps evolving, and
reappears changed.

The released ReMind-5B model is a Wan2.2-TI2V-5B DiT with a camera-conditioned positional encoding in every
block (a pose-dependent phase on the RoPE rotation, plus projective transforms of keys and values), distilled
with DMD into a four-step causal generator.

This repository contains:

- **The inference stack:**
  - the causal Wan2.2 5B DiT with camera-modulated RoPE and a KV cache that is never evicted;
  - the Wan2.2 VAE (16× spatial, 4× temporal, 48 latent channels), with streaming encode and decode;
  - the UMT5-XXL text encoder and its tokenizer;
  - the flow-matching schedule and the four-step DMD rollout `[1000, 938, 833, 625]`;
  - upstream's control schedules, which describe an occluder or a camera move to the model as seven chunk-local
    prompts;
  - `ReMindPipeline.rollout` as a generator that yields each finished chunk.
- **The released weights,** fetched from Hugging Face by HTTP range requests and converted to SafeTensors by a
  torch-free reader that refuses any pickle global a state dict does not need.
- **Probes and sweeps** for studying the dynamic memory ([Probes](#probes)).
- **A web demo** with live frames, a gallery with side-by-side comparison, and the probes.

## Results

Five of the eleven presets, each from upstream's input and seed. Every video is 81 frames (5 s at 16 fps).

<table>
  <tr>
    <td align="center" width="50%"><img src="assets/whisking-camera-pan.gif" width="100%"></td>
    <td align="center" width="50%"><img src="assets/pour-camera-pan-v2v.gif" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><sub><b>03 · camera pan, I2V.</b> The camera turns away from the bowl and comes back to
    batter that has kept being whisked.</sub></td>
    <td align="center"><sub><b>10 · camera pan, V2V.</b> Seeded with 21 real frames. The glass is fuller when the
    camera returns.</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/drink-occluder-v2v.gif" width="100%"></td>
    <td align="center"><img src="assets/dry-ice-clean.gif" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><sub><b>09 · occluder, V2V.</b> A flaming drink foams over behind the box.</sub></td>
    <td align="center"><sub><b>06 · no control, I2V.</b> Dry-ice fog with nothing hidden.</sub></td>
  </tr>
</table>

<p align="center">
  <img src="assets/whisking-camera-pan-strip.jpg" width="100%">
</p>

<p align="center">
  <sub>Preset 03 at 0, 1, 2, 3, 4 and 5 s. The bowl is out of the frame in the middle.</sub>
</p>

These are single runs at upstream's seeds, shown as they came out. No comparison against upstream's own videos
was made (see [Limitations](#limitations)).

## Speed and memory on an M4 Max

M4 Max, 36 GB. DiT and text encoder in bf16, as the reference runs them; VAE in fp32 (see
[Lucid findings](#lucid-findings)). Ranges are over the five runs above.

| stage | time |
|---|---|
| prompt → features (UMT5-XXL, 11.4 GB) | 20–24 s including the load; cached on disk afterwards |
| load the generator and merge the LoRA | 23–51 s |
| DiT, one chunk of 3 latent frames (4 denoising passes + 1 KV commit) | 7 s for the first chunk, 8–24 s with history |
| VAE decode, 81 frames | 75–114 s; 2.0 s per latent frame when the VAE runs alone |
| **one 5 s video, end to end** | **177–295 s** |

That is roughly 35–60× slower than real time. Peak Metal memory is **25.2 GB**: 11.1 GB of generator weights,
up to 3 GB of cached keys and values, and the VAE with its streaming histories. UMT5 is released before the
generator is loaded, so the two never coexist.

These numbers are pessimistic. On 36 GB the run sits close to the limit, and the machine had 10–14 GB in swap
while they were measured; the VAE decodes at about half the speed it reaches alone.

**Lucid vs PyTorch (MPS) at this model's shapes:**

| op | Lucid 3.16.4 | torch 2.15.0, MPS |
|---|---|---|
| linear 1170×3072→14336, bf16 | 9.5 ms | 9.2 ms |
| attention, q 1,170 × kv 1,170, bf16 | 1.9 ms | 1.8 ms |
| attention, q 1,170 × kv 8,190, bf16 | 11.6 ms | 11.8 ms |
| attention, q 1,170 × kv 8,190, fp32 | 17.9 ms | 17.4 ms |

The two are level on every op measured (`tests/bench/ops_lucid.py`, `tests/ref/ops_mps.py`).

## Numerical parity

`python -m pytest` runs 55 tests. Reference outputs come from `tests/ref/*.py`, run in a separate torch
environment against the upstream repository's own modules.

| check | against | result |
|---|---|---|
| causal DiT, toy width: every block, flow, cached K/V, history and gap chunks | upstream `CausalWanModel`, fp32, CPU and Metal | < 1e-4 rel. |
| UMT5 encoder, toy width | upstream `T5Encoder` | < 1e-4 rel. |
| Wan2.2 VAE encode + decode, toy width | upstream `WanVAE_` | < 1e-4 rel. |
| VAE with the released weights, 832×480, fp32 | upstream, fp32 | latents < 1.5e-5 rel., decode 84.8 dB PSNR |
| causality of the VAE encoder (encoding only the prefix frames) | the 81-frame encode | < 1e-5 rel. |
| input decoding (ffmpeg here, PIL and imageio upstream) | upstream's 8-bit frames | identical pixels |
| tokenizer, cleaning and truncation included | upstream `HuggingfaceTokenizer` | 53 / 53 prompts, every id |
| flow-matching table, `add_noise`, x0 from flow | upstream `FlowMatchScheduler` | bit-exact |
| prompt and camera schedules of the 11 presets, plus custom controls | upstream `inference.py --dry-run` | equal; camera metrics to 1e-4 |
| probes switched off | the plain rollout | bit-identical |

**Not run yet: the end-to-end comparison on the released weights.** [`tests/test_parity.py`](tests/test_parity.py)
compares UMT5 features, the first-frame latent, all 30 block outputs, the KV cache, chunk latents, decoded frames
and the merged LoRA weights with upstream itself, with upstream's noise injected. It needs a fixture from
`tests/ref/remind_upstream.py`, which runs the 5B model in bf16 on the CPU. That fixture has not been produced,
so these seven tests are skipped and the full generator is so far verified only at toy width and by its output.

[`docs/PORTING_PLAN.md`](docs/PORTING_PLAN.md) lists every constant, dtype, random draw and cache mutation of the
reference path and where each one lives here.

## Lucid findings

Found while building this, each with a measured repro:

- **The Metal buffer cache has to be emptied by hand in a long pipeline.** After the VAE's 480×832 stages, 9 GB
  stays in the cache beside 20 GB in use. On a 36 GB machine that goes to swap, and the same rollout takes 441 s.
  Calling `lucid.metal.empty_cache()` after each chunk and each decoded latent frame brings it to 190 s.
- **`lucid.compile` keeps memory for every input signature and does not give it back.** One DiT block (0.37 GB
  of weights) retains 0.82–0.86 GB per history length, still in use after `clear_cache()`
  (`tests/bench/compile_memory.py`). The compiled streaming VAE decoder passes 64 GB
  (`tests/bench/compile_vae.py`); split into one program per decoder layer with `lucid.eval` between them, it
  still passes 42 GB, growing with every program called.
- **Compiling does not make a DiT block faster.** 42.9 ms eager against 44.2 ms compiled with no history, 53.7
  against 63.3 ms with 7,020 cached tokens (`tests/bench/compile_dit.py`). The pipeline is eager by default;
  `--compile` compiles two blocks so that the path stays tested.
- **The VAE runs in fp32, not the reference's bf16.** In bf16 on Metal the latents differ from upstream's fp32
  latents by up to 7% (cosine 0.9998) and decoding reaches 58 dB; in fp32 the difference is 1.5e-5 and 84.8 dB
  (`tests/test_vae_weights.py`).
- **A lazily built 480×832 residual block keeps every intermediate alive.** The VAE evaluates after each causal
  convolution, and the high-resolution decoder stages take their four frames one at a time.

## Getting started

### Install

An Apple Silicon Mac with 36 GB or more, Python 3.14, and `ffmpeg` (`brew install ffmpeg`) for image and video I/O.

```bash
git clone https://github.com/lucid-official/remind-lucid && cd remind-lucid
uv venv --python 3.14 .venv && source .venv/bin/activate
uv pip install -e ".[test]"
python -m pytest -q
```

### Download the weights

```bash
python -m remind_lucid.download            # 26.8 GB
python -m remind_lucid.download --status   # per-file status only
```

| file | size | |
|---|---|---|
| `ReMind-5B.safetensors` | 11.1 GB | as released (bf16), SHA-256 verified |
| `ReMind-5b-dmd-ema.safetensors` | 1.5 GB | as released (fp32), SHA-256 verified |
| UMT5-XXL encoder | 11.4 GB | `.pth` read remotely, written as SafeTensors |
| Wan2.2 VAE | 2.8 GB | `.pth` read remotely, written as SafeTensors |

### Generate

```bash
# an upstream preset, by name or number
python -m remind_lucid.generate --preset 01_latte_occluder_recovery
python -m remind_lucid.generate --preset 03 --seed 7 --output outputs/whisk.mp4

# the resolved prompt and camera schedule, without loading weights
python -m remind_lucid.generate --preset 01 --dry-run

# custom inputs: settings not given come from --preset (default 05, no control)
python -m remind_lucid.generate --image my.png --prompt "A kettle boils ..." --camera 05_yaw_right_21deg
python -m remind_lucid.generate --video prefix.mp4 --prompts-file seven_prompts.txt
```

Frames are decoded and written while later chunks are still generating. Next to the video goes a `.json` with
the reference's metadata fields, timings and memory. Prompts follow upstream's content-only caption contract;
its `docs/inference.md` explains it.

| preset | task | control |
|---|---|---|
| `01_latte_occluder_recovery` | I2V | latte art evolves behind a cardboard box |
| `02_pancake_occluder_recovery` | I2V | pancake batter evolves behind a cardboard box |
| `03_whisking_camera_pan` | I2V | whisking under a camera pan |
| `04_cereal_camera_pan` | I2V | cereal pouring under a camera pan |
| `05_dog_clean_i2v` | I2V | none |
| `06_dry_ice_clean_i2v` | I2V | none |
| `07_flour_clean_i2v` | I2V | none |
| `08_cake_decorating_occluder_v2v` | V2V | syrup decoration behind a cardboard box |
| `09_ice_melting_drink_occluder_v2v` | V2V | a flaming, foaming drink behind a cardboard box |
| `10_pour_wine_beer_camera_pan_v2v` | V2V | a drink fills under a camera pan |
| `11_slime_stretch_camera_pan_v2v` | V2V | slime accumulates under a camera pan |

### Web demo

```bash
python -m remind_lucid.server        # binds the Tailscale IP if there is one, else 127.0.0.1; port 7860
```

Pick a preset or drop in an image or a video prefix, press **Generate**, and watch the frames arrive. Every run
is kept under `outputs/web/`, so the gallery, with side-by-side comparison, survives restarts.

## Probes

`remind_lucid/probes/` asks what the model's memory is made of. The batch runner keeps the generator resident
across runs and writes a video, a `.json` report and a frame strip per sweep.

```bash
# how long can the scene stay hidden? 0, 1, 2 fully hidden chunks (0.75 s each), and no interruption at all
python -m remind_lucid.probes.batch --preset 01 --sweep interruption --values -1,0,1,2

# open a gap of G latent frames in the temporal positions from chunk 5 on: memories look G frames older
python -m remind_lucid.probes.batch --preset 01 --sweep gap --values 0,8,32,128 --from-chunk 5

# hide cached chunks from later ones, and log the attention mass each layer puts on each chunk
python -m remind_lucid.probes.batch --preset 01 --sweep knockout --sets "-;0;3;0,1,2" --from-chunk 5 \
    --attention 0,7,14,21,29

python -m remind_lucid.probes.batch --preset 01 --sweep seed --values 1,2,3,4
python -m remind_lucid.probes.batch --all-presets
```

No probe results are reported here yet; the hooks are tested for correctness at toy size
([`tests/test_probes.py`](tests/test_probes.py)).

## Repository layout

```
remind_lucid/
  models/wan_dit.py      causal Wan2.2 5B DiT, camera-modulated RoPE, projective K/V, KV cache, LoRA merge
  models/wan22_vae.py    Wan2.2 VAE, streaming encode and decode
  models/umt5.py         UMT5-XXL encoder
  text/                  tokenizer (id-exact), cached prompt encoder, occluder / camera prompt schedules
  scheduler.py           flow-matching table (bit-exact), x0 from flow in float64
  camera.py              trajectories, pixel → latent cameras, per-chunk pose normalisation
  presets.py, presets/   the 11 upstream presets as JSON, their inputs, 12 camera trajectories
  pipeline.py            ReMindPipeline: staged loading, the DMD rollout as a chunk generator, streaming decoder
  generate.py            CLI
  probes/                RoPE gap, anchor knockout, attention mass, sweeps, batch runner
  server/                web demo (stdlib HTTP + Server-Sent Events)
  hub.py, download.py    range-request downloads into SafeTensors
  io/                    torch.save reader (no unpickling of code), .npy / .npz reader, ffmpeg media I/O
  accel.py               lucid.compile programs (measured, off by default)
tests/                   parity and behaviour tests; tests/ref/ makes the reference outputs in a torch venv;
                         tests/bench/ holds the benchmarks quoted above
docs/PORTING_PLAN.md     the upstream path, constant by constant
assets/                  README media
```

## Limitations

- **Not yet checked end to end against upstream.** Components match upstream's modules at toy width, the VAE
  matches with the released weights, and the videos show the behaviour the paper describes. The full-size
  comparison is written but has not been run (see [Numerical parity](#numerical-parity)).
- **A seed does not reproduce an upstream video.** Upstream draws its noise from torch's CUDA generator, this
  port from Lucid's.
- **No benchmark numbers.** None of the paper's metrics were computed here.
- **Far from real time, and tight on 36 GB.** About 3–5 minutes for 5 s of video, at a 25 GB peak.
- **Five seconds only.** The released recipe is 81 frames in seven chunks and nothing is evicted from the KV
  cache; longer rollouts were not tried.
- **`lucid.compile` does not help.** No faster for the DiT, and unusable for the VAE decoder at this resolution.
- **Not ported:** upstream's Qwen3-VL prompt rewriter for custom inputs (an authoring tool, not part of
  inference). Its twelve camera trajectories are bundled. Upstream has not released training code or the 1.3B
  checkpoint.

## Citation

```bibtex
@article{xu2026teaching,
  title   = {Teaching Video Generators to Remember: Eliciting Dynamic Memory for Out-of-Sight State Evolution},
  author  = {Xu, Tianshuo and Xie, Yichen and Meng, Depu and Peng, Chensheng and Herau, Quentin and Jiang, Bo
             and Hu, Yihan and Zhan, Wei},
  journal = {arXiv preprint arXiv:2605.25333},
  year    = {2026}
}
```

## Acknowledgements

- [ReMind](https://github.com/Applied-Intuition-Open-Source/ReMind), for the method, the released code that
  settles what the paper leaves open, the presets and the
  [checkpoints](https://huggingface.co/AppliedIntuitionResearch/ReMind).
- [Wan2.2](https://github.com/Wan-Video/Wan2.2), for the backbone, the VAE and the text encoder.
- [PRoPE](https://www.liruilong.cn/prope/), for the projective camera positional encoding ReMind builds on.
- [Lucid](https://github.com/lucid-official/lucid), for the Apple Silicon runtime.

No model weights are redistributed here; they are downloaded by the user. The input images, video prefixes and
camera trajectories under `remind_lucid/presets/` come from the upstream repository. The videos in
[`assets/`](assets/) were generated by the released checkpoint with this code.

## License

The code ports the ReMind implementation and is distributed under
[CC BY-NC 4.0](LICENSE) (non-commercial use only). See [NOTICE](NOTICE) for the Apache-2.0 and MIT projects it
also derives from. The released weights carry the same license.
