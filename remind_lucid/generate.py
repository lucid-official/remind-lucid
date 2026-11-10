"""Generate a ReMind video from a preset or from custom inputs.

    python -m remind_lucid.generate --preset 01_latte_occluder_recovery
    python -m remind_lucid.generate --preset 03 --seed 7 --output outputs/whisk.mp4
    python -m remind_lucid.generate --preset 01 --dry-run

    # custom inputs (the settings not given come from --preset, default 05: clean I2V)
    python -m remind_lucid.generate --image my.png --prompt "A kettle boils ..." --camera 05_yaw_right_21deg
    python -m remind_lucid.generate --video prefix.mp4 --prompts-file seven_prompts.txt

The defaults reproduce the reference recipe: 81 frames at 832x480 and 16 fps,
seven 3-latent-frame chunks, the four-step DMD schedule ``[1000, 938, 833,
625]`` and the DMD EMA LoRA merged after the bf16 cast. Frames are decoded
and written while later chunks are still generating. Next to the video goes a
``.json`` with the reference's metadata fields plus timings and memory.

A seed does not reproduce an upstream video: upstream draws its noise from
torch's CUDA generator, this port from Lucid's.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lucid
from tqdm import tqdm

from remind_lucid import catalog, presets
from remind_lucid.io.media import VideoWriter, to_uint8_frames
from remind_lucid.pipeline import DTYPES, Job, NoiseSource, ReMindPipeline, peak_memory_gb


def build_preset(args) -> tuple[Path, dict]:
    """The preset named on the command line with the custom inputs laid over it."""
    custom = any(v is not None for v in (args.image, args.video, args.prompt, args.prompts_file, args.camera))  # fmt: skip
    path = presets.resolve_preset_path(args.preset or "05")
    data = json.loads(path.read_text(encoding="utf-8"))
    base_dir = path.parent
    if custom:
        data = {k: v for k, v in data.items() if k not in ("website_case",)}
        data["name"] = args.name or "custom"
        if args.preset is None:
            data["control"] = {"kind": "clean"}
        if args.image:
            data.update(task="i2v", input_image=str(Path(args.image).resolve()))
            data.pop("input_video", None)
            data.pop("seed_latent_frames", None)
        if args.video:
            data.update(task="v2v", input_video=str(Path(args.video).resolve()))
            data.pop("input_image", None)
            data.setdefault("seed_latent_frames", 6)
        if args.prompt:
            data["prompt"] = args.prompt
        control = dict(data.get("control") or {"kind": "clean"})
        if args.camera:
            control = {"kind": "camera", "trajectory": args.camera}
        if args.prompts_file:
            lines = [ln.strip() for ln in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if ln.strip()]  # fmt: skip
            control["prompts"] = lines
            data["prompt"] = data.get("prompt") or lines[0]
        data["control"] = control
    return path, presets.validate(data, base_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--preset", help=f"bundled preset (name or number prefix) or a preset .json; bundled: {', '.join(presets.preset_names())}")  # fmt: skip
    p.add_argument("--image", help="custom I2V: first image (resized to 832x480 with Lanczos)")
    p.add_argument("--video", help="custom V2V: a video whose first 21 frames seed 6 latent frames")  # fmt: skip
    p.add_argument("--prompt", help="content-only temporal caption (see the reference's docs/inference.md)")  # fmt: skip
    p.add_argument("--prompts-file", help="7 chunk-local prompts, one per line; bypasses the control schedule")  # fmt: skip
    p.add_argument("--camera", help=f"bundled trajectory or an .npz path; bundled: {', '.join(presets.trajectory_names())}")  # fmt: skip
    p.add_argument("--name", help="name of a custom run")
    p.add_argument("--seed", type=int)
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16", help="generator / text-encoder dtype (reference: bf16)")  # fmt: skip
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False, help="use the lucid.compile programs")  # fmt: skip
    p.add_argument("--output", type=Path)
    p.add_argument("--dry-run", action="store_true", help="validate and print the resolved schedule without loading weights")  # fmt: skip
    args = p.parse_args()

    preset_path, preset = build_preset(args)
    if args.dry_run:
        print(json.dumps(presets.dry_run_summary(preset_path, preset), indent=2, ensure_ascii=False))  # fmt: skip
        return
    if not catalog.ready():
        raise SystemExit("weights are missing; run `python -m remind_lucid.download`")

    job = Job.from_preset(preset_path, preset, args.seed)
    out = (args.output or Path("outputs") / f"{preset['name']}.mp4").expanduser().resolve()
    t_start = time.time()
    lucid.metal.reset_peak_memory_stats()
    pipe = ReMindPipeline(dtype=DTYPES[args.dtype], accelerate=args.compile, log=lambda m: tqdm.write(m))  # fmt: skip
    features = pipe.encode_prompts(job.prompts)
    source = pipe.encode_source(job)
    pipe.ensure_generator()
    decode = pipe.decoder()
    writer = VideoWriter(out, job.width, job.height, fps=job.fps)
    bar = tqdm(total=job.frames, unit="frame", desc=f"{preset['name']} -> {out.name}", dynamic_ncols=True)  # fmt: skip
    written, decode_seconds = 0, 0.0
    for chunk in pipe.rollout(job, features, source, NoiseSource(job.seed, pipe.dtype)):
        t0 = time.time()
        pixels = decode(chunk.latents)
        writer.write(to_uint8_frames(pixels))
        decode_seconds += time.time() - t0
        written += pixels.shape[1]
        bar.update(pixels.shape[1])
        bar.set_postfix_str(f"chunk {chunk.index + 1}/{job.num_chunks}, {chunk.seconds:.1f}s, kv {chunk.kv_tokens}")  # fmt: skip
    bar.close()
    writer.close()

    metadata = job.metadata()
    metadata.update(
        output=str(out),
        lucid_version=lucid.__version__,
        dtype=args.dtype,
        compile=bool(args.compile),
        timings={
            **pipe.timings.stages,
            "chunks": pipe.timings.chunks,
            "vae_decode": round(decode_seconds, 3),
            "total": round(time.time() - t_start, 3),
        },
        memory_gb={"peak": round(peak_memory_gb(), 3), **pipe.timings.memory},
    )
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # fmt: skip
    print(json.dumps({k: metadata[k] for k in ("name", "output", "seed", "timings", "memory_gb")}, indent=2))  # fmt: skip


if __name__ == "__main__":
    main()
