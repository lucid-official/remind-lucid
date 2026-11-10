"""Run probe sweeps from the command line, keeping the generator resident across runs.

    python -m remind_lucid.probes.batch --preset 01 --sweep interruption --values -1,0,1,2
    python -m remind_lucid.probes.batch --preset 01 --sweep seed --values 1,2,3,4
    python -m remind_lucid.probes.batch --preset 01 --sweep gap --values 0,8,32,128 --from-chunk 5
    python -m remind_lucid.probes.batch --preset 01 --sweep knockout --sets "-;0;3;0,1,2" --from-chunk 5 --attention 0,7,14,21,29
    python -m remind_lucid.probes.batch --all-presets          # the 11 presets, one after another

Each run writes ``<out>/<name>.mp4`` and ``.json`` (metadata, timings, probe
report); a sweep also gets ``<out>/strip.jpg``, one row per run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lucid

from remind_lucid import catalog, presets
from remind_lucid.io.media import VideoWriter, make_strip, to_uint8_frames
from remind_lucid.pipeline import (
    DTYPES,
    Job,
    NoiseSource,
    ReMindPipeline,
    peak_memory_gb,
)
from remind_lucid.probes import ProbeSet, sweeps
from remind_lucid.probes.heatmap import render as render_heatmap

def run_one(pipe: ReMindPipeline, preset: dict, probes: dict | None, out_dir: Path, seed: int | None = None, log=print) -> dict:  # fmt: skip
    """One generation with the resident pipeline. Returns its metadata (also written next to the video)."""
    job = Job.from_preset("batch", preset, seed)
    video = out_dir / f"{preset['name']}.mp4"
    t_start = time.time()
    lucid.metal.reset_peak_memory_stats()
    pipe.timings = type(pipe.timings)()
    if not pipe.prompt_encoder.cached(job.prompts):
        pipe.release_generator()  # room for UMT5
    features = pipe.encode_prompts(job.prompts)
    source = pipe.encode_source(job)
    pipe.ensure_generator()
    pipe.probes = ProbeSet.from_dict(probes)
    decode = pipe.decoder()
    writer = VideoWriter(video, job.width, job.height, fps=job.fps)
    decode_seconds = 0.0
    try:
        for chunk in pipe.rollout(
            job, features, source, NoiseSource(job.seed, pipe.dtype)
        ):
            t0 = time.time()
            writer.write(to_uint8_frames(decode(chunk.latents)))
            decode_seconds += time.time() - t0
            log(f"  {preset['name']}: chunk {chunk.index + 1}/{job.num_chunks} in {chunk.seconds:.1f}s (history {chunk.kv_tokens} tokens)")  # fmt: skip
    finally:
        writer.close()
        probe, pipe.probes = pipe.probes, None
    metadata = job.metadata()
    metadata.update(
        output=str(video),
        sweep=preset.get("sweep"),
        probes=probe.to_dict() if probe is not None else None,
        lucid_version=lucid.__version__,
        compile=pipe.accelerate,
        timings={**pipe.timings.stages, "chunks": pipe.timings.chunks, "vae_decode": round(decode_seconds, 3), "total": round(time.time() - t_start, 3)},  # fmt: skip
        memory_gb={"peak": round(peak_memory_gb(), 3), **pipe.timings.memory},
    )
    if probe is not None and probe.mass:
        report = probe.report()
        metadata["attention"] = report
        render_heatmap(report, max(int(c) for c in report["attention_mass"]), video.with_suffix(".heatmap.png"))  # fmt: skip
    video.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")  # fmt: skip
    return metadata


def _ints(text: str) -> list[int]:
    return [int(v) for v in text.replace(" ", "").split(",") if v != ""]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--preset", default="01")
    p.add_argument("--all-presets", action="store_true")
    p.add_argument("--sweep", choices=("interruption", "seed", "gap", "knockout", "none"), default="none")  # fmt: skip
    p.add_argument("--values", default="", help="hidden chunks / seeds / gap frames, comma separated")  # fmt: skip
    p.add_argument("--sets", default="-;0;1;2", help="knockout: chunk sets separated by ';' ('-' = none)")  # fmt: skip
    p.add_argument("--from-chunk", type=int, default=5, help="gap / knockout: the first chunk affected")  # fmt: skip
    p.add_argument("--start-chunk", type=int, help="interruption: the chunk in which the scene starts to disappear")  # fmt: skip
    p.add_argument("--attention", default="", help="blocks whose attention mass is logged, e.g. 0,7,14,21,29")  # fmt: skip
    p.add_argument("--seed", type=int)
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--out", type=Path, default=Path("outputs") / "probes")
    args = p.parse_args()
    if not catalog.ready():
        raise SystemExit("weights are missing; run `python -m remind_lucid.download`")

    hooks: dict = {}
    if args.attention:
        hooks["attention_layers"] = _ints(args.attention)
    runs: list[tuple[dict, dict | None]] = []
    if args.all_presets:
        runs = [(presets.load_preset(name)[1], hooks or None) for name in presets.preset_names()]  # fmt: skip
    else:
        _, base = presets.load_preset(args.preset)
        if args.sweep == "interruption":
            variants = sweeps.interruption_sweep(base, _ints(args.values or "-1,0,1,2"), start_chunk=args.start_chunk)  # fmt: skip
            runs = [(presets.validate(v, presets.PRESET_DIR), hooks or None) for v in variants]  # fmt: skip
        elif args.sweep == "seed":
            variants = sweeps.seed_sweep(base, _ints(args.values or "1,2,3,4"))
            runs = [(presets.validate(v, presets.PRESET_DIR), hooks or None) for v in variants]  # fmt: skip
        elif args.sweep == "gap":
            for config in sweeps.gap_sweep(_ints(args.values or "0,8,32,128"), args.from_chunk):  # fmt: skip
                variant = {k: v for k, v in base.items() if not k.startswith("_")}
                variant["name"] = f"{base['name']}_gap{config['gap_frames']}"
                variant["sweep"] = {"kind": "gap", **config}
                runs.append((presets.validate(variant, presets.PRESET_DIR), {**hooks, **config}))  # fmt: skip
        elif args.sweep == "knockout":
            for text in args.sets.split(";"):
                drop = [] if text.strip() == "-" else _ints(text)
                variant = {k: v for k, v in base.items() if not k.startswith("_")}
                variant["name"] = (
                    f"{base['name']}_drop{'-'.join(map(str, drop)) or 'none'}"
                )
                variant["sweep"] = {"kind": "knockout", "drop": drop, "at_chunk": args.from_chunk}  # fmt: skip
                config = dict(hooks)
                if drop:
                    config["knockout"] = {str(args.from_chunk): drop}
                runs.append((presets.validate(variant, presets.PRESET_DIR), config or None))  # fmt: skip
        else:
            runs = [(base, hooks or None)]

    args.out.mkdir(parents=True, exist_ok=True)
    pipe = ReMindPipeline(dtype=DTYPES[args.dtype], accelerate=args.compile, log=print)
    results = []
    for preset, probes in runs:
        print(f"[{preset['name']}]", flush=True)
        results.append(run_one(pipe, preset, probes, args.out, args.seed))
        print(f"  total {results[-1]['timings']['total']}s, peak {results[-1]['memory_gb']['peak']} GB", flush=True)  # fmt: skip
    if len(results) > 1 and not args.all_presets:
        make_strip([r["output"] for r in results], args.out / "strip.jpg", [0, 16, 28, 36, 44, 52, 64, 80])  # fmt: skip
    (args.out / "summary.json").write_text(json.dumps([{k: r[k] for k in ("name", "output", "seed", "sweep", "timings", "memory_gb")} for r in results], indent=2, default=str) + "\n")  # fmt: skip


if __name__ == "__main__":
    main()
