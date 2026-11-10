"""Presets: the 11 released demo cases and custom ones, validated as ``inference.py`` does.

The reference reads YAML presets. There is no YAML parser in the standard
library and the runtime takes no third-party package, so the 11 presets are
converted once, at development time, by ``tests/ref/presets_to_json.py``
(PyYAML in ``.venv-ref``) and shipped as JSON next to their inputs in
``remind_lucid/presets/``. A hand-written parser for "the subset of YAML the
presets use" would have been a second, untested YAML dialect; JSON is exact
and a test compares every resolved schedule with upstream's ``--dry-run``.

A custom preset is the same JSON (a file, or a dict from the web demo);
relative paths resolve against the preset file's folder.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from remind_lucid import camera, catalog
from remind_lucid.text.schedule import chunk_text_prompts

PRESET_DIR = Path(__file__).resolve().parent / "presets"
TRAJECTORY_DIR = PRESET_DIR / "trajectories"
CONTROL_KINDS = {"clean", "camera", "occluder", "light"}
TASKS = {"i2v", "v2v"}


def preset_names() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


def trajectory_names() -> list[str]:
    return sorted(p.stem for p in TRAJECTORY_DIR.glob("*.npz"))


def trajectory_index() -> list[dict]:
    path = TRAJECTORY_DIR / "index.json"
    return json.loads(path.read_text(encoding="utf-8"))["trajectories"] if path.exists() else []  # fmt: skip


def resolve_preset_path(name_or_path: str | Path) -> Path:
    """A bundled preset by name (``01_latte_occluder_recovery`` or a unique prefix like ``01``), or a JSON path."""
    path = Path(name_or_path).expanduser()
    if path.suffix == ".json" and path.is_file():
        return path.resolve()
    matches = [p for p in sorted(PRESET_DIR.glob("*.json")) if p.stem == str(name_or_path) or p.stem.startswith(f"{name_or_path}_")]  # fmt: skip
    if len(matches) != 1:
        raise FileNotFoundError(
            f"no preset {name_or_path!r}; bundled: {preset_names()}"
        )
    return matches[0]


def validate(data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """``load_preset`` after parsing: defaults, checks and resolved input paths (keys starting with ``_``)."""
    data = dict(data)
    missing = [key for key in ("name", "prompt") if not data.get(key)]
    if missing:
        raise ValueError(f"preset is missing required keys {missing}")
    task = str(data.get("task", "i2v")).lower()
    if task not in TASKS:
        raise ValueError(f"unsupported task {task!r}; choose {sorted(TASKS)}")
    data["task"] = task
    input_key = "input_image" if task == "i2v" else "input_video"
    if not data.get(input_key):
        raise ValueError(f"{task} preset is missing required key {input_key!r}")
    control = dict(data.get("control") or {"kind": "clean"})
    kind = str(control.get("kind", "clean")).lower()
    if kind not in CONTROL_KINDS:
        raise ValueError(f"unsupported control kind {kind!r}; choose {sorted(CONTROL_KINDS)}")  # fmt: skip
    control["kind"] = kind
    data["control"] = control
    input_path = (base_dir / str(data[input_key])).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    data[f"_{input_key}_path"] = input_path

    chunk_size = int(data.get("chunk_size", 3))
    latent_frames = (int(data.get("num_frames", 81)) - 1) // 4 + 1
    latent_frames -= latent_frames % chunk_size
    default_seed = 1 if task == "i2v" else 2 * chunk_size
    seed_latent_frames = int(data.get("seed_latent_frames", default_seed))
    if not 0 < seed_latent_frames < latent_frames:
        raise ValueError(f"seed_latent_frames must be in [1, {latent_frames - 1}], got {seed_latent_frames}")  # fmt: skip
    if task == "i2v" and seed_latent_frames != 1:
        raise ValueError("i2v presets must use exactly one seed latent frame")
    if task == "v2v" and seed_latent_frames % chunk_size:
        raise ValueError("v2v seed_latent_frames must be a whole number of chunks")
    data["seed_latent_frames"] = seed_latent_frames
    trajectory = control.get("trajectory")
    if trajectory:
        candidates = [base_dir / str(trajectory), TRAJECTORY_DIR / f"{trajectory}.npz", PRESET_DIR / "assets" / f"{trajectory}.npz"]  # fmt: skip
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            raise FileNotFoundError(candidates[0].resolve())
        control["_trajectory_path"] = found.resolve()
    return data


def load_preset(name_or_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = resolve_preset_path(name_or_path)
    data = json.loads(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"preset must be a mapping: {path}")
    return path, validate(data, path.parent)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def _event_span(control: dict, num_chunks: int) -> tuple[int, int]:
    start = int(control.get("start_chunk", 2))
    end = int(control.get("end_chunk", min(4, num_chunks - 2)))
    if start < 1 or end < start or end >= num_chunks - 1:
        raise ValueError(f"event chunks must leave a clean seed and recovery tail; got [{start}, {end}] for {num_chunks} chunks")  # fmt: skip
    return start, end


def _loop_caption(base: str, loop: dict, total_frames: int, motion: str = "") -> str:
    base = (base or "A video.").rstrip(". ")
    peak = int(loop.get("peak_frame", 0))
    returned = int(loop.get("return_frame", total_frames - 1))
    label = str(motion or "").strip().replace("_", " ")
    clause = (
        f" The camera-motion label is {label}; follow this camera path/direction exactly."
        if label and label.lower() not in {"unknown", "loop"}
        else ""
    )
    return (
        f"{base}. The camera trajectory is a loop: starting from frame 0, "
        f"the camera moves away to its furthest point at frame {peak}, then "
        f"returns to its starting framing by frame {returned} (out of "
        f"{total_frames} frames).{clause} This is camera motion only: "
        "exposure and illumination stay stable and normal throughout; do not "
        "dim the scene, fade to black, toggle the lights, or insert an "
        "occluder. During the away interval, the original subject or main "
        "action may leave the frame because of the camera motion while the "
        "visible adjacent background remains normally lit and from the same "
        "environment. Do not duplicate, teleport, refresh, or reintroduce the "
        "original subject in the off-screen view. When the camera returns, "
        "the same scene is revealed naturally and the visible state has "
        "advanced continuously rather than resetting. No scene cuts; no new "
        "people or objects enter the scene; preserve object identity, "
        "background, stable lighting, scale, and time continuity."
    )


def _default_event_params(kind: str, control: dict, *, pixel_frames: int, latent_frames: int, chunk_size: int) -> dict:  # fmt: skip
    """``_pixel_event_bounds`` + ``_default_event_params``: a three-phase schedule from ``start_chunk`` / ``end_chunk``."""
    num_chunks = math.ceil(latent_frames / chunk_size)
    start_chunk, end_chunk = _event_span(control, num_chunks)
    latent_start = min(latent_frames - 1, start_chunk * chunk_size)
    latent_end = min(latent_frames - 1, (end_chunk + 1) * chunk_size - 1)
    start = int(round(latent_start * (pixel_frames - 1) / max(1, latent_frames - 1)))
    end = int(round(latent_end * (pixel_frames - 1) / max(1, latent_frames - 1)))
    ramp = max(2, max(6, end - start) // 4)
    enter_end, hold_end = start + ramp, max(start + ramp + 2, end - ramp)
    if kind == "light":
        return {
            "t_off_start": start, "t_off_end": enter_end, "t_hold_start": enter_end, "t_hold_end": hold_end,
            "t_on_start": hold_end, "t_on_end": end, "low_gain": float(control.get("low_gain", 0.04)), "high_gain": 1.0,
        }  # fmt: skip
    label = str(control.get("label", "brown cardboard panel"))
    slug = str(control.get("slug", label.lower().replace(" ", "_")))
    return {
        "rgba_path": f"procedural/{slug}/occluder.png",
        "enter_side": str(control.get("enter_side", "top")), "exit_side": str(control.get("exit_side", "bottom")),
        "t_enter_start": start, "t_enter_end": enter_end, "t_hold_start": enter_end, "t_hold_end": hold_end,
        "t_exit_start": hold_end, "t_exit_end": end, "peak_scale": float(control.get("peak_scale", 1.9)), "force_rect_mask": True,
    }  # fmt: skip


def build_prompt(prompt: str, control: dict, *, pixel_frames: int, latent_frames: int, chunk_size: int) -> list[str]:  # fmt: skip
    """``build_prompt``: the chunk-local prompt list ReMind inference encodes.

    ``control["prompts"]`` (a list with one prompt per chunk) is this port's
    addition for custom inputs: it bypasses the schedule and is used verbatim.
    """
    kind = str(control.get("kind", "clean"))
    num_chunks = math.ceil(latent_frames / chunk_size)
    explicit = control.get("prompts")
    if explicit is not None:
        if len(explicit) != num_chunks:
            raise ValueError(
                f"expected {num_chunks} chunk prompts, got {len(explicit)}"
            )
        return [str(p) for p in explicit]
    args = (pixel_frames, latent_frames, chunk_size)
    if kind == "clean":
        return chunk_text_prompts(str(prompt), "none", {}, *args)
    if kind == "camera":
        if control.get("_trajectory_path"):
            return [str(prompt)] * num_chunks
        peak = int(
            round((pixel_frames - 1) * float(control.get("peak_fraction", 0.50)))
        )
        back = int(round((pixel_frames - 1) * float(control.get("return_fraction", 0.86))))  # fmt: skip
        degrees = float(control.get("degrees", 20.0))
        if str(control.get("axis", "yaw")) == "yaw":
            direction = "left" if degrees < 0 else "right"
        else:
            direction = "down" if degrees < 0 else "up"
        text = _loop_caption(str(prompt), {"peak_frame": peak, "return_frame": back}, pixel_frames, motion=f"pan_{control.get('axis', 'yaw')}_{direction}")  # fmt: skip
        return [text] * num_chunks
    recovery = int(control.get("recovery_chunks", 2))
    if control.get("params") is not None:
        params = dict(control["params"])
        meta = dict(control.get("meta") or params.get("meta") or {})
        return chunk_text_prompts(str(prompt), kind, params, *args, meta=meta, recovery_chunks=recovery)  # fmt: skip
    params = _default_event_params(kind, control, pixel_frames=pixel_frames, latent_frames=latent_frames, chunk_size=chunk_size)  # fmt: skip
    if kind == "light":
        return chunk_text_prompts(str(prompt), "light", params, *args, recovery_chunks=recovery)  # fmt: skip
    label = str(control.get("label", "brown cardboard panel"))
    slug = str(control.get("slug", label.lower().replace(" ", "_")))
    return chunk_text_prompts(str(prompt), "occluder", params, *args, meta={"slug": slug, "label": label, "caption": ""}, recovery_chunks=recovery)  # fmt: skip


def geometry(preset: dict) -> dict[str, int]:
    frames = int(preset.get("num_frames", 81))
    chunk_size = int(preset.get("chunk_size", 3))
    latent_frames = (frames - 1) // 4 + 1
    latent_frames -= latent_frames % chunk_size
    return {
        "frames": frames,
        "height": int(preset.get("height", 480)),
        "width": int(preset.get("width", 832)),
        "fps": int(preset.get("fps", 16)),
        "chunk_size": chunk_size,
        "latent_frames": latent_frames,
    }


def dry_run_summary(preset_path: Path | str, preset: dict) -> dict[str, Any]:
    """``dry_run_summary``: everything ``--dry-run`` reports, without loading weights."""
    g = geometry(preset)
    prompts = build_prompt(str(preset["prompt"]), preset["control"], pixel_frames=g["frames"], latent_frames=g["latent_frames"], chunk_size=g["chunk_size"])  # fmt: skip
    ext, _ = camera.build_pixel_cameras(preset["control"], frames=g["frames"], height=g["height"], width=g["width"])  # fmt: skip
    task = str(preset["task"])
    input_key = "input_image" if task == "i2v" else "input_video"
    seed_latent_frames = int(preset["seed_latent_frames"])
    return {
        "name": preset["name"],
        "preset": str(preset_path),
        "task": task,
        input_key: str(preset[f"_{input_key}_path"]),
        "control_kind": preset["control"]["kind"],
        "num_frames": g["frames"],
        "latent_frames": g["latent_frames"],
        "seed_latent_frames": seed_latent_frames,
        "seed_pixel_frames": 1 + 4 * (seed_latent_frames - 1),
        "chunk_size": g["chunk_size"],
        "rollout_steps": list(catalog.DENOISING_STEPS),
        "prompt_mode": "chunk_local",
        "prompt_schedule": prompts,
        "camera": camera.camera_metrics(ext),
    }
