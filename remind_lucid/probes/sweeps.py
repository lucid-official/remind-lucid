"""Sweeps: families of presets that differ in one thing.

A sweep returns plain preset dicts (the JSON schema of
:mod:`remind_lucid.presets`), each tagged with ``sweep`` metadata, so the
CLI, the batch runner and the web demo run them like any other preset.
"""

from __future__ import annotations

import math

from remind_lucid.text.schedule import phase_prompts


def _public(preset: dict) -> dict:
    return {k: v for k, v in preset.items() if not k.startswith("_")}


def _chunks(preset: dict) -> int:
    chunk_size = int(preset.get("chunk_size", 3))
    latent = (int(preset.get("num_frames", 81)) - 1) // 4 + 1
    return math.ceil((latent - latent % chunk_size) / chunk_size)


def interruption_phases(num_chunks: int, start_chunk: int, hold_chunks: int, recovery_chunks: int = 2) -> list[str]:  # fmt: skip
    """``none.. | enter | hold x N | exit | recover x R | none..``; ``hold_chunks = -1`` is no interruption at all."""
    if hold_chunks < 0:
        return ["none"] * num_chunks
    phases = ["none"] * start_chunk + ["enter"] + ["hold"] * hold_chunks + ["exit"]
    if len(phases) > num_chunks:
        raise ValueError(f"interruption of {hold_chunks} hold chunks from chunk {start_chunk} does not fit {num_chunks} chunks")  # fmt: skip
    phases += ["recover"] * min(recovery_chunks, num_chunks - len(phases))  # the tail may leave no room to recover
    return phases + ["none"] * (num_chunks - len(phases))


def interruption_sweep(preset: dict, hold_chunks: list[int], *, start_chunk: int | None = None, kind: str | None = None, recovery_chunks: int = 2) -> list[dict]:  # fmt: skip
    """Vary how long the scene is out of sight: 0, 1, 2, ... fully hidden chunks (12 frames = 0.75 s each).

    The interruption is conveyed only by chunk-local text, so its length is
    quantised to chunks. ``-1`` gives the uninterrupted control. The
    appearance text (occluder description, sides) comes from the preset.
    """
    base = _public(preset)
    control = dict(base.get("control") or {})
    kind = kind or (control.get("kind") if control.get("kind") in ("occluder", "light") else "occluder")  # fmt: skip
    params = dict(control.get("params") or {"enter_side": control.get("enter_side", "right"), "exit_side": control.get("exit_side", "left")})  # fmt: skip
    meta = dict(control.get("meta") or {"label": control.get("label", "brown cardboard box")})  # fmt: skip
    n = _chunks(base)
    seed_chunks = math.ceil(int(base.get("seed_latent_frames", 1)) / int(base.get("chunk_size", 3)))  # fmt: skip
    start = max(seed_chunks, 2) if start_chunk is None else start_chunk
    out = []
    for hold in hold_chunks:
        phases = interruption_phases(n, start, hold, recovery_chunks)
        variant = dict(base)
        variant["name"] = f"{base['name']}_hold{hold if hold >= 0 else 'none'}"
        variant["control"] = {"kind": kind, "params": params, "meta": meta, "prompts": phase_prompts(str(base["prompt"]), kind, phases, params, meta)}  # fmt: skip
        variant["sweep"] = {"kind": "interruption", "hold_chunks": hold, "start_chunk": start, "phases": phases}  # fmt: skip
        out.append(variant)
    return out


def seed_sweep(preset: dict, seeds: list[int]) -> list[dict]:
    out = []
    for seed in seeds:
        variant = _public(preset)
        variant["name"] = f"{variant['name']}_seed{seed}"
        variant["seed"] = int(seed)
        variant["sweep"] = {"kind": "seed", "seed": int(seed)}
        out.append(variant)
    return out


def gap_sweep(gaps: list[int], from_chunk: int) -> list[dict]:
    """Probe configurations for a RoPE gap sweep (``ProbeSet.from_dict`` inputs)."""
    return [{"gap_from_chunk": from_chunk, "gap_frames": int(g)} for g in gaps]
