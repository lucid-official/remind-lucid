"""Chunk-local prompts for reversible visibility events (``pipeline/prompt_schedule.py``).

The preset's caption describes the scene only. An occluder or a light toggle
is told to the model through the text of the chunks it spans: *enter*,
*hold*, *exit*, then a *recover* clause on the chunks that follow. The
clause texts are the reference's, word for word — the model was trained on
them.
"""

from __future__ import annotations

import math
import os
from typing import Any, Sequence


def _base_caption(caption: str) -> str:
    return str(caption or "A video.").strip().rstrip(". ") + "."


def _short_description(meta: dict | None, params: dict) -> str:
    meta = meta or {}
    label = str(meta.get("label") or "").strip()
    rich = str(meta.get("caption") or "").strip().rstrip(". ")
    if not label:
        slug = str(meta.get("slug") or "").strip()
        if not slug:
            slug = os.path.basename(os.path.dirname(str(params.get("rgba_path") or "")))
        label = (slug or "opaque occluder").replace("_", " ")
    if rich and rich.lower() != label.lower():
        return f"{label}, {' '.join(rich.split()[:40])}"
    return label


def pixel_to_chunk(pixel_index: int, pixel_frames: int, latent_frames: int, chunk_size: int) -> int:  # fmt: skip
    if latent_frames <= 1:
        return 0
    latent = round(max(0, int(pixel_index)) * float(latent_frames - 1) / float(max(1, pixel_frames - 1)))  # fmt: skip
    return int(max(0, min(latent_frames - 1, latent))) // chunk_size


def _event_bounds(kind: str, params: dict) -> Sequence[int] | None:
    if kind == "light":
        keys = ("t_off_start", "t_off_end", "t_hold_end", "t_on_end")
    elif kind == "occluder":
        keys = ("t_enter_start", "t_enter_end", "t_hold_end", "t_exit_end")
    else:
        return None
    if not all(key in params for key in keys):
        return None
    return tuple(int(params[key]) for key in keys)


def _light_clauses() -> dict[str, str]:
    return {
        "enter": "During this segment, the lights are turning off and the same scene becomes progressively darker while its action continues.",
        "hold": "During this entire segment, the lights remain fully off. The same scene is dark and invisible while its underlying action continues out of sight.",
        "exit": "During this segment, the lights turn back on and progressively reveal the same continuously evolving scene.",
        "recover": "The lights are fully on again in this segment. The original scene is visible and its action continues from before the darkness without resetting.",
        "brief": "During this segment, the lights briefly turn off and back on within the segment: the scene darkens, stays dark for a moment, then returns to full brightness while its action continues without resetting.",
    }  # fmt: skip


def _occluder_clauses(description: str, enter_side: str, exit_side: str) -> dict[str, str]:  # fmt: skip
    return {
        "enter": f"During this segment, the same {description} enters from the {enter_side} and progressively covers the camera view without interacting with the scene.",
        "hold": f"Throughout this segment, the same {description} remains in front of the camera and completely blocks the scene. The underlying action continues out of sight.",
        "exit": f"During this segment, the same {description} moves out toward the {exit_side} and progressively reveals the original, continuously evolving scene.",
        "recover": f"The {description} is completely gone in this segment. The original scene is fully visible and its action continues from before the occlusion without resetting.",
        "brief": f"During this segment, the same {description} briefly enters, covers the camera view, and leaves again within the segment, revealing the original continuously evolving scene by the segment's end.",
    }  # fmt: skip


def _event_chunk_plan(base: str, kind: str, params: dict, meta: dict | None, pixel_frames: int, latent_frames: int, chunk_size: int, num_chunks: int) -> dict[str, Any] | None:  # fmt: skip
    bounds = _event_bounds(kind, params)
    if bounds is None:
        return None
    start, enter_end, hold_end, end = bounds
    start_chunk = pixel_to_chunk(start, pixel_frames, latent_frames, chunk_size)
    end_chunk = pixel_to_chunk(end, pixel_frames, latent_frames, chunk_size)
    if kind == "light":
        clauses = _light_clauses()
    else:
        clauses = _occluder_clauses(
            _short_description(meta or params.get("meta"), params),
            str(params.get("enter_side") or "side"),
            str(params.get("exit_side") or "side"),
        )
    finishes = end < pixel_frames
    enter_finishes = enter_end < pixel_frames
    hold_finishes = hold_end < pixel_frames

    def centre_pixel(chunk: int) -> float:
        latent = min(float(latent_frames - 1), chunk * chunk_size + (chunk_size - 1) / 2.0)  # fmt: skip
        if latent_frames <= 1:
            return 0.0
        return latent * float(pixel_frames - 1) / float(latent_frames - 1)

    single = start_chunk == end_chunk and finishes
    chunk_prompts: dict[int, str] = {}
    for chunk in range(start_chunk, end_chunk + 1):
        if chunk >= num_chunks:
            continue
        if single:
            phase = "brief"
        elif chunk == start_chunk:
            phase = "enter"
        elif chunk == end_chunk and finishes:
            phase = "exit"
        elif not enter_finishes:
            phase = "enter"
        elif not hold_finishes:
            phase = "enter" if centre_pixel(chunk) < enter_end else "hold"
        else:
            centre = centre_pixel(chunk)
            phase = "enter" if centre < enter_end else "hold" if centre < hold_end else "exit"  # fmt: skip
        chunk_prompts[chunk] = f"{base} {clauses[phase]}"
    return {
        "chunk_prompts": chunk_prompts,
        "recover_prompt": f"{base} {clauses['recover']}",
        "last_event_chunk": end_chunk,
        "finishes_in_clip": finishes,
        "start_chunk": start_chunk,
    }


def chunk_text_prompts_multi(caption: str, events: Sequence[dict], pixel_frames: int, latent_frames: int, chunk_size: int, *, recovery_chunks: int = 2) -> list[str]:  # fmt: skip
    """One prompt per latent chunk for zero or more ``{"kind", "params", "meta"}`` events."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    base = _base_caption(caption)
    num_chunks = math.ceil(latent_frames / chunk_size)
    prompts = [base for _ in range(num_chunks)]
    plans = []
    for event in events or []:
        if not isinstance(event, dict):
            continue
        kind = str(event.get("kind") or "none").strip().lower()
        params = event.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        meta = event.get("meta") or params.get("meta")
        if not isinstance(meta, dict):
            meta = None
        plan = _event_chunk_plan(base, kind, params, meta, pixel_frames, latent_frames, chunk_size, num_chunks)  # fmt: skip
        if plan is not None:
            plans.append(plan)
    plans.sort(key=lambda item: item["start_chunk"])
    claimed = set()
    for plan in plans:
        for chunk, text in plan["chunk_prompts"].items():
            prompts[chunk] = text
            claimed.add(chunk)
    for plan in plans:
        if not plan["finishes_in_clip"]:
            continue
        for offset in range(1, max(0, int(recovery_chunks)) + 1):
            chunk = plan["last_event_chunk"] + offset
            if chunk >= num_chunks or chunk in claimed:
                break
            prompts[chunk] = plan["recover_prompt"]
    return prompts


def chunk_text_prompts(caption: str, kind: str, params: dict | None, pixel_frames: int, latent_frames: int, chunk_size: int, *, meta: dict | None = None, recovery_chunks: int = 2) -> list[str]:  # fmt: skip
    """Single-event wrapper of :func:`chunk_text_prompts_multi`."""
    return chunk_text_prompts_multi(caption, [{"kind": kind, "params": params, "meta": meta}], pixel_frames, latent_frames, chunk_size, recovery_chunks=recovery_chunks)  # fmt: skip


PHASES = ("none", "enter", "hold", "exit", "recover", "brief")


def phase_prompts(caption: str, kind: str, phases: Sequence[str], params: dict | None = None, meta: dict | None = None) -> list[str]:  # fmt: skip
    """One prompt per chunk from an explicit phase per chunk (not in upstream).

    ``chunk_text_prompts`` derives the phase of each chunk from pixel-frame
    timings; the conditioning the model receives is only the resulting text.
    This builds the same texts from the phases directly (``"none"`` is the
    bare caption), which is what the interruption-duration probe varies.
    """
    params = dict(params or {})
    base = _base_caption(caption)
    if kind == "light":
        clauses = _light_clauses()
    else:
        clauses = _occluder_clauses(
            _short_description(meta or params.get("meta"), params),
            str(params.get("enter_side") or "side"),
            str(params.get("exit_side") or "side"),
        )
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        raise ValueError(f"unknown phases {unknown}; choose from {PHASES}")
    return [base if p == "none" else f"{base} {clauses[p]}" for p in phases]


def phases_of(caption: str, kind: str, prompts: Sequence[str], params: dict | None = None, meta: dict | None = None) -> list[str]:  # fmt: skip
    """The phase of each prompt of a schedule built by this module (``"none"`` if it is the bare caption)."""
    table = {text: phase for phase in PHASES for text in phase_prompts(caption, kind, [phase], params, meta)}  # fmt: skip
    return [table.get(p, "custom") for p in prompts]
