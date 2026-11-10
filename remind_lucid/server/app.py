"""Web demo: weights, generation with live frames, a gallery with comparison, and probes.

    python -m remind_lucid.server                 # binds the Tailscale IP if there is one, else 127.0.0.1; port 7860
    python -m remind_lucid.server --host 0.0.0.0  # any interface

Standard library only (``http.server`` + Server-Sent Events). One worker
thread owns the models and runs one generation at a time; HTTP threads stay
responsive because Lucid releases the GIL while the GPU works. Every run is
kept under ``outputs/web/<id>/`` (video, metadata, thumbnail, probe report),
so the gallery survives restarts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import queue
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import lucid

from remind_lucid import catalog, download, presets
from remind_lucid.io.media import VideoWriter, encode_jpeg, to_uint8_frames
from remind_lucid.pipeline import DTYPES, Job, NoiseSource, ReMindPipeline, memory_gb, peak_memory_gb
from remind_lucid.probes import ProbeSet
from remind_lucid.probes import sweeps
from remind_lucid.probes.heatmap import render as render_heatmap
from remind_lucid.text.schedule import PHASES, phase_prompts, phases_of

STATIC = Path(__file__).resolve().parent / "static"
OUTPUTS = catalog.ROOT / "outputs" / "web"
UPLOADS = OUTPUTS / "uploads"
_CTYPES = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".png": "image/png", ".json": "application/json", ".html": "text/html; charset=utf-8"}  # fmt: skip


# --------------------------------------------------------------------------- #
# Requests -> presets
# --------------------------------------------------------------------------- #


def _save_upload(data_url: str) -> Path:
    """Store an uploaded image/video (a ``data:`` URL) under a content hash; return its path."""
    head, _, payload = data_url.partition(",")
    raw = base64.b64decode(payload)
    ext = ".mp4" if "video" in head else ".png" if "png" in head else ".jpg" if ("jpeg" in head or "jpg" in head) else ".bin"  # fmt: skip
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{hashlib.sha256(raw).hexdigest()[:20]}{ext}"
    if not path.exists():
        path.write_bytes(raw)
    return path


def _known_input(value: str) -> Path:
    """A path the browser may name: an earlier upload or a bundled preset input, nothing else on disk."""
    path = Path(value).resolve()
    roots = (UPLOADS.resolve(), (presets.PRESET_DIR / "assets").resolve())
    if not path.is_file() or not any(root in path.parents for root in roots):
        raise ValueError("input must be an uploaded file or a bundled preset asset")
    return path


def build_preset(spec: dict) -> dict:
    """A validated preset from a request: a bundled preset, optionally with fields laid over it.

    Recognised fields: ``preset`` (bundled name), ``name``, ``prompt``,
    ``seed``, ``image`` / ``video`` (data URL or an earlier upload's path),
    ``control`` (the preset schema), ``phases`` (one of
    :data:`remind_lucid.text.schedule.PHASES` per chunk, for occluder/light
    controls) and ``prompts`` (seven explicit chunk prompts).
    """
    base = spec.get("preset") or "05_dog_clean_i2v"
    path = presets.resolve_preset_path(base)
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("name", "prompt", "seed"):
        if spec.get(key) not in (None, ""):
            data[key] = spec[key]
    for kind, task, other in (("image", "i2v", "video"), ("video", "v2v", "image")):
        value = spec.get(kind)
        if value:
            source = _save_upload(value) if str(value).startswith("data:") else _known_input(str(value))  # fmt: skip
            data.update({"task": task, f"input_{kind}": str(source.resolve())})
            data.pop(f"input_{other}", None)
            if task == "i2v":
                data.pop("seed_latent_frames", None)
            else:
                data.setdefault("seed_latent_frames", 6)
            data.pop("website_case", None)
    if spec.get("control") is not None:
        data["control"] = dict(spec["control"])
    control = dict(data.get("control") or {"kind": "clean"})
    if spec.get("phases"):
        kind = control.get("kind") if control.get("kind") in ("occluder", "light") else "occluder"  # fmt: skip
        params = dict(control.get("params") or {"enter_side": control.get("enter_side", "right"), "exit_side": control.get("exit_side", "left")})  # fmt: skip
        meta = dict(control.get("meta") or {"label": control.get("label", "brown cardboard box")})  # fmt: skip
        control = {"kind": kind, "params": params, "meta": meta, "prompts": phase_prompts(str(data["prompt"]), kind, spec["phases"], params, meta)}  # fmt: skip
    if spec.get("prompts"):
        control["prompts"] = [str(p) for p in spec["prompts"]]
    data["control"] = control
    if spec.get("sweep_tag"):
        data["sweep"] = spec["sweep_tag"]
    return presets.validate(data, path.parent)


def expand_batch(spec: dict) -> list[tuple[dict, dict | None]]:
    """``(preset, probe config)`` pairs for a probe sweep request."""
    sweep = spec.get("sweep") or {}
    kind = sweep.get("kind")
    base = build_preset(spec)
    probes = spec.get("probes")
    if kind == "interruption":
        holds = [int(v) for v in sweep.get("hold_chunks", [-1, 0, 1, 2])]
        variants = sweeps.interruption_sweep(base, holds, start_chunk=sweep.get("start_chunk"), kind=sweep.get("event"))  # fmt: skip
        return [(presets.validate(v, presets.PRESET_DIR), probes) for v in variants]
    if kind == "seed":
        variants = sweeps.seed_sweep(base, [int(v) for v in sweep.get("seeds", [1, 2, 3, 4])])  # fmt: skip
        return [(presets.validate(v, presets.PRESET_DIR), probes) for v in variants]
    if kind == "gap":
        out = []
        for config in sweeps.gap_sweep([int(v) for v in sweep.get("gaps", [0, 8, 32, 128])], int(sweep.get("from_chunk", 3))):  # fmt: skip
            variant = {k: v for k, v in base.items() if not k.startswith("_")}
            variant["name"] = f"{base['name']}_gap{config['gap_frames']}"
            variant["sweep"] = {"kind": "gap", **config}
            out.append((presets.validate(variant, presets.PRESET_DIR), {**(probes or {}), **config}))  # fmt: skip
        return out
    if kind == "knockout":
        out = []
        at = int(sweep.get("at_chunk", 5))
        for drop in sweep.get("sets", [[], [0], [1], [2]]):
            variant = {k: v for k, v in base.items() if not k.startswith("_")}
            tag = "-".join(str(d) for d in drop) or "none"
            variant["name"] = f"{base['name']}_drop{tag}"
            variant["sweep"] = {"kind": "knockout", "drop": drop, "at_chunk": at}
            config = {**(probes or {})}
            if drop:
                config["knockout"] = {str(at): [int(d) for d in drop]}
            out.append((presets.validate(variant, presets.PRESET_DIR), config))
        return out
    return [(base, probes)]


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


@dataclass
class Run:
    id: str
    preset: dict
    probes: dict | None
    seed: int | None
    batch: str | None = None
    events: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    stop: bool = False
    state: str = "queued"

    def emit(self, kind: str, **data) -> None:
        self.events.put({"type": kind, "t": time.time(), **data})

    @property
    def folder(self) -> Path:
        return OUTPUTS / self.id


class Engine:
    """Owns the pipeline; runs queued generations one at a time on its own thread."""

    def __init__(self, dtype: str = "bf16", accelerate: bool = False) -> None:
        self.dtype_name = dtype
        self.pipe = ReMindPipeline(dtype=DTYPES[dtype], accelerate=accelerate)
        self.jobs: "queue.Queue[Run]" = queue.Queue()
        self.runs: dict[str, Run] = {}
        self.current: Run | None = None
        self.download_events: list[dict] = []
        self.downloading = False
        threading.Thread(target=self._loop, daemon=True, name="remind-engine").start()

    def submit(self, run: Run) -> None:
        self.runs[run.id] = run
        self.jobs.put(run)

    def queue_state(self) -> dict:
        return {
            "current": self.current.id if self.current else None,
            "queued": [r.id for r in self.runs.values() if r.state == "queued"],
            "memory_gb": round(memory_gb(), 2),
        }

    def _loop(self) -> None:
        while True:
            run = self.jobs.get()
            if run.stop:
                run.state = "cancelled"
                run.emit("end")
                continue
            self.current = run
            run.state = "running"
            try:
                self._run(run)
                run.state = "done"
            except Exception as exc:  # report to the browser, keep serving
                traceback.print_exc()
                run.state = "error"
                run.emit("error", message=f"{type(exc).__name__}: {exc}")
            finally:
                run.emit("end")
                self.current = None

    def _run(self, run: Run) -> None:
        pipe = self.pipe
        pipe.log = lambda message: run.emit("status", message=message)
        pipe.timings = type(pipe.timings)()
        job = Job.from_preset(run.preset.get("_source", "web"), run.preset, run.seed)
        t_start = time.time()
        lucid.metal.reset_peak_memory_stats()
        if not pipe.prompt_encoder.cached(job.prompts):
            run.emit("status", message="encoding prompts with UMT5-XXL (first time for this text) ...")  # fmt: skip
            pipe.release_generator()  # make room for the 11 GB text encoder
        features = pipe.encode_prompts(job.prompts)
        run.emit("status", message="encoding the input with the VAE ...")
        source = pipe.encode_source(job)
        if pipe.generator is None:
            run.emit("status", message="loading the generator and merging the DMD EMA LoRA ...")  # fmt: skip
        pipe.ensure_generator()
        pipe.probes = ProbeSet.from_dict(run.probes)
        decode = pipe.decoder()
        run.folder.mkdir(parents=True, exist_ok=True)
        video = run.folder / "video.mp4"
        writer = VideoWriter(video, job.width, job.height, fps=job.fps)
        run.emit("start", name=run.preset["name"], frames=job.frames, chunks=job.num_chunks, width=job.width, height=job.height, fps=job.fps, prompts=job.prompts)  # fmt: skip
        index, decode_seconds = 0, 0.0
        stream = pipe.rollout(job, features, source, NoiseSource(job.seed, pipe.dtype))
        try:
            for chunk in stream:
                t0 = time.time()
                pixels = decode(chunk.latents)
                rgb = to_uint8_frames(pixels)
                writer.write(rgb)
                decode_seconds += time.time() - t0
                jpegs = [base64.b64encode(encode_jpeg(f, job.width, job.height)).decode() for f in rgb]  # fmt: skip
                if index == 0:
                    (run.folder / "thumb.jpg").write_bytes(base64.b64decode(jpegs[0]))
                run.emit("frames", start=index, images=jpegs, chunk=chunk.index, seconds=round(chunk.seconds, 2), kv_tokens=chunk.kv_tokens, memory_gb=round(memory_gb(), 2))  # fmt: skip
                index += len(rgb)
                if run.stop:
                    break
        finally:
            stream.close()
            writer.close()
            probes, pipe.probes = pipe.probes, None
            if probes is not None:
                probes.end()
        metadata = job.metadata()
        metadata.update(
            id=run.id,
            batch=run.batch,
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
            output=str(video),
            frames_written=index,
            stopped=run.stop,
            sweep=run.preset.get("sweep"),
            control=_public(run.preset.get("control") or {}),
            probes=probes.to_dict() if probes is not None else None,
            lucid_version=lucid.__version__,
            dtype=self.dtype_name,
            compile=pipe.accelerate,
            timings={**pipe.timings.stages, "chunks": pipe.timings.chunks, "vae_decode": round(decode_seconds, 3), "total": round(time.time() - t_start, 3)},  # fmt: skip
            memory_gb={"peak": round(peak_memory_gb(), 3), **pipe.timings.memory},
        )
        if probes is not None and probes.mass:
            report = probes.report()
            (run.folder / "probes.json").write_text(json.dumps(report), encoding="utf-8")
            last = max(int(c) for c in report["attention_mass"])
            render_heatmap(report, last, run.folder / "heatmap.png")
            metadata["attention_report"] = True
        (run.folder / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")  # fmt: skip
        run.emit("done", id=run.id, frames=index, video=f"/api/file/{run.id}/video.mp4", metadata=json.loads(json.dumps(metadata, default=str)))  # fmt: skip

    # -- weights -------------------------------------------------------------- #

    def start_download(self) -> bool:
        if self.downloading:
            return False
        self.downloading = True
        self.download_events.clear()

        def progress(name: str, done: int, total: int) -> None:
            self.download_events.append({"name": name, "done": done, "total": total, "t": time.time()})  # fmt: skip
            del self.download_events[:-200]

        def work() -> None:
            try:
                download.fetch_all(progress)
                self.download_events.append({"name": "done", "done": 1, "total": 1, "t": time.time()})  # fmt: skip
            except Exception as exc:
                traceback.print_exc()
                self.download_events.append({"error": f"{type(exc).__name__}: {exc}", "t": time.time()})  # fmt: skip
            finally:
                self.downloading = False

        threading.Thread(target=work, daemon=True, name="remind-download").start()
        return True


def _public(mapping: dict) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in mapping.items() if not k.startswith("_")}  # fmt: skip


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _history() -> list[dict]:
    rows = []
    for meta in sorted(OUTPUTS.glob("*/metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True):  # fmt: skip
        try:
            rows.append(json.loads(meta.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return rows


def _info() -> dict:
    cases = []
    for name in presets.preset_names():
        path, preset = presets.load_preset(name)
        job_prompts = presets.build_prompt(str(preset["prompt"]), preset["control"], pixel_frames=81, latent_frames=21, chunk_size=3)  # fmt: skip
        control = preset["control"]
        kind = control["kind"]
        input_key = "input_image" if preset["task"] == "i2v" else "input_video"
        cases.append({
            "key": name, "name": preset["name"], "task": preset["task"], "kind": kind, "prompt": preset["prompt"],
            "seed": preset["seed"], "seed_latent_frames": preset["seed_latent_frames"],
            "input": f"/api/asset/{Path(preset[f'_{input_key}_path']).name}",
            "control": _public(control), "prompts": job_prompts,
            "phases": phases_of(str(preset["prompt"]), kind, job_prompts, control.get("params"), control.get("meta")) if kind in ("occluder", "light") else None,
        })  # fmt: skip
    return {
        "presets": cases,
        "trajectories": presets.trajectory_index(),
        "phases": list(PHASES),
        "device": lucid.metal.get_device_name() if lucid.metal.is_available() else "cpu",
        "lucid": lucid.__version__,
        "ready": catalog.ready(),
        "steps": list(catalog.DENOISING_STEPS),
    }


class Handler(BaseHTTPRequestHandler):
    engine: Engine

    def log_message(self, fmt, *args) -> None:  # quiet
        pass

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):  # <video> seeks with range requests
            lo, _, hi = rng[6:].partition("-")
            start = int(lo) if lo else max(0, size - int(hi))
            end = min(size - 1, int(hi)) if (lo and hi) else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", _CTYPES.get(path.suffix, "application/octet-stream"))  # fmt: skip
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            try:
                self.wfile.write(fh.read(end - start + 1))
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _sse(self, next_event) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        while True:
            event = next_event()
            try:
                self.wfile.write(f"data: {json.dumps(event, default=str)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            if event["type"] == "end":
                return

    def do_GET(self) -> None:
        url = urlparse(self.path)
        engine = self.engine
        if url.path in ("/", "/index.html"):
            return self._file(STATIC / "index.html")
        if url.path == "/api/info":
            return self._json(_info())
        if url.path == "/api/weights":
            free = shutil.disk_usage(catalog.CHECKPOINTS if catalog.CHECKPOINTS.exists() else catalog.ROOT).free  # fmt: skip
            last = engine.download_events[-1] if engine.download_events else None
            return self._json({"files": catalog.status(), "free": free, "downloading": engine.downloading, "progress": last, "ready": catalog.ready()})  # fmt: skip
        if url.path == "/api/jobs":
            return self._json({"jobs": _history(), **engine.queue_state()})
        if url.path.startswith("/api/asset/"):
            return self._file(presets.PRESET_DIR / "assets" / Path(url.path).name)
        if url.path.startswith("/api/file/"):
            _, _, _, run_id, name = url.path.split("/", 4)
            return self._file(OUTPUTS / Path(run_id).name / Path(name).name)
        if url.path == "/api/events":
            run = engine.runs.get(parse_qs(url.query).get("job", [""])[0])
            if run is None:
                return self._json({"error": "unknown job"}, 404)

            def next_event() -> dict:
                try:
                    return run.events.get(timeout=15)
                except queue.Empty:
                    return {"type": "ping"}

            return self._sse(next_event)
        self.send_error(404)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json({"error": "invalid JSON"}, 400)
        engine = self.engine
        try:
            if url.path == "/api/validate":
                preset = build_preset(body)
                summary = presets.dry_run_summary(body.get("preset") or "custom", preset)
                kind = preset["control"]["kind"]
                summary["phases"] = phases_of(str(preset["prompt"]), kind, summary["prompt_schedule"], preset["control"].get("params"), preset["control"].get("meta"))  # fmt: skip
                return self._json({"ok": True, "summary": summary})
            if url.path in ("/api/generate", "/api/batch"):
                if not catalog.ready():
                    return self._json({"error": "weights are missing; download them in the Weights tab"}, 400)  # fmt: skip
                pairs = expand_batch(body) if url.path == "/api/batch" else [(build_preset(body), body.get("probes"))]  # fmt: skip
                batch = uuid.uuid4().hex[:8] if len(pairs) > 1 else None
                ids = []
                for preset, probes in pairs:
                    seed = body.get("seed") if (preset.get("sweep") or {}).get("kind") != "seed" else None  # fmt: skip
                    run = Run(id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6], preset=preset, probes=probes, seed=int(seed) if seed not in (None, "") else None, batch=batch)  # fmt: skip
                    engine.submit(run)
                    ids.append(run.id)
                return self._json({"jobs": ids, "batch": batch})
            if url.path == "/api/stop":
                run = engine.runs.get(body.get("job", ""))
                if run is None:
                    return self._json({"error": "unknown job"}, 404)
                run.stop = True
                return self._json({"ok": True})
            if url.path == "/api/delete":
                folder = OUTPUTS / Path(str(body.get("job", ""))).name
                if folder.is_dir() and (folder / "metadata.json").exists():
                    shutil.rmtree(folder)
                return self._json({"ok": True})
            if url.path == "/api/weights/download":
                return self._json({"started": engine.start_download()})
        except (ValueError, FileNotFoundError, KeyError) as exc:
            return self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)
        self.send_error(404)


def _tailscale_ip() -> str | None:
    exe = shutil.which("tailscale")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=5).stdout.split()  # fmt: skip
        return out[0] if out else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--host", help="default: the Tailscale IPv4 address, else 127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()
    host = args.host or _tailscale_ip() or "127.0.0.1"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    Handler.engine = Engine(args.dtype, args.compile)
    server = ThreadingHTTPServer((host, args.port), Handler)
    server.daemon_threads = True
    print(f"ReMind demo on http://{host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
