"""Convert the upstream YAML presets to the JSON this port ships, and record upstream's dry runs.

    .venv-ref/bin/python tests/ref/presets_to_json.py

Writes ``remind_lucid/presets/*.json`` (+ ``assets/`` and ``trajectories/``,
copied verbatim) and ``tests/fixtures/presets_dry_run.json``.
"""

import _upstream  # noqa: F401
import json
import shutil
from pathlib import Path

import yaml

from pipeline.remind_inference import dry_run_summary, load_preset

OUT = _upstream.ROOT / "remind_lucid" / "presets"
REF = _upstream.REFERENCE
(OUT / "assets").mkdir(parents=True, exist_ok=True)
(OUT / "trajectories").mkdir(parents=True, exist_ok=True)

summaries = {}
for path in sorted((REF / "examples" / "presets").glob("*.yaml")):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("input_image", "input_video"):
        if key in data:
            src = (path.parent / data[key]).resolve()
            shutil.copyfile(src, OUT / "assets" / src.name)
            data[key] = f"assets/{src.name}"
    trajectory = (data.get("control") or {}).get("trajectory")
    if trajectory:
        src = (path.parent / trajectory).resolve()
        shutil.copyfile(src, OUT / "assets" / src.name)
        data["control"]["trajectory"] = f"assets/{src.name}"
    (OUT / f"{path.stem}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    preset_path, preset = load_preset(path)
    summary = dry_run_summary(preset_path, preset)
    summaries[path.stem] = {
        k: v
        for k, v in summary.items()
        if k not in ("preset", "input_image", "input_video")
    }

# Custom-control cases (no explicit params): the analytic camera, default occluder and light schedules.
base = yaml.safe_load(
    (REF / "examples" / "presets" / "05_dog_clean_i2v.yaml").read_text(encoding="utf-8")
)
custom = {
    "custom_camera_yaw": {"kind": "camera", "axis": "yaw", "degrees": -25.0},
    "custom_camera_pitch_oneway": {
        "kind": "camera",
        "axis": "pitch",
        "degrees": 15.0,
        "return": False,
    },
    "custom_occluder": {
        "kind": "occluder",
        "label": "red curtain",
        "enter_side": "left",
        "exit_side": "right",
        "start_chunk": 2,
        "end_chunk": 3,
    },
    "custom_light": {
        "kind": "light",
        "start_chunk": 1,
        "end_chunk": 4,
        "recovery_chunks": 1,
    },
}
tmp = REF / "examples" / "presets" / "_tmp_custom.yaml"
for name, control in custom.items():
    tmp.write_text(
        yaml.safe_dump({**base, "name": name, "control": control}), encoding="utf-8"
    )
    preset_path, preset = load_preset(tmp)
    summary = dry_run_summary(preset_path, preset)
    summaries[name] = {
        "control": control,
        **{
            k: v
            for k, v in summary.items()
            if k not in ("preset", "input_image", "input_video")
        },
    }
tmp.unlink()

traj = REF / "prompt_engineering"
index = yaml.safe_load((traj / "trajectory-index.yaml").read_text(encoding="utf-8"))
for entry in index["trajectories"]:
    src = traj / entry["file"]
    shutil.copyfile(src, OUT / "trajectories" / src.name)
    entry["file"] = src.name
(OUT / "trajectories" / "index.json").write_text(
    json.dumps(index, indent=2) + "\n", encoding="utf-8"
)
(_upstream.FIXTURES / "presets_dry_run.json").write_text(
    json.dumps(summaries, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(len(summaries), "dry runs;", len(list(OUT.glob("*.json"))), "presets")
