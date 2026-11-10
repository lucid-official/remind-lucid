"""Every bundled preset resolves to the schedule upstream's ``--dry-run`` prints."""

import json

import pytest

from conftest import FIXTURES
from remind_lucid import presets

PATH = FIXTURES / "presets_dry_run.json"
REF = json.loads(PATH.read_text(encoding="utf-8")) if PATH.exists() else {}
CAMERA_TOL = 1e-9


def _check(ours, ref):
    for key in ("name", "task", "control_kind", "num_frames", "latent_frames", "seed_latent_frames",
                "seed_pixel_frames", "chunk_size", "rollout_steps", "prompt_schedule"):  # fmt: skip
        assert ours[key] == ref[key], key
    for key, value in ref["camera"].items():
        assert ours["camera"][key] == pytest.approx(value, abs=1e-4), key


def test_all_eleven_presets_are_bundled():
    assert len(presets.preset_names()) == 11
    assert len(presets.trajectory_names()) == 12


@pytest.mark.parametrize("name", [n for n in REF if not n.startswith("custom_")])
def test_bundled_preset(name):
    path, preset = presets.load_preset(name)
    _check(presets.dry_run_summary(path, preset), REF[name])


@pytest.mark.parametrize("name", [n for n in REF if n.startswith("custom_")])
def test_custom_control(name):
    _, base = presets.load_preset("05")
    data = {k: v for k, v in base.items() if not k.startswith("_")}
    data.update(name=name, control=REF[name]["control"])
    preset = presets.validate(data, presets.PRESET_DIR)
    _check(presets.dry_run_summary("custom", preset), REF[name])
