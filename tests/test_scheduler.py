"""The flow-matching table and its lookups are those of ``utils/scheduler.py``, bit for bit."""

import lucid

from conftest import load_fixture
from remind_lucid import camera
from remind_lucid.scheduler import FlowMatchSchedule


def test_table_bit_exact():
    ref = load_fixture("schedule_torch.safetensors")
    s = FlowMatchSchedule(shift=5.0)
    assert (s.sigmas == ref["sigmas"]).all().item()
    assert (s.timesteps == ref["timesteps"]).all().item()
    assert [s.index(t) for t in (1000, 938, 833, 625, 0)] == ref["probe_index"].tolist()


def test_add_noise_and_x0():
    ref = load_fixture("schedule_torch.safetensors")
    s = FlowMatchSchedule(shift=5.0)
    x, n = ref["x"].unsqueeze(0), ref["noise"].unsqueeze(0)
    noisy = s.add_noise(x, n, [0, 833, 833])
    assert noisy.dtype == lucid.bfloat16
    assert (noisy[0].float() == ref["noisy"].float()).all().item()
    x0 = s.x0_from_flow(x, n, [0, 833, 833])  # flow = x, xt = noise in the fixture
    assert (x0[0].float() == ref["x0"].float()).all().item()


def test_latent_camera_indices():
    ref = load_fixture("schedule_torch.safetensors")
    ext, ks = camera.neutral_cameras(81, 480, 832)
    marks = lucid.arange(81, dtype=lucid.float32).reshape(1, 81, 1, 1) + ext
    picked, _ = camera.pixel_to_latent(marks, ks, 21)
    assert (picked[0, :, 0, 1].to(lucid.int64) == ref["latent_index"]).all().item()
