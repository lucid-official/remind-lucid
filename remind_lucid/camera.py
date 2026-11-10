"""Camera trajectories and the projective matrices of PM-RoPE.

Ports the camera handling of ``pipeline/remind_inference.py`` (trajectory
loading, the analytic fallback, pixel -> latent subsampling, metrics), the
per-call normalisation at the top of ``CausalWanModel.forward`` and the
matrix construction of ``prope/camera_rope.py`` (``_invert_SE3``,
``_lift_K``, ``_invert_K``, ``P = lift(K) @ viewmats``).

All of it is float32 arithmetic on a handful of 4x4 matrices, done on Lucid's
CPU backend; only the finished matrices move to the GPU.

Two properties of the reference that are easy to miss are kept as they are:

* translations are centred and scaled over the frames *of the call* — a
  3-frame chunk — not over the whole trajectory;
* both rows of the intrinsics are divided by the image *width*, and the
  principal point is dropped before ``P`` is built.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import lucid

from remind_lucid.io.npy import load_npz

Tensor = lucid.Tensor


def neutral_cameras(frames: int, height: int, width: int) -> tuple[Tensor, Tensor]:
    """``_neutral_cameras``: identity poses and a ``max(H, W)`` focal length. ``(1, F, 4, 4)``, ``(1, F, 3, 3)``."""
    focal = float(max(height, width))
    k = [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]]
    eye = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    return (
        lucid.tensor([[eye] * frames], dtype=lucid.float32),
        lucid.tensor([[k] * frames], dtype=lucid.float32),
    )


def _parse_resolution(value) -> tuple[int | None, int | None]:
    if not value:
        return None, None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    text = str(value).lower().replace(" ", "")
    if "x" not in text:
        return None, None
    width, height = text.split("x", 1)
    try:
        return int(width), int(height)
    except ValueError:
        return None, None


def load_trajectory(path: str | Path, frames: int, height: int, width: int, source_resolution=None) -> tuple[Tensor, Tensor]:  # fmt: skip
    """``_camera_npz_to_tensors`` for frame indices ``0..frames-1`` (clamped). ``(1, F, 4, 4)``, ``(1, F, 3, 3)``."""
    arrays = load_npz(path)
    ext_key = next((k for k in ("extrinsics", "c2w", "poses") if k in arrays), None)
    int_key = next((k for k in ("intrinsics", "K", "Ks") if k in arrays), None)
    if ext_key is None:
        raise KeyError(f"camera npz has no extrinsics key; keys={sorted(arrays)}")
    ext = arrays[ext_key].to(lucid.float32)
    if ext.ndim == 2:
        ext = ext.unsqueeze(0)
    n = ext.shape[0]
    if tuple(ext.shape[-2:]) != (4, 4):
        bottom = lucid.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=lucid.float32).expand(n, 1, 4)  # fmt: skip
        ext = lucid.cat([ext[:, :3, :4], bottom], dim=1)
    index = lucid.tensor([min(max(i, 0), n - 1) for i in range(frames)], dtype=lucid.int64)  # fmt: skip
    ext = ext.index_select(0, index)
    if int_key is None:
        return ext.unsqueeze(0), neutral_cameras(frames, height, width)[1]
    ks = arrays[int_key].to(lucid.float32)
    if ks.ndim == 2:
        ks = ks.unsqueeze(0)
    m = ks.shape[0]
    index = lucid.tensor([min(max(i, 0), m - 1) for i in range(frames)], dtype=lucid.int64)  # fmt: skip
    ks = ks.index_select(0, index)
    src_w, src_h = _parse_resolution(source_resolution)
    if src_w and src_h:
        sx, sy = float(width) / float(src_w), float(height) / float(src_h)
        scale = lucid.tensor([[sx, 1.0, sx], [1.0, sy, sy], [1.0, 1.0, 1.0]], dtype=lucid.float32)  # fmt: skip
        ks = ks * scale
    return ext.unsqueeze(0), ks.unsqueeze(0)


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def build_pixel_cameras(control: dict, *, frames: int, height: int, width: int) -> tuple[Tensor, Tensor]:  # fmt: skip
    """``build_pixel_cameras``: a bundled trajectory, a static camera, or the analytic yaw/pitch loop."""
    trajectory = control.get("_trajectory_path")
    if trajectory:
        return load_trajectory(trajectory, frames, height, width, control.get("source_resolution"))  # fmt: skip
    ext, ks = neutral_cameras(frames, height, width)
    if str(control.get("kind", "clean")) != "camera":
        return ext, ks
    axis = str(control.get("axis", "yaw"))
    if axis not in {"yaw", "pitch"}:
        raise ValueError("camera axis must be 'yaw' or 'pitch'")
    target = math.radians(float(control.get("degrees", 20.0)))
    returns = bool(control.get("return", True))
    peak = float(control.get("peak_fraction", 0.50))
    back = float(control.get("return_fraction", 0.86))
    if not 0.0 < peak < back <= 1.0:
        raise ValueError("camera fractions must satisfy 0 < peak < return <= 1")
    mats = []
    for index in range(frames):
        phase = index / max(1, frames - 1)
        if returns:
            if phase <= peak:
                scale = _smoothstep(phase / peak)
            elif phase <= back:
                scale = 1.0 - _smoothstep((phase - peak) / (back - peak))
            else:
                scale = 0.0
        else:
            scale = _smoothstep(phase / back) if phase <= back else 1.0
        angle = target * scale
        c, s = math.cos(angle), math.sin(angle)
        if axis == "yaw":
            rot = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
        else:
            rot = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
        mats.append([r + [0.0] for r in rot] + [[0.0, 0.0, 0.0, 1.0]])
    return lucid.tensor([mats], dtype=lucid.float32), ks


def pixel_to_latent(ext: Tensor, ks: Tensor, latent_frames: int) -> tuple[Tensor, Tensor]:  # fmt: skip
    """``pixel_to_latent_cameras``: ``linspace(0, F-1, latent_frames).round()`` (float32, half to even)."""
    from remind_lucid.scheduler import _linspace_f32

    picks = [round(v) for v in _linspace_f32(0.0, float(ext.shape[1] - 1), latent_frames)]  # fmt: skip
    index = lucid.tensor(picks, dtype=lucid.int64)
    return ext.index_select(1, index), ks.index_select(1, index)


def camera_metrics(ext: Tensor) -> dict[str, float]:
    """``camera_metrics``: rotation angle and translation relative to frame 0, in float64."""
    rot = ext[0, :, :3, :3].double()
    rel = lucid.matmul(rot, rot[0].swapaxes(0, 1))
    trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    angles = [math.degrees(math.acos(max(-1.0, min(1.0, (t - 1.0) / 2.0)))) for t in trace.tolist()]  # fmt: skip
    pos = ext[0, :, :3, 3].double()
    dist = ((pos - pos[0]).square().sum(dim=-1).sqrt()).tolist()
    return {
        "max_angle_deg": float(max(angles)),
        "final_angle_deg": float(angles[-1]),
        "max_translation": float(max(dist)),
        "final_translation": float(dist[-1]),
    }


@dataclass
class CameraChunk:
    """What one model call needs from the cameras of its frames (all float32, CPU).

    Attributes
    ----------
    pose : Tensor
        ``(1, F, 14)``: translation, flattened rotation, log focal lengths
        (``_camera_pose_features``), the input of each block's phase MLP.
    P, P_inv : Tensor
        ``(1, F, 4, 4)``: the output-side and value-side projective matrices.
    """

    pose: Tensor
    P: Tensor
    P_inv: Tensor


def prepare_chunk(viewmats: Tensor, ks: Tensor, image_width: float) -> CameraChunk:
    """The normalisation of ``CausalWanModel.forward`` and ``_prepare_apply_fns_all_dim``.

    Parameters
    ----------
    viewmats : Tensor
        ``(1, F, 4, 4)`` camera-to-world matrices of the frames of this call.
    ks : Tensor
        ``(1, F, 3, 3)`` pixel intrinsics.
    image_width : float
        ``latent_w * patch * vae_stride`` (832): both intrinsics rows are divided by it.
    """
    viewmats = viewmats.to("cpu").float()
    ks = ks.to("cpu").float()
    f = viewmats.shape[1]
    ks = lucid.cat([ks[..., :2, :] / image_width, ks[..., 2:, :]], dim=-2)
    t_all = viewmats[:, :, :3, 3]
    centre = t_all.mean(dim=1, keepdim=True)
    scale = (t_all - centre).square().sum(dim=-1).sqrt().max(dim=1, keepdim=True)
    scale = scale[0] if isinstance(scale, tuple) else scale
    scale = scale.unsqueeze(-1).clamp(min=0.01)
    t_norm = (t_all - centre) / scale  # (1, F, 3)
    rot = viewmats[:, :, :3, :3]

    fx = ks[..., 0, 0].clamp(min=1e-3)
    fy = ks[..., 1, 1].clamp(min=1e-3)
    pose = lucid.cat([t_norm, rot.reshape(1, f, 9), lucid.stack([fx.log(), fy.log()], dim=-1)], dim=-1)  # fmt: skip

    zeros = lucid.zeros((1, f), dtype=lucid.float32)
    ones = lucid.ones((1, f), dtype=lucid.float32)

    def rows(*r):  # rows of scalars-per-frame -> (1, F, n, n)
        return lucid.stack([lucid.stack(list(row), dim=-1) for row in r], dim=-2)

    kx, ky = ks[..., 0, 0], ks[..., 1, 1]  # Ks_norm keeps only the focal lengths
    lift_k = rows((kx, zeros, zeros, zeros), (zeros, ky, zeros, zeros), (zeros, zeros, ones, zeros), (zeros, zeros, zeros, ones))  # fmt: skip
    # _invert_K of Ks_norm: 1/fx, 1/fy, and -0/fx = -0.0 off the diagonal.
    lift_k_inv = rows((1.0 / kx, zeros, -zeros / kx, zeros), (zeros, 1.0 / ky, -zeros / ky, zeros), (zeros, zeros, ones, zeros), (zeros, zeros, zeros, ones))  # fmt: skip
    bottom = lucid.tensor([0.0, 0.0, 0.0, 1.0], dtype=lucid.float32).reshape(1, 1, 1, 4).expand(1, f, 1, 4)  # fmt: skip
    view = lucid.cat([lucid.cat([rot, t_norm.unsqueeze(-1)], dim=-1), bottom], dim=-2)
    r_inv = rot.swapaxes(-1, -2)
    t_inv = -lucid.matmul(r_inv, t_norm.unsqueeze(-1))
    view_inv = lucid.cat([lucid.cat([r_inv, t_inv], dim=-1), bottom], dim=-2)
    return CameraChunk(pose=pose, P=lucid.matmul(lift_k, view), P_inv=lucid.matmul(view_inv, lift_k_inv))  # fmt: skip
