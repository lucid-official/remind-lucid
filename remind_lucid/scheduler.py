"""The shifted flow-matching table of the ReMind DMD student.

``utils.scheduler.FlowMatchScheduler(shift=5.0, sigma_min=0.0,
extra_one_step=True)`` after ``set_timesteps(1000, training=True)``: the
1000-entry table every timestep is looked up in (``argmin |timesteps - t|``).
The table is computed with the same float32 operations as the reference
(``torch.linspace`` then the shift), one rounding per operation, so lookups
land on the same entries.

The two places the reference leaves the latent dtype are kept:
``add_noise`` mixes in fp32 and casts to the noise's dtype;
``_convert_flow_pred_to_x0`` computes ``xt - sigma_t * flow`` in float64,
which Metal does not have — it runs on the CPU (a chunk is 224,640 values).
"""

from __future__ import annotations

import math
import struct

import lucid

Tensor = lucid.Tensor


def _f32(v: float) -> float:
    """Round a Python float to the nearest float32 (correct rounding for one +, -, * or /)."""
    return struct.unpack("f", struct.pack("f", v))[0]


def _linspace_f32(start: float, end: float, steps: int) -> list[float]:
    """``torch.linspace`` on the CPU in float32, bit for bit.

    The first half counts up from ``start`` and the second down from ``end``;
    each element is one fused multiply-add (``start + step * i`` rounded
    once), which the product being exact in double precision reproduces.
    """
    step = _f32((_f32(end) - _f32(start)) / (steps - 1))
    half = steps // 2
    return [
        _f32(start + step * i) if i < half else _f32(end - step * (steps - 1 - i))
        for i in range(steps)
    ]


class FlowMatchSchedule:
    """``sigma_i`` and ``timestep_i = 1000 sigma_i`` of the shifted rectified flow."""

    def __init__(self, shift: float = 5.0, num_train_timesteps: int = 1000) -> None:
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps
        f, sh = _f32, _f32(shift)
        lin = _linspace_f32(1.0, 0.0, num_train_timesteps + 1)[:-1]
        # shift * s / (1 + (shift - 1) * s), one float32 rounding per operation.
        self._s = [f(f(sh * s) / f(1.0 + f(f(sh - 1.0) * s))) for s in lin]
        self._t = [f(s * num_train_timesteps) for s in self._s]
        self.sigmas = lucid.tensor(self._s, dtype=lucid.float32)
        self.timesteps = lucid.tensor(self._t, dtype=lucid.float32)

    def index(self, timestep: float) -> int:
        """``argmin |timesteps - t|`` (first index on ties, as ``torch.argmin``)."""
        best, best_d = 0, math.inf
        for i, v in enumerate(self._t):
            d = abs(v - timestep)
            if d < best_d:
                best, best_d = i, d
        return best

    def sigma(self, timestep: float) -> float:
        return self._s[self.index(timestep)]

    def _per_frame(self, timesteps, like: Tensor, dtype) -> Tensor:
        """Per-frame sigmas shaped to broadcast over ``(1, F, C, H, W)``."""
        values = [self.sigma(float(t)) for t in timesteps]
        return lucid.tensor(values, dtype=dtype, device=like.device.type).reshape(1, -1, 1, 1, 1)  # fmt: skip

    def add_noise(self, x: Tensor, noise: Tensor, timesteps) -> Tensor:
        """``(1 - sigma) x + sigma noise`` per frame, in the dtype of ``noise`` (``type_as(noise)``)."""
        s = self._per_frame(timesteps, x, lucid.float32)
        return ((1.0 - s) * x.float() + s * noise.float()).to(noise.dtype)

    def x0_from_flow(self, flow: Tensor, xt: Tensor, timesteps) -> Tensor:
        """``_convert_flow_pred_to_x0``: ``xt - sigma_t * flow`` in float64 on the CPU, cast back."""
        flow_c, xt_c = flow.to("cpu"), xt.to("cpu")
        s = self._per_frame(timesteps, flow_c, lucid.float64)
        x0 = xt_c.double() - s * flow_c.double()
        return x0.to(flow.dtype).to(flow.device.type)
