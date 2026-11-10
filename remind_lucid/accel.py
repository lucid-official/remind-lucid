"""``lucid.compile`` acceleration.

**Streaming VAE decoder.** Decoding one latent frame is a pure function of
the frame and the two-frame histories of the decoder's causal convolutions.
:class:`CompiledDecoder` traces that function twice — the first frame (no
temporal upsampling, zero histories) and the steady state — so MPSGraph
fuses the norm/SiLU/conv chains that dominate the 480x832 tail. Shapes never
change, so each program compiles once.

Whether a compiled program is used anywhere is decided by measurement; the
numbers are in the README's speed section.
"""

from __future__ import annotations

import os

os.environ.setdefault("LUCID_COMPILE_MAX_CACHE", "256")

import lucid  # noqa: E402
import lucid.nn as nn  # noqa: E402

from remind_lucid.models.wan22_vae import CausalConv3d, StreamState, WanVAE, _run, unpatchify  # noqa: E402

Tensor = lucid.Tensor


def _history_slots(vae: WanVAE) -> list[CausalConv3d]:
    return [m for m in vae.decoder.modules() if isinstance(m, CausalConv3d) and m.time_pad]


class _DecodeFirst(nn.Module):
    """First latent frame: zero histories, temporal upsamplers skipped. Returns pixels and histories."""

    def __init__(self, vae: WanVAE) -> None:
        super().__init__()
        self.decoder = vae.decoder
        self.slots = [m.slot for m in _history_slots(vae)]

    def forward(self, x: Tensor):
        state = StreamState(sync=False)
        out = _run(self.decoder.layers(), x, state)
        return (out, *[state.hist[s] for s in self.slots])


class _DecodeStep(nn.Module):
    """A later latent frame: histories in, pixels and new histories out."""

    def __init__(self, vae: WanVAE) -> None:
        super().__init__()
        self.decoder = vae.decoder
        self.slots = [m.slot for m in _history_slots(vae)]

    def forward(self, x: Tensor, *hists: Tensor):
        state = StreamState(hist=dict(zip(self.slots, hists)), first=False, sync=False)
        out = _run(self.decoder.layers(), x, state)
        return (out, *[state.hist[s] for s in self.slots])


class CompiledDecoder:
    """Drop-in for :class:`remind_lucid.pipeline.StreamingDecoder`, through two compiled programs."""

    def __init__(self, vae: WanVAE) -> None:
        self.vae = vae
        self.first = lucid.compile(_DecodeFirst(vae).eval())
        self.step = lucid.compile(_DecodeStep(vae).eval())
        self.hists: list[Tensor] | None = None

    def reset(self) -> None:
        self.hists = None

    def __call__(self, latents: Tensor) -> Tensor:
        """``(1, T, 48, h, w)`` latents -> ``(3, T', H, W)`` pixels in ``[-1, 1]``."""
        vae = self.vae
        with lucid.no_grad():
            x = vae.prepare(latents.permute(0, 2, 1, 3, 4))
            outs = []
            for i in range(x.shape[2]):
                frame = x[:, :, i : i + 1]
                if self.hists is None:
                    out, *self.hists = self.first(frame)
                else:
                    out, *self.hists = self.step(frame, *self.hists)
                lucid.eval(out, *self.hists)
                outs.append(out)
            pixels = outs[0] if len(outs) == 1 else lucid.cat(outs, dim=2)
            pixels = unpatchify(pixels, vae.cfg.patch).float().clamp(-1.0, 1.0)
            lucid.eval(pixels)
        return pixels[0]
