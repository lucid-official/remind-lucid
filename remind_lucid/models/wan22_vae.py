"""Wan2.2 video VAE (``WanVideoVAE38``: 48-channel latents, x4 time, x16 space), streaming.

A port of ``wan/modules/vae_wan22.py``. Parameter names match
``Wan2.2_VAE.pth``.

It differs from the Wan2.1 VAE in more than width: pixels are patchified
2x2 before the encoder (and unpatchified after the decoder), every stage is a
``Down_ResidualBlock`` / ``Up_ResidualBlock`` whose main path is added to a
parameter-free shortcut (``AvgDown3D`` / ``DupUp3D``), and the resampling
convolutions keep their channel count.

The reference threads a ``feat_cache`` list through every causal 3-D
convolution while it walks a video chunk by chunk (1, 4, 4, ... frames when
encoding; one latent frame at a time when decoding). That bookkeeping is the
same as giving each temporal-kernel-3 convolution a two-frame *history* that
starts as zeros: its input is ``cat(history, x)`` and the new history is the
last two frames of that input. The temporal resamplers are the exceptions —
skipped on the first chunk, then fed one remembered frame (encoder) or a
zero-started history (decoder). :class:`StreamState` holds those histories,
so latent frames can be decoded as the DiT produces them; the reference's own
``decode`` is that same one-latent-frame loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

Tensor = lucid.Tensor

LATENT_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]  # fmt: skip
LATENT_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]  # fmt: skip


@dataclass(frozen=True)
class VAEConfig:
    dim: int = 160
    dec_dim: int = 256
    z_dim: int = 48
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    patch: int = 2


@dataclass
class StreamState:
    """Histories of the causal convolutions of one encoder or decoder stream."""

    hist: dict[int, Tensor] = field(default_factory=dict)
    first: bool = True  # the next call carries the first chunk
    # Evaluate after every causal convolution. Lazily, one 480x832 residual
    # block keeps all its multi-GB intermediates alive at once and spills to
    # swap; evaluating as it goes frees each before the next is allocated.
    sync: bool = True


class CausalConv3d(nn.Conv3d):
    """``nn.Conv3d`` whose temporal front padding is the stream's history.

    The reference subclasses ``nn.Conv3d`` the same way (zero temporal
    padding in the convolution itself, causal padding in ``forward``).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size, padding=0, stride=1) -> None:  # fmt: skip
        k = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
        p = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
        super().__init__(in_ch, out_ch, k, stride=stride, padding=(0, p[1], p[2]))
        self.time_pad = 2 * p[0]
        self.slot = -1

    def stream(self, x: Tensor, state: StreamState | None) -> Tensor:
        out = self._stream(x, state)
        if state is not None and state.sync:
            lucid.eval(out)
        return out

    def _stream(self, x: Tensor, state: StreamState | None) -> Tensor:
        if self.time_pad == 0:
            return self(x)
        hist = None if state is None else state.hist.get(self.slot)
        if hist is None:
            b, c, _, h, w = x.shape
            hist = lucid.zeros((b, c, self.time_pad, h, w), dtype=x.dtype, device=x.device.type)  # fmt: skip
        xin = lucid.cat([hist, x], dim=2)
        if state is not None:
            state.hist[self.slot] = xin[:, :, -self.time_pad :]
        return self(xin)


# nn.RMSNorm was considered: it normalises the *last* axis by the root *mean*
# square with an epsilon inside the root; Wan's RMS_norm L2-normalises the
# channel axis (dim 1, F.normalize's clamp) and scales by sqrt(C) * gamma.
class RMSNorm(nn.Module):
    """Wan ``RMS_norm``: L2-normalise over channels, times ``sqrt(C) * gamma``."""

    def __init__(self, dim: int, images: bool = False) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(lucid.ones(dim, *((1, 1) if images else (1, 1, 1))))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm(in_dim), nn.SiLU(), CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim), nn.SiLU(), nn.Dropout(0.0), CausalConv3d(out_dim, out_dim, 3, padding=1),
        )  # fmt: skip
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()  # fmt: skip

    def stream(self, x: Tensor, state: StreamState | None) -> Tensor:
        h = self.shortcut(x)
        for layer in self.residual:
            x = layer.stream(x, state) if isinstance(layer, CausalConv3d) else layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention over each frame."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def stream(self, x: Tensor, state: StreamState | None = None) -> Tensor:
        identity = x
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        qkv = self.to_qkv(self.norm(x)).reshape(b * t, 1, c * 3, h * w).permute(0, 1, 3, 2)  # fmt: skip
        q, k, v = qkv[..., :c], qkv[..., c : 2 * c], qkv[..., 2 * c :]
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.reshape(b * t, h * w, c).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4) + identity


def _per_frame(x: Tensor, fn) -> Tensor:
    b, c, t, h, w = x.shape
    y = fn(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
    c2, h2, w2 = y.shape[1:]
    return y.reshape(b, t, c2, h2, w2).permute(0, 2, 1, 3, 4)


class Resample(nn.Module):
    """``Resample38``: spatial x2 resampling at constant width, with a causal temporal x2 for ``*3d``."""

    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        if mode.startswith("upsample"):
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="nearest"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            if mode == "upsample3d":
                self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))  # fmt: skip
        else:
            # ZeroPad2d((0, 1, 0, 1)) then a stride-2 conv; slot 0 holds no weights.
            self.resample = nn.Sequential(nn.Identity(), nn.Conv2d(dim, dim, 3, stride=2))  # fmt: skip
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1))

    def _spatial(self, x: Tensor) -> Tensor:
        if self.mode.startswith("upsample"):
            # The reference upsamples in fp32 and casts back; nearest copies values, so the dtype is immaterial.
            return self.resample(x)
        return self.resample[1](F.pad(x, (0, 1, 0, 1)))

    def stream(self, x: Tensor, state: StreamState | None) -> Tensor:
        b, c, t, h, w = x.shape
        first = state is None or state.first
        if self.mode == "upsample3d" and first and state is not None:
            # Skipped on the first frame; its history starts as zeros at the next one.
            state.hist[self.time_conv.slot] = lucid.zeros(
                (b, c, self.time_conv.time_pad, h, w), dtype=x.dtype, device=x.device.type
            )  # fmt: skip
        if self.mode == "upsample3d" and not first:
            x = self.time_conv.stream(x, state)
            x = x.reshape(b, 2, c, t, h, w).permute(0, 2, 3, 1, 4, 5).reshape(b, c, t * 2, h, w)  # fmt: skip
        x = _per_frame(x, self._spatial)
        if self.mode == "downsample3d":
            if first:
                if state is not None:
                    state.hist[self.time_conv.slot] = x[:, :, -1:]
            else:
                prev = state.hist[self.time_conv.slot]
                state.hist[self.time_conv.slot] = x[:, :, -1:]
                x = self.time_conv(lucid.cat([prev, x], dim=2))
        if state is not None and state.sync:
            lucid.eval(x)
        return x


# No Lucid module regroups (time, height, width) factors into channels:
# nn.PixelUnshuffle is 2-D only and has no group mean. New parameter-free modules.
class AvgDown3D(nn.Module):
    """Space/time-to-channel followed by a mean over channel groups (the encoder shortcut)."""

    def __init__(self, in_ch: int, out_ch: int, factor_t: int, factor_s: int) -> None:
        super().__init__()
        self.out_ch, self.factor_t, self.factor_s = out_ch, factor_t, factor_s
        self.group = in_ch * factor_t * factor_s * factor_s // out_ch

    def forward(self, x: Tensor) -> Tensor:
        ft, fs = self.factor_t, self.factor_s
        pad_t = (ft - x.shape[2] % ft) % ft
        if pad_t:
            x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t // ft, ft, h // fs, fs, w // fs, fs)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
        x = x.reshape(b, self.out_ch, self.group, t // ft, h // fs, w // fs)
        return x.mean(dim=2)


class DupUp3D(nn.Module):
    """Channel repetition followed by channel-to-space/time (the decoder shortcut)."""

    def __init__(self, in_ch: int, out_ch: int, factor_t: int, factor_s: int) -> None:
        super().__init__()
        self.out_ch, self.factor_t, self.factor_s = out_ch, factor_t, factor_s
        self.repeats = out_ch * factor_t * factor_s * factor_s // in_ch

    def forward(self, x: Tensor, first_chunk: bool = False) -> Tensor:
        ft, fs = self.factor_t, self.factor_s
        b, _, t, h, w = x.shape
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.reshape(b, self.out_ch, ft, fs, fs, t, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.reshape(b, self.out_ch, t * ft, h * fs, w * fs)
        return x[:, :, ft - 1 :] if first_chunk else x


class DownResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, mult: int, temporal: bool, down: bool) -> None:  # fmt: skip
        super().__init__()
        self.avg_shortcut = AvgDown3D(in_dim, out_dim, 2 if temporal else 1, 2 if down else 1)  # fmt: skip
        layers: list[nn.Module] = []
        for _ in range(mult):
            layers.append(ResidualBlock(in_dim, out_dim))
            in_dim = out_dim
        if down:
            layers.append(Resample(out_dim, "downsample3d" if temporal else "downsample2d"))  # fmt: skip
        self.downsamples = nn.Sequential(*layers)

    def stream(self, x: Tensor, state: StreamState | None) -> Tensor:
        y = x
        for layer in self.downsamples:
            y = layer.stream(y, state)
        return y + self.avg_shortcut(x)


class UpResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, mult: int, temporal: bool, up: bool) -> None:  # fmt: skip
        super().__init__()
        self.temporal = temporal and up
        self.avg_shortcut = DupUp3D(in_dim, out_dim, 2 if temporal else 1, 2) if up else None  # fmt: skip
        layers: list[nn.Module] = []
        for _ in range(mult):
            layers.append(ResidualBlock(in_dim, out_dim))
            in_dim = out_dim
        if up:
            layers.append(Resample(out_dim, "upsample3d" if temporal else "upsample2d"))
        self.upsamples = nn.Sequential(*layers)

    def stream(self, x: Tensor, state: StreamState | None) -> Tensor:
        first = state is None or state.first
        y = x
        for layer in self.upsamples:
            y = layer.stream(y, state)
        if self.avg_shortcut is None:
            return y
        return y + self.avg_shortcut(x, first)


def _run(layers: list[nn.Module], x: Tensor, state: StreamState | None) -> Tensor:
    for layer in layers:
        if isinstance(layer, nn.Sequential):  # head: norm, SiLU, conv
            for sub in layer:
                x = sub.stream(x, state) if isinstance(sub, CausalConv3d) else sub(x)
        else:
            x = layer.stream(x, state)
    return x


class Encoder3d(nn.Module):
    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        dims = [cfg.dim * u for u in [1, *cfg.dim_mult]]
        self.conv1 = CausalConv3d(3 * cfg.patch**2, dims[0], 3, padding=1)
        stages = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            temporal = cfg.temperal_downsample[i] if i < len(cfg.temperal_downsample) else False  # fmt: skip
            stages.append(DownResidualBlock(in_dim, out_dim, cfg.num_res_blocks, temporal, i != len(cfg.dim_mult) - 1))  # fmt: skip
        self.downsamples = nn.Sequential(*stages)
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim))  # fmt: skip
        self.head = nn.Sequential(RMSNorm(out_dim), nn.SiLU(), CausalConv3d(out_dim, cfg.z_dim * 2, 3, padding=1))  # fmt: skip

    def layers(self) -> list[nn.Module]:
        return [self.conv1, *self.downsamples, *self.middle, self.head]


class Decoder3d(nn.Module):
    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        mult = list(cfg.dim_mult)
        up_t = list(cfg.temperal_downsample)[::-1]
        dims = [cfg.dec_dim * u for u in [mult[-1], *mult[::-1]]]
        self.conv1 = CausalConv3d(cfg.z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0]), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0]))  # fmt: skip
        stages = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            temporal = up_t[i] if i < len(up_t) else False
            stages.append(UpResidualBlock(in_dim, out_dim, cfg.num_res_blocks + 1, temporal, i != len(mult) - 1))  # fmt: skip
        self.upsamples = nn.Sequential(*stages)
        self.head = nn.Sequential(RMSNorm(out_dim), nn.SiLU(), CausalConv3d(out_dim, 3 * cfg.patch**2, 3, padding=1))  # fmt: skip

    def layers(self) -> list[nn.Module]:
        return [self.conv1, *self.middle, *self.upsamples, self.head]

    def frame_major_from(self) -> int:
        """Index in :meth:`layers` just past the last temporal upsampler."""
        last = max(i for i, m in enumerate(self.layers()) if getattr(m, "temporal", False))  # fmt: skip
        return last + 1


def patchify(x: Tensor, p: int) -> Tensor:
    """``b c f (h q) (w r) -> b (c r q) f h w``."""
    b, c, f, h, w = x.shape
    x = x.reshape(b, c, f, h // p, p, w // p, p).permute(0, 1, 6, 4, 2, 3, 5)
    return x.reshape(b, c * p * p, f, h // p, w // p)


def unpatchify(x: Tensor, p: int) -> Tensor:
    """``b (c r q) f h w -> b c f (h q) (w r)``."""
    b, cp, f, h, w = x.shape
    c = cp // (p * p)
    x = x.reshape(b, c, p, p, f, h, w).permute(0, 1, 4, 5, 3, 6, 2)
    return x.reshape(b, c, f, h * p, w * p)


class WanVAE(nn.Module):
    """Encoder, decoder and the latent normalisation of ``WanVAEWrapper``.

    Parameters
    ----------
    cfg : VAEConfig
    scale_dtype
        The dtype ``mean`` and ``1 / std`` are built in. The reference builds
        them in the dtype of the tensor being (de)normalised, which on its
        inference path is always bf16 — whatever dtype the convolutions run in
        here, the constants stay the reference's.
    """

    def __init__(self, cfg: VAEConfig = VAEConfig(), scale_dtype=lucid.bfloat16) -> None:  # fmt: skip
        super().__init__()
        self.cfg = cfg
        self.scale_dtype = scale_dtype
        self.encoder = Encoder3d(cfg)
        self.conv1 = CausalConv3d(cfg.z_dim * 2, cfg.z_dim * 2, 1)
        self.conv2 = CausalConv3d(cfg.z_dim, cfg.z_dim, 1)
        self.decoder = Decoder3d(cfg)
        for part in (self.encoder, self.decoder):
            slot = 0
            for m in part.modules():
                if isinstance(m, CausalConv3d) and (m.time_pad or m.stride[0] == 2):
                    m.slot = slot
                    slot += 1

    @property
    def dtype(self):
        return self.conv2.weight.dtype

    def _scale(self, like: Tensor) -> tuple[Tensor, Tensor]:
        mean = lucid.tensor(LATENT_MEAN, dtype=lucid.float32, device="cpu").to(
            self.scale_dtype
        )
        inv_std = 1.0 / lucid.tensor(LATENT_STD, dtype=lucid.float32, device="cpu").to(self.scale_dtype)  # fmt: skip
        dev, dt = like.device.type, like.dtype
        return (
            mean.to(dt).to(dev).reshape(1, -1, 1, 1, 1),
            inv_std.to(dt).to(dev).reshape(1, -1, 1, 1, 1),
        )

    # -- encode ------------------------------------------------------------ #

    def encode(self, video: Tensor) -> Tensor:
        """Pixels ``(B, 3, T, H, W)`` in ``[-1, 1]`` -> normalised latents ``(B, 48, 1+(T-1)//4, H/16, W/16)``.

        Chunks of 1, 4, 4, ... frames, as ``VideoVAE38_.encode``. The encoder
        is causal, so the first ``n`` latent frames depend only on the first
        ``1 + 4(n-1)`` pixel frames; callers that need a prefix pass only
        those frames.
        """
        state = StreamState()
        x = patchify(video.to(self.dtype), self.cfg.patch)
        t = x.shape[2]
        outs = []
        for i in range(1 + (t - 1) // 4):
            chunk = x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i]
            outs.append(_run(self.encoder.layers(), chunk, state))
            state.first = False
        out = outs[0] if len(outs) == 1 else lucid.cat(outs, dim=2)
        mu = self.conv1(out)[:, : self.cfg.z_dim]
        mean, inv_std = self._scale(mu)
        return (mu - mean) * inv_std

    # -- decode ------------------------------------------------------------ #

    def new_state(self) -> StreamState:
        return StreamState()

    def prepare(self, z: Tensor) -> Tensor:
        """De-normalise latents ``(B, 48, T, h, w)`` and apply ``conv2``."""
        mean, inv_std = self._scale(z)
        return self.conv2((z / inv_std + mean).to(self.dtype))

    def decode_stream(self, z: Tensor, state: StreamState) -> Tensor:
        """Normalised latent frames ``(B, 48, T, h, w)`` -> pixels ``(B, 3, T', H, W)`` in ``[-1, 1]``.

        Continues ``state``; returns ``1 + 4(T-1)`` frames on the first call
        and ``4T`` after. One latent frame per decoder pass, as the reference.
        """
        x = self.prepare(z)
        layers = self.decoder.layers()
        split = self.decoder.frame_major_from()
        outs = []
        for i in range(x.shape[2]):
            frame = x[:, :, i : i + 1]
            if state.first:
                outs.append(_run(layers, frame, state))
                state.first = False
                continue
            # Up to the last temporal upsampler one latent frame becomes four
            # video frames; past it every layer is causal and per-frame in
            # time, so the high-resolution stages take the four frames one at
            # a time (same result, a quarter of the peak memory).
            y = _run(layers[:split], frame, state)
            for t in range(y.shape[2]):
                outs.append(_run(layers[split:], y[:, :, t : t + 1], state))
        out = outs[0] if len(outs) == 1 else lucid.cat(outs, dim=2)
        return unpatchify(out, self.cfg.patch).float().clamp(-1.0, 1.0)

    def decode(self, z: Tensor) -> Tensor:
        """``VideoVAE38_.decode``: every latent frame, from a fresh state."""
        return self.decode_stream(z, self.new_state())
