"""UMT5-XXL encoder: a port of ``wan/modules/t5.py`` (``umt5_xxl(encoder_only=True)``).

24 pre-norm blocks, 64 heads of 64, gated-GELU feed-forward, unscaled
attention and a relative position bias per block (``shared_pos=False``).
Parameter names match ``models_t5_umt5-xxl-enc-bf16.pth``.

The reference runs it in bf16 and is explicit about which steps leave bf16:
the norm's ``rsqrt(mean(x^2))`` and the attention softmax are computed in
fp32. Both are kept, so the rounding points are the reference's.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

Tensor = lucid.Tensor


@dataclass(frozen=True)
class UMT5Config:
    vocab_size: int = 256384
    dim: int = 4096
    dim_attn: int = 4096
    dim_ffn: int = 10240
    num_heads: int = 64
    num_layers: int = 24
    num_buckets: int = 32
    max_dist: int = 128


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def relative_position_buckets(length: int, num_buckets: int = 32, max_dist: int = 128) -> list[list[int]]:  # fmt: skip
    """``T5RelativeEmbedding._relative_position_bucket`` (bidirectional) for ``k - q``.

    The reference takes the logarithm in float32 and truncates; a bucket edge
    is one ulp away from flipping, so the float32 steps are reproduced with
    the standard library instead of trusting another libm.
    """
    half = num_buckets // 2
    max_exact = half // 2
    denom = math.log(max_dist / max_exact)
    by_distance = []
    for dist in range(length):
        if dist < max_exact:
            by_distance.append(dist)
            continue
        ratio = _f32(float(dist) / max_exact)
        value = _f32(_f32(_f32(math.log(ratio)) / _f32(denom)) * (half - max_exact))
        by_distance.append(min(max_exact + int(value), half - 1))
    return [
        [(half if k > q else 0) + by_distance[abs(k - q)] for k in range(length)]
        for q in range(length)
    ]


class T5LayerNorm(nn.RMSNorm):
    """``nn.RMSNorm`` with the reference's precision: the scale in fp32, the result in the weight's dtype."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__(dim, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        scale = lucid.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x.float() * scale).to(self.weight.dtype)


class T5RelativeEmbedding(nn.Module):
    def __init__(self, num_buckets: int, num_heads: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, buckets: Tensor) -> Tensor:
        """``(Lq, Lk)`` bucket ids -> ``(1, heads, Lq, Lk)``."""
        return self.embedding(buckets).permute(2, 0, 1).unsqueeze(0)


# nn.MultiheadAttention was considered: it packs q/k/v into one in_proj
# weight, scales by 1/sqrt(d) and has no additive per-head bias input, so the
# checkpoint's separate bias-free q/k/v/o and T5's unscaled attention do not fit.
class T5Attention(nn.Module):
    def __init__(self, cfg: UMT5Config) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.dim_attn // cfg.num_heads
        self.q = nn.Linear(cfg.dim, cfg.dim_attn, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.dim_attn, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.dim_attn, bias=False)
        self.o = nn.Linear(cfg.dim_attn, cfg.dim, bias=False)

    def forward(self, x: Tensor, bias: Tensor) -> Tensor:
        b, l, _ = x.shape
        n, c = self.num_heads, self.head_dim
        q = self.q(x).reshape(b, l, n, c).permute(0, 2, 1, 3)
        k = self.k(x).reshape(b, l, n, c).permute(0, 2, 3, 1)
        v = self.v(x).reshape(b, l, n, c).permute(0, 2, 1, 3)
        attn = lucid.matmul(q, k) + bias  # T5 does not scale
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        out = lucid.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, l, n * c)
        return self.o(out)


class T5FeedForward(nn.Module):
    def __init__(self, cfg: UMT5Config) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(cfg.dim, cfg.dim_ffn, bias=False), nn.GELU(approximate="tanh"))  # fmt: skip
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ffn, bias=False)
        self.fc2 = nn.Linear(cfg.dim_ffn, cfg.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x) * self.gate(x))


class T5SelfAttention(nn.Module):
    def __init__(self, cfg: UMT5Config) -> None:
        super().__init__()
        self.norm1 = T5LayerNorm(cfg.dim)
        self.attn = T5Attention(cfg)
        self.norm2 = T5LayerNorm(cfg.dim)
        self.ffn = T5FeedForward(cfg)
        self.pos_embedding = T5RelativeEmbedding(cfg.num_buckets, cfg.num_heads)

    def forward(self, x: Tensor, buckets: Tensor, key_mask: Tensor) -> Tensor:
        bias = self.pos_embedding(buckets).to(x.dtype)
        bias = lucid.where(key_mask, bias, float(lucid.finfo(x.dtype).min))
        x = x + self.attn(self.norm1(x), bias)
        return x + self.ffn(self.norm2(x))


class UMT5Encoder(nn.Module):
    def __init__(self, cfg: UMT5Config = UMT5Config()) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([T5SelfAttention(cfg) for _ in range(cfg.num_layers)])  # fmt: skip
        self.norm = T5LayerNorm(cfg.dim)

    @property
    def dtype(self):
        return self.norm.weight.dtype

    def forward(self, ids: Tensor, mask: Tensor, num_layers: int | None = None) -> Tensor:  # fmt: skip
        """``ids`` / ``mask``: ``(B, L)`` integers. Returns ``(B, L, dim)``, final norm applied.

        ``num_layers`` stops after that many blocks (and skips the final
        norm); it exists for layer-by-layer parity tests.
        """
        device = self.norm.weight.device.type
        ids, mask = ids.to(device), mask.to(device)
        b, l = ids.shape
        x = self.token_embedding(ids)
        buckets = lucid.tensor(
            relative_position_buckets(l, self.cfg.num_buckets, self.cfg.max_dist),
            dtype=lucid.int64,
            device=device,
        )
        key_mask = (mask != 0).reshape(b, 1, 1, l)
        for block in self.blocks[:num_layers]:
            x = block(x, buckets, key_mask)
            lucid.eval(x)
        return x if num_layers is not None else self.norm(x)
