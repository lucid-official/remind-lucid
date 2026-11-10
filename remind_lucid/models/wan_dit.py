"""The ReMind-5B generator: a chunk-causal Wan2.2-TI2V-5B with PM-RoPE (``cc_full``).

A port of the KV-cache inference path of ``wan/modules/causal_model.py``
(``CausalWanModel``), ``prope/camera_rope.py`` and the pieces of
``wan/modules/model.py`` it uses. Parameter names match
``ReMind-5B.safetensors``; the training paths (flex-attention block masks,
sequence parallelism, clean-history teacher forcing) are not ported.

One call processes the frames of one chunk. Each of the 30 blocks does:

* self-attention over ``[cached history | this chunk]``. Queries and keys
  are rotated by the 3-D RoPE of their absolute (time, row, column) position,
  multiplied — in every one of the 64 frequency slots — by a per-frame camera
  phasor ``exp(i * camera_phase_mlp(pose))``; keys are cached *after* the
  rotation, so history is never re-rotated;
* a value-side residual ``V += value_proj(P^-1 V)`` before attention and an
  output-side residual ``o(x) + prope_proj(P x)`` after it, where ``P`` is the
  frame's projective camera matrix tiled over the head dimension in 4x4
  blocks;
* cross-attention to the text context of the chunk the frames belong to;
* the feed-forward, with per-frame timestep modulation throughout.

Precision follows the reference, which runs in bf16 and says where it does
not: RoPE and the camera matrices are applied in fp32, the query/key RMS norm
is computed in fp32, and the sinusoid and RoPE tables are built in float64.
Metal has no float64, so those tables are built on Lucid's CPU backend and
moved over as fp32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from remind_lucid.camera import CameraChunk

Tensor = lucid.Tensor


@dataclass(frozen=True)
class DiTConfig:
    """``Wan2.2-TI2V-5B/config.json`` + the ``generator`` section of ``configs/model_5b.yaml``."""

    in_dim: int = 48
    out_dim: int = 48
    dim: int = 3072
    ffn_dim: int = 14336
    freq_dim: int = 256
    text_dim: int = 4096
    text_len: int = 512
    num_heads: int = 24
    num_layers: int = 30
    patch_size: tuple[int, int, int] = (1, 2, 2)
    eps: float = 1e-6
    pose_dim: int = 14
    vae_stride: int = 16

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


# --------------------------------------------------------------------------- #
# Float64 tables (CPU)
# --------------------------------------------------------------------------- #


def sinusoidal_embedding(dim: int, positions: list[float]) -> Tensor:
    """``sinusoidal_embedding_1d``: ``[cos, sin]`` of ``pos * 10000^(-i / half)`` in float64 (CPU)."""
    half = dim // 2
    pos = lucid.tensor(positions, dtype=lucid.float64, device="cpu")
    inv = lucid.pow(10000.0, -lucid.arange(half, dtype=lucid.float64, device="cpu") / half)
    angle = pos.reshape(-1, 1) * inv.reshape(1, -1)
    return lucid.cat([lucid.cos(angle), lucid.sin(angle)], dim=1)


def rope_angles(max_len: int, dim: int, theta: float = 10000.0) -> Tensor:
    """The angles of ``rope_params``: ``pos / theta^(2i / dim)`` in float64, ``(max_len, dim / 2)``."""
    inv = 1.0 / lucid.pow(theta, lucid.arange(0, dim, 2, dtype=lucid.float64, device="cpu") / dim)
    return lucid.arange(max_len, dtype=lucid.float64, device="cpu").reshape(-1, 1) * inv.reshape(1, -1)  # fmt: skip


@dataclass
class ChunkGeometry:
    """Everything positional one chunk needs, fixed across its denoising steps and its cache pass.

    Attributes
    ----------
    cos, sin : list[Tensor]
        Per block, ``(L, head_dim)`` fp32 tables of the camera-modulated RoPE,
        each frequency repeated twice (the interleaved layout of
        :func:`lucid.nn.functional.apply_rotary_emb`).
    P_t, P_inv_t : Tensor
        ``(1, 1, F, 1, 4, 4)`` fp32 transposes of the projective matrices.
    frames : int
    """

    cos: list[Tensor]
    sin: list[Tensor]
    P_t: Tensor
    P_inv_t: Tensor
    frames: int


# --------------------------------------------------------------------------- #
# KV cache
# --------------------------------------------------------------------------- #


@dataclass
class KVCache:
    """Rotated keys and values of the committed chunks, per block.

    The reference preallocates ``(1, 21 * 390, heads, head_dim)`` buffers and
    writes token ranges in place. Here each committed chunk is one entry, kept
    as a contiguous copy (a view would pin the whole graph of the pass that
    produced it); a read concatenates the visible entries in key order, which
    is the reference's compact layout. Temporal positions live in the keys'
    rotation, not in the slot order, so entries may be dropped, replaced or
    left with gaps in time — what the research probes do.
    """

    num_blocks: int
    entries: list[dict[int, tuple[Tensor, Tensor]]] = field(default_factory=list)
    hidden: set[int] = field(default_factory=set)
    _joined: list[tuple[Tensor, Tensor] | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.entries = [dict() for _ in range(self.num_blocks)]
        self._joined = [None] * self.num_blocks

    def write(self, block: int, key: int, k: Tensor, v: Tensor) -> None:
        k, v = lucid.contiguous(k), lucid.contiguous(v)
        lucid.eval(k, v)
        self.entries[block][key] = (k, v)
        self._joined[block] = None

    def visible(self) -> list[int]:
        return sorted(k for k in self.entries[0] if k not in self.hidden)

    def hide(self, keys) -> None:
        self.hidden = set(keys)
        self._joined = [None] * self.num_blocks

    def drop(self, key: int) -> None:
        for block in self.entries:
            block.pop(key, None)
        self._joined = [None] * self.num_blocks

    def read(self, block: int) -> tuple[Tensor, Tensor] | None:
        keys = [k for k in sorted(self.entries[block]) if k not in self.hidden]
        if not keys:
            return None
        if self._joined[block] is None:
            ks = [self.entries[block][k][0] for k in keys]
            vs = [self.entries[block][k][1] for k in keys]
            joined = (ks[0], vs[0]) if len(keys) == 1 else (lucid.cat(ks, dim=2), lucid.cat(vs, dim=2))  # fmt: skip
            lucid.eval(*joined)
            self._joined[block] = joined
        return self._joined[block]

    def tokens(self) -> int:
        return sum(self.entries[0][k][0].shape[2] for k in self.visible()) if self.entries else 0  # fmt: skip


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #


class WanRMSNorm(nn.RMSNorm):
    """``nn.RMSNorm`` with the reference's precision: normalised in fp32, cast back, then scaled."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__(dim, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        normed = xf * lucid.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


class WanLayerNorm(nn.LayerNorm):
    """``nn.LayerNorm`` over the last axis in the input's dtype (the reference's ``type_as``)."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False) -> None:  # fmt: skip
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x).to(x.dtype)


class LoRALinear(nn.Linear):
    """``nn.Linear`` that can absorb a low-rank update: ``W += (B @ A) * alpha / r``.

    Mirrors ``peft``'s ``merge_and_unload`` for a model already in half
    precision: ``A`` and ``B`` are cast to the weight's dtype first, their
    product is taken in fp32 and rounded to that dtype, then added to ``W``
    in that dtype.
    """

    def merge_lora(self, a: Tensor, b: Tensor, alpha: float, rank: int) -> None:
        dt, dev = self.weight.dtype, self.weight.device.type
        a = a.to(dev).to(dt).float()
        b = b.to(dev).to(dt).float()
        delta = (lucid.matmul(b, a).to(dt) * (alpha / rank)).to(dt)
        merged = self.weight + delta
        lucid.eval(merged)
        self.weight = nn.Parameter(merged, requires_grad=False)


class CameraPhaseMLP(nn.Module):
    """One bias-free linear layer: pose ``(.., 14)`` -> a phase per RoPE frequency ``(.., 64)``."""

    def __init__(self, pose_dim: int, head_dim_half: int) -> None:
        super().__init__()
        self.proj = nn.Linear(pose_dim, head_dim_half, bias=False)


def _tiled_projection(x: Tensor, matrix_t: Tensor, frames: int) -> Tensor:
    """``_apply_tiled_projmat``: apply each frame's 4x4 matrix to every 4-vector of its tokens, in fp32.

    ``x`` is ``(B, heads, L, d)``; ``matrix_t`` is ``(1, 1, F, 1, 4, 4)``, the
    transpose, so that ``feats @ matrix_t`` is ``matrix @ feat`` per 4-vector.
    """
    b, n, l, d = x.shape
    feats = x.float().reshape(b, n, frames, l // frames, d // 4, 4)
    return lucid.matmul(feats, matrix_t).reshape(b, n, l, d).to(x.dtype)


# nn.MultiheadAttention was considered for both attention classes: its packed
# in_proj weight, absence of q/k RMS norms and fixed q.k^T path leave no room
# for the checkpoint's separate q/k/v/o, the rotated cached keys or the
# projective residuals. Fresh modules; the kernel is Lucid's fused SDPA.
class CausalSelfAttention(nn.Module):
    """``CausalWanSelfAttention`` in ``cc_full`` mode, KV-cache path."""

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.num_heads, self.head_dim = cfg.num_heads, cfg.head_dim
        dim = cfg.dim
        self.q = LoRALinear(dim, dim)
        self.k = LoRALinear(dim, dim)
        self.v = LoRALinear(dim, dim)
        self.o = LoRALinear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=cfg.eps)
        self.norm_k = WanRMSNorm(dim, eps=cfg.eps)
        self.prope_proj = LoRALinear(dim, dim)
        self.value_proj = LoRALinear(dim, dim)
        self.camera_phase_mlp = CameraPhaseMLP(cfg.pose_dim, cfg.head_dim // 2)
        self.probe = None  # research hook (remind_lucid.probes); None = the reference path

    def qkv(self, x: Tensor, cos: Tensor, sin: Tensor, p_inv_t: Tensor, frames: int):
        """Rotated queries and keys and the camera-augmented values, each ``(B, heads, L, d)``."""
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).reshape(b, s, n, d).permute(0, 2, 1, 3)
        k = self.norm_k(self.k(x)).reshape(b, s, n, d).permute(0, 2, 1, 3)
        v = self.v(x)
        dtype = v.dtype
        q, k = F.apply_rotary_emb(q.float(), k.float(), cos, sin, interleaved=True)
        q, k = q.to(dtype), k.to(dtype)
        # V <- V + value_proj(P^-1 V)
        v_hnd = v.reshape(b, s, n, d).permute(0, 2, 1, 3)
        v_p = _tiled_projection(v_hnd, p_inv_t, frames)
        v = v + self.value_proj(v_p.permute(0, 2, 1, 3).reshape(b, s, n * d))
        v = v.reshape(b, s, n, d).permute(0, 2, 1, 3)
        return q, k, v

    def attend(self, q: Tensor, k_full: Tensor, v_full: Tensor, p_t: Tensor, frames: int, block: int = -1) -> Tensor:  # fmt: skip
        """Attention over ``[history | chunk]`` and the output projections: ``o(x) + prope_proj(P x)``."""
        b, _, s, _ = q.shape
        if self.probe is not None:
            out = self.probe(block, q, k_full, v_full, s)
        else:
            out = F.scaled_dot_product_attention(q, k_full, v_full)
        flat = out.permute(0, 2, 1, 3).reshape(b, s, -1)
        out_p = _tiled_projection(out, p_t, frames)
        return self.o(flat) + self.prope_proj(out_p.permute(0, 2, 1, 3).reshape(b, s, -1))  # fmt: skip


class CrossAttention(nn.Module):
    """``WanT2VCrossAttention``: no key mask — all 512 (zero-padded, then embedded) text positions are attended."""

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.num_heads, self.head_dim = cfg.num_heads, cfg.head_dim
        dim = cfg.dim
        self.q = LoRALinear(dim, dim)
        self.k = LoRALinear(dim, dim)
        self.v = LoRALinear(dim, dim)
        self.o = LoRALinear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=cfg.eps)
        self.norm_k = WanRMSNorm(dim, eps=cfg.eps)

    def keys_values(self, context: Tensor) -> tuple[Tensor, Tensor]:
        b, l, _ = context.shape
        n, d = self.num_heads, self.head_dim
        k = self.norm_k(self.k(context)).reshape(b, l, n, d).permute(0, 2, 1, 3)
        v = self.v(context).reshape(b, l, n, d).permute(0, 2, 1, 3)
        return k, v

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).reshape(b, s, n, d).permute(0, 2, 1, 3)
        k, v = self.keys_values(context)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o(out.permute(0, 2, 1, 3).reshape(b, s, n * d))


class AttentionBlock(nn.Module):
    """``CausalWanAttentionBlock``."""

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.norm1 = WanLayerNorm(cfg.dim, cfg.eps)
        self.self_attn = CausalSelfAttention(cfg)
        self.norm3 = WanLayerNorm(cfg.dim, cfg.eps, elementwise_affine=True)
        self.cross_attn = CrossAttention(cfg)
        self.norm2 = WanLayerNorm(cfg.dim, cfg.eps)
        self.ffn = nn.Sequential(LoRALinear(cfg.dim, cfg.ffn_dim), nn.GELU(approximate="tanh"), LoRALinear(cfg.ffn_dim, cfg.dim))  # fmt: skip
        self.modulation = nn.Parameter(lucid.zeros(1, 6, cfg.dim))

    def step(self, x: Tensor, e: Tensor, context: Tensor, cos: Tensor, sin: Tensor, p_t: Tensor, p_inv_t: Tensor, hist_k: Tensor | None = None, hist_v: Tensor | None = None, block: int = -1):  # fmt: skip
        """One block as a pure function of tensors: returns ``(x_out, k, v)``.

        ``x``: ``(B, L, C)``; ``e``: ``(B, F, 6, C)``; ``context``: ``(N, 512,
        C)``, one per group of ``F / N`` frames; ``hist_k`` / ``hist_v``: the
        cached history ``(B, heads, L_hist, d)`` or ``None``. ``k`` and ``v``
        are this chunk's rotated keys and augmented values (what a commit
        pass caches). This is the unit ``remind_lucid.accel`` compiles.
        """
        b, l, d = x.shape
        f = e.shape[1]
        tpf = l // f
        em = self.modulation.unsqueeze(1) + e  # (B, F, 6, C)
        mod = [em[:, :, i : i + 1] for i in range(6)]

        def modulate(h: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
            return (h.reshape(b, f, tpf, d) * (1 + scale) + shift).reshape(b, l, d)

        def residual(h: Tensor, y: Tensor, gate: Tensor) -> Tensor:
            return (h.reshape(b, f, tpf, d) + y.reshape(b, f, tpf, d) * gate).reshape(b, l, d)  # fmt: skip

        attn = self.self_attn
        q, k, v = attn.qkv(modulate(self.norm1(x), mod[1], mod[0]), cos, sin, p_inv_t, f)
        if hist_k is None:
            k_full, v_full = k, v
        else:
            k_full, v_full = lucid.cat([hist_k, k], dim=2), lucid.cat([hist_v, v], dim=2)
        x = residual(x, attn.attend(q, k_full, v_full, p_t, f, block), mod[2])
        n = context.shape[0]
        cross = self.cross_attn(self.norm3(x).reshape(b * n, l // n, d), context)
        x = x + cross.reshape(b, l, d)
        y = self.ffn(modulate(self.norm2(x), mod[4], mod[3]))
        return residual(x, y, mod[5]), k, v


class Head(nn.Module):
    """``CausalHead``."""

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.norm = WanLayerNorm(cfg.dim, cfg.eps)
        self.head = nn.Linear(cfg.dim, math.prod(cfg.patch_size) * cfg.out_dim)
        self.modulation = nn.Parameter(lucid.zeros(1, 2, cfg.dim))

    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        b, l, d = x.shape
        f = e.shape[1]
        ss = self.modulation.unsqueeze(1) + e.unsqueeze(2)  # (B, F, 2, C)
        shift, scale = ss[:, :, 0:1], ss[:, :, 1:2]
        h = self.norm(x).reshape(b, f, l // f, d) * (1 + scale) + shift
        return self.head(h.reshape(b, l, d))


class CausalWanModel(nn.Module):
    """The generator. See the module docstring; :meth:`forward` is one KV-cache call of the reference."""

    def __init__(self, cfg: DiTConfig = DiTConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = nn.Conv3d(cfg.in_dim, cfg.dim, cfg.patch_size, stride=cfg.patch_size)  # fmt: skip
        self.text_embedding = nn.Sequential(nn.Linear(cfg.text_dim, cfg.dim), nn.GELU(approximate="tanh"), nn.Linear(cfg.dim, cfg.dim))  # fmt: skip
        self.time_embedding = nn.Sequential(nn.Linear(cfg.freq_dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))  # fmt: skip
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(cfg.dim, cfg.dim * 6))
        self.blocks = nn.ModuleList([AttentionBlock(cfg) for _ in range(cfg.num_layers)])  # fmt: skip
        self.head = Head(cfg)
        d = cfg.head_dim
        # ``init_freqs``: 44 / 42 / 42 of the 128 head dims for time / row / column (float64, CPU).
        self._angles = (
            rope_angles(1024, d - 4 * (d // 6)),
            rope_angles(1024, 2 * (d // 6)),
            rope_angles(1024, 2 * (d // 6)),
        )
        self._phase_weights: Tensor | None = None
        # Optional ``runner(i, args, hist) -> (x, k, v)`` replacing ``blocks[i].step``
        # (``remind_lucid.accel.BlockPrograms``); None = eager.
        self.runner = None

    def _probing(self) -> bool:
        return any(blk.self_attn.probe is not None for blk in self.blocks)

    @property
    def dtype(self):
        return self.patch_embedding.weight.dtype

    @property
    def device(self) -> str:
        return self.patch_embedding.weight.device.type

    def new_cache(self) -> KVCache:
        return KVCache(len(self.blocks))

    # -- LoRA --------------------------------------------------------------- #

    def lora_targets(self) -> dict[str, LoRALinear]:
        """``causal_block_linear_targets``: every linear of a block except ``camera_phase_mlp.proj``."""
        return {name: m for name, m in self.named_modules() if isinstance(m, LoRALinear)}  # fmt: skip

    def merge_lora(self, state: dict[str, Tensor], alpha: float, rank: int) -> int:
        """Merge an adapter saved as ``base_model.model.<module>.lora_{A,B}.weight``; strict both ways."""
        targets = self.lora_targets()
        prefix = "base_model.model."
        expected = {f"{prefix}{n}.lora_{ab}.weight" for n in targets for ab in "AB"}
        if expected != set(state):
            missing, unexpected = sorted(expected - set(state)), sorted(set(state) - expected)  # fmt: skip
            raise RuntimeError(f"LoRA key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")  # fmt: skip
        for name, module in targets.items():
            a, b = state[f"{prefix}{name}.lora_A.weight"], state[f"{prefix}{name}.lora_B.weight"]  # fmt: skip
            if a.shape[0] != rank:
                raise RuntimeError(f"{name}: adapter rank {a.shape[0]} != {rank}")
            module.merge_lora(a, b, alpha, rank)
        self._phase_weights = None
        return len(targets)

    # -- geometry ----------------------------------------------------------- #

    def geometry(self, positions: list[int], camera: CameraChunk, grid: tuple[int, int]) -> ChunkGeometry:  # fmt: skip
        """RoPE tables (camera phase included) and projective matrices for one chunk.

        ``positions`` are the absolute temporal positions of the frames;
        ``grid`` is ``(rows, columns)`` of tokens per frame.

        The reference multiplies its complex128 RoPE table by a complex64
        phasor ``polar(1, phi)`` and casts the product to complex64. The same
        here: ``phi`` in the weights' dtype then fp32, ``cos``/``sin`` of it
        in fp32, the product with the float64 table in float64, cast to fp32.
        """
        h, w = grid
        f = len(positions)
        cfg = self.cfg
        index = lucid.tensor(positions, dtype=lucid.int64, device="cpu")
        ta = self._angles[0].index_select(0, index)  # (F, 22)
        angle = lucid.cat(
            [
                ta.reshape(f, 1, 1, -1).expand(f, h, w, ta.shape[1]),
                self._angles[1][:h].reshape(1, h, 1, -1).expand(f, h, w, self._angles[1].shape[1]),
                self._angles[2][:w].reshape(1, 1, w, -1).expand(f, h, w, self._angles[2].shape[1]),
            ],
            dim=-1,
        ).reshape(f, h * w, -1)  # fmt: skip
        base_cos, base_sin = lucid.cos(angle), lucid.sin(angle)  # float64 (F, hw, 64)

        if self._phase_weights is None:
            self._phase_weights = lucid.stack([blk.self_attn.camera_phase_mlp.proj.weight.to("cpu") for blk in self.blocks])  # fmt: skip
        weights = self._phase_weights  # (blocks, 64, 14) in the model dtype
        pose = camera.pose.to(weights.dtype)[0]  # (F, 14), rounded as ``pose_c.to(w.dtype)``
        phi = lucid.matmul(pose.double().unsqueeze(0), weights.double().swapaxes(1, 2))
        phi = phi.to(weights.dtype).float()  # (blocks, F, 64): F.linear in the weight dtype, then .float()
        pc, ps = lucid.cos(phi).double().unsqueeze(2), lucid.sin(phi).double().unsqueeze(2)  # fmt: skip
        cos = (base_cos.unsqueeze(0) * pc - base_sin.unsqueeze(0) * ps).float()
        sin = (base_sin.unsqueeze(0) * pc + base_cos.unsqueeze(0) * ps).float()
        nb, half = cos.shape[0], cos.shape[-1]
        cos = cos.reshape(nb, f * h * w, half, 1).expand(nb, f * h * w, half, 2).reshape(nb, f * h * w, 2 * half)  # fmt: skip
        sin = sin.reshape(nb, f * h * w, half, 1).expand(nb, f * h * w, half, 2).reshape(nb, f * h * w, 2 * half)  # fmt: skip
        cos, sin = cos.to(self.device), sin.to(self.device)
        p_t = camera.P.swapaxes(-1, -2).reshape(1, 1, f, 1, 4, 4).to(self.device)
        p_inv_t = camera.P_inv.swapaxes(-1, -2).reshape(1, 1, f, 1, 4, 4).to(self.device)  # fmt: skip
        lucid.eval(cos, sin, p_t, p_inv_t)
        return ChunkGeometry([cos[i] for i in range(nb)], [sin[i] for i in range(nb)], p_t, p_inv_t, f)  # fmt: skip

    # -- embeddings --------------------------------------------------------- #

    def embed_text(self, context: Tensor) -> Tensor:
        """``text_embedding`` of raw UMT5 features ``(N, L, 4096)``, padded or cut to ``text_len``."""
        n, l, c = context.shape
        text_len = self.cfg.text_len
        if l > text_len:
            context = context[:, :text_len]
        elif l < text_len:
            pad = lucid.zeros((n, text_len - l, c), dtype=context.dtype, device=context.device.type)  # fmt: skip
            context = lucid.cat([context, pad], dim=1)
        return self.text_embedding(context.to(self.dtype))

    def embed_time(self, timesteps: list[float]) -> tuple[Tensor, Tensor]:
        """Per-frame ``e`` ``(1, F, C)`` and ``e0`` ``(1, F, 6, C)``."""
        f = len(timesteps)
        sin = sinusoidal_embedding(self.cfg.freq_dim, timesteps).to(self.dtype).to(self.device)  # fmt: skip
        e = self.time_embedding(sin)
        e0 = self.time_projection(e).reshape(1, f, 6, self.cfg.dim)
        return e.reshape(1, f, self.cfg.dim), e0

    def unpatchify(self, x: Tensor, grid: tuple[int, int, int]) -> Tensor:
        """``(1, L, C * 4)`` -> ``(1, C, F, 2h, 2w)`` (``fhwpqrc -> cfphqwr``)."""
        f, h, w = grid
        pt, ph, pw = self.cfg.patch_size
        c = self.cfg.out_dim
        u = x.reshape(f, h, w, pt, ph, pw, c).permute(6, 0, 3, 1, 4, 2, 5)
        return u.reshape(1, c, f * pt, h * ph, w * pw)

    # -- forward ------------------------------------------------------------ #

    def forward(
        self,
        x: Tensor,
        timesteps: list[float],
        context: Tensor,
        geom: ChunkGeometry,
        cache: KVCache,
        write_key: int | None = None,
        taps: dict | None = None,
    ) -> Tensor:
        """One KV-cache call.

        Parameters
        ----------
        x : Tensor
            ``(1, 48, F, H, W)`` latents of the chunk.
        timesteps : list[float]
            One timestep per frame.
        context : Tensor
            ``(N, 512, C)`` *embedded* text contexts (:meth:`embed_text`), one
            per group of ``F / N`` frames.
        geom : ChunkGeometry
        cache : KVCache
            Read when ``write_key`` is ``None`` (a denoising step,
            ``kv_size = (0, hist)``); otherwise this chunk's K/V are written
            under ``write_key`` and the chunk attends to itself only
            (``kv_size = (hist, -1)``).
        taps : dict, optional
            Filled with ``block_<i>`` outputs for parity tests.

        Returns
        -------
        Tensor
            The flow prediction, ``(1, 48, F, H, W)``.
        """
        f = x.shape[2]
        h, w = x.shape[3] // self.cfg.patch_size[1], x.shape[4] // self.cfg.patch_size[2]  # fmt: skip
        tokens = self.patch_embedding(x.to(self.dtype))
        tokens = tokens.reshape(1, self.cfg.dim, f * h * w).permute(0, 2, 1)
        e, e0 = self.embed_time(timesteps)
        if taps is not None:
            taps["patch"], taps["e"] = tokens, e
        runner = self.runner if (self.runner is not None and not self._probing()) else None
        for i, block in enumerate(self.blocks):
            # A commit pass attends to the chunk alone and caches its K/V; a
            # denoising step reads the history.
            hist = None if write_key is not None else cache.read(i)
            args = (tokens, e0, context, geom.cos[i], geom.sin[i], geom.P_t, geom.P_inv_t)
            if runner is not None:
                tokens, k, v = runner(i, args, hist)
            elif hist is None:
                tokens, k, v = block.step(*args, block=i)
            else:
                tokens, k, v = block.step(*args, hist[0], hist[1], block=i)
            if write_key is not None:
                cache.write(i, write_key, k, v)
            lucid.eval(tokens)
            if taps is not None:
                taps[f"block_{i}"] = tokens
        out = self.head(tokens, e)
        return self.unpatchify(out, (f, h, w))
