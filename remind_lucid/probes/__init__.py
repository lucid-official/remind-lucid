"""Research probes for ReMind's dynamic memory.

Every probe is off unless asked for, and with no probe the pipeline takes
exactly the reference path: ``ReMindPipeline.probes`` is ``None`` and
``CausalSelfAttention.probe`` is ``None``. ``tests/test_probes.py`` checks
that an all-off :class:`ProbeSet`, and attention logging (which only
observes), leave the output bit-identical.

Hooks (they change what the generator sees):

* **RoPE gap** — from chunk ``c`` on, temporal positions are shifted by ``G``
  latent frames, leaving an empty stretch of time between the history and
  the chunks being generated. Cached keys keep their rotation, so the model
  sees its memories as ``G`` frames older.
* **Anchor knockout** — hide selected cached chunks from later chunks, or
  replace their cache entries by re-encoding other content (noise, zeros, or
  another chunk's latents) at the same temporal position.
* **Attention mass** — for chosen layers, the softmax mass each head puts on
  each cached chunk, averaged over queries and denoising steps.

Sweeps (they only build variations of a preset): interruption duration and
seeds, in :mod:`remind_lucid.probes.sweeps`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lucid
import lucid.nn.functional as F

Tensor = lucid.Tensor


@dataclass
class ProbeSet:
    """The hooks of one run. All defaults are "off".

    Attributes
    ----------
    gap_from_chunk, gap_frames : int
        RoPE gap: chunks ``>= gap_from_chunk`` are placed ``gap_frames``
        latent frames later in time (0 = off).
    knockout : dict[int, list[int]]
        ``{c: [a, b]}``: while generating chunk ``c`` and later, cached chunks
        ``a`` and ``b`` are hidden. Applied cumulatively in chunk order.
    replace : dict[int, str]
        ``{a: mode}``: right after chunk ``a`` is committed, its cache entry
        is rewritten from other content: ``"noise"`` (unit Gaussian),
        ``"zeros"``, or ``"chunk:<k>"`` (the latents of chunk ``k``).
    attention_layers : list[int]
        Blocks whose attention mass is logged (empty = off).
    replace_seed : int
        Seed of the ``"noise"`` replacement.
    """

    gap_from_chunk: int = 0
    gap_frames: int = 0
    knockout: dict[int, list[int]] = field(default_factory=dict)
    replace: dict[int, str] = field(default_factory=dict)
    attention_layers: list[int] = field(default_factory=list)
    replace_seed: int = 0

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_dict(cls, data: dict | None) -> "ProbeSet | None":
        """A probe set from the JSON the web demo and the batch runner use; ``None`` if nothing is on."""
        if not data:
            return None
        probe = cls(
            gap_from_chunk=int(data.get("gap_from_chunk", 0)),
            gap_frames=int(data.get("gap_frames", 0)),
            knockout={int(k): [int(a) for a in v] for k, v in (data.get("knockout") or {}).items()},  # fmt: skip
            replace={int(k): str(v) for k, v in (data.get("replace") or {}).items()},
            attention_layers=[int(v) for v in data.get("attention_layers") or []],
            replace_seed=int(data.get("replace_seed", 0)),
        )
        return probe if probe.active() else None

    def to_dict(self) -> dict:
        return {
            "gap_from_chunk": self.gap_from_chunk,
            "gap_frames": self.gap_frames,
            "knockout": {str(k): v for k, v in self.knockout.items()},
            "replace": {str(k): v for k, v in self.replace.items()},
            "attention_layers": list(self.attention_layers),
            "replace_seed": self.replace_seed,
        }

    def active(self) -> bool:
        return bool(self.gap_frames or self.knockout or self.replace or self.attention_layers)  # fmt: skip

    # -- pipeline hooks ----------------------------------------------------- #

    def begin(self, pipeline, job, cache) -> None:
        self._pipeline, self._job, self._cache = pipeline, job, cache
        self._hidden: set[int] = set()
        self._latents: dict[int, Tensor] = {}
        self._chunk = 0
        self.mass: dict[int, dict[int, list]] = (
            {}
        )  # chunk -> layer -> [heads][key chunks]
        self._acc: dict[int, list] = {}
        if self.attention_layers:
            model = pipeline.generator
            for index in self.attention_layers:
                model.blocks[index].self_attn.probe = self._observe

    def positions(self, chunk: int, positions: list[int]) -> list[int]:
        if self.gap_frames and chunk >= self.gap_from_chunk:
            return [p + self.gap_frames for p in positions]
        return positions

    def before_chunk(self, chunk: int) -> None:
        self._chunk = chunk
        self._acc = {}
        if chunk in self.knockout:
            self._hidden |= set(self.knockout[chunk])
            self._cache.hide(self._hidden)

    def after_commit(self, chunk: int) -> None:
        if self._acc:
            self.mass[chunk] = {layer: [[v / n for v in head] for head in rows] for layer, (rows, n) in self._acc.items()}  # fmt: skip
        mode = self.replace.get(chunk)
        if mode is not None:
            self._rewrite(chunk, mode)

    def end(self) -> None:
        if self.attention_layers:
            for index in self.attention_layers:
                self._pipeline.generator.blocks[index].self_attn.probe = None

    # -- anchor replacement --------------------------------------------------- #

    def _rewrite(self, chunk: int, mode: str) -> None:
        """Re-run the commit pass of ``chunk`` on replacement latents (``replace_chunk_in_cache``)."""
        pipe, job = self._pipeline, self._job
        model = pipe.generator
        cs = job.chunk_size
        entry = self._cache.entries[0][chunk][0]
        h, w = job.height // 16, job.width // 16
        if mode == "noise":
            generator = lucid.manual_seed(self.replace_seed + chunk)
            latents = lucid.randn(1, cs, 48, h, w, dtype=lucid.float32, device="cpu", generator=generator)  # fmt: skip
        elif mode == "zeros":
            latents = lucid.zeros((1, cs, 48, h, w), dtype=lucid.float32, device="cpu")
        elif mode.startswith("chunk:"):
            latents = self._committed[int(mode.split(":", 1)[1])]
        else:
            raise ValueError(f"unknown replacement {mode!r}")
        latents = latents.to(entry.device.type).to(entry.dtype)
        context, geom = self._context[chunk], self._geom[chunk]
        model(latents.permute(0, 2, 1, 3, 4), [0.0] * cs, context, geom, self._cache, write_key=chunk)  # fmt: skip

    def remember(self, chunk: int, latents: Tensor, context: Tensor, geom) -> None:
        """Called by the pipeline at commit time; replacement needs the chunk's context and geometry."""
        if not hasattr(self, "_committed"):
            self._committed, self._context, self._geom = {}, {}, {}
        self._committed[chunk], self._context[chunk], self._geom[chunk] = latents, context, geom  # fmt: skip

    # -- attention mass ----------------------------------------------------- #

    def _observe(self, block: int, q: Tensor, k: Tensor, v: Tensor, own: int) -> Tensor:
        """Log the mass on each cached chunk; return what the reference path returns.

        The output is the fused kernel's, untouched. The mass is a side
        computation in fp32 over slices of queries.
        """
        out = F.scaled_dot_product_attention(q, k, v)
        if (
            self._cache.entries[0]
            and k.shape[2] > own
            and getattr(self, "_logging", True)
        ):
            keys = self._cache.visible()
            sizes = [self._cache.entries[block][key][0].shape[2] for key in keys] + [
                own
            ]
            scale = q.shape[-1] ** -0.5
            kf = k.float().swapaxes(-1, -2)
            heads = q.shape[1]
            totals = lucid.zeros((heads, len(sizes)), dtype=lucid.float32, device=q.device.type)  # fmt: skip
            step = 195
            for start in range(0, q.shape[2], step):
                attn = F.softmax(lucid.matmul(q[:, :, start : start + step].float(), kf) * scale, dim=-1)  # fmt: skip
                offset, cols = 0, []
                for size in sizes:
                    cols.append(
                        attn[..., offset : offset + size].sum(dim=-1).sum(dim=-1)[0]
                    )  # (heads,)
                    offset += size
                totals = totals + lucid.stack(cols, dim=-1)
                lucid.eval(totals)
            rows = (totals / q.shape[2]).to("cpu").tolist()
            if block in self._acc:
                prev, n = self._acc[block]
                rows = [[a + b for a, b in zip(r0, r1)] for r0, r1 in zip(prev, rows)]
                self._acc[block] = (rows, n + 1)
            else:
                self._acc[block] = (rows, 1)
            self.keys = getattr(self, "keys", {})
            self.keys[self._chunk] = keys + [self._chunk]
        return out

    def report(self) -> dict:
        """JSON-ready results: ``mass[chunk][layer][head][key]`` and the key chunk of each column."""
        return {
            "config": self.to_dict(),
            "attention_mass": {str(c): {str(l): rows for l, rows in layers.items()} for c, layers in self.mass.items()},  # fmt: skip
            "attention_keys": {
                str(c): keys for c, keys in getattr(self, "keys", {}).items()
            },
        }
