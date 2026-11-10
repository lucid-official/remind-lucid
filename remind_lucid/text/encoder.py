"""Prompt -> UMT5 features, as ``WanTextEncoder.forward``, with a disk cache.

The reference encodes each *unique* chunk prompt once, zeroes the padded
positions and gathers the rows back into ``(1, chunks, 512, 4096)``. The
same is done here; features are additionally cached on disk under a hash of
the cleaned text and the encoder settings, so a repeated prompt costs one
file read and the 11 GB encoder is not loaded at all.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Sequence

import lucid

from remind_lucid import catalog
from remind_lucid.models.umt5 import UMT5Encoder
from remind_lucid.text.tokenizer import UMT5Tokenizer, clean_prompt

Tensor = lucid.Tensor
SEQ_LEN = 512


def load_text_encoder(device: str = "metal", dtype=lucid.bfloat16) -> UMT5Encoder:
    from remind_lucid.pipeline import build_on

    model = build_on(device, dtype, UMT5Encoder)
    state = lucid.load_safetensors(str(catalog.t5_path()), device=device)
    state = {k: (v if v.dtype == dtype else v.to(dtype)) for k, v in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def _cache_path(text: str, dtype_name: str) -> Path:
    key = f"umt5-xxl|seq{SEQ_LEN}|{dtype_name}|{clean_prompt(text)}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return catalog.prompt_cache_dir() / f"{digest}.safetensors"


def encode_one(model: UMT5Encoder, tokenizer: UMT5Tokenizer, text: str) -> Tensor:
    """One prompt -> ``(512, 4096)`` with padded rows zeroed."""
    ids, mask = tokenizer.batch([text])
    with lucid.no_grad():
        ids_t = lucid.tensor(ids, dtype=lucid.int64, device="cpu")
        mask_t = lucid.tensor(mask, dtype=lucid.int64, device="cpu")
        context = model(ids_t, mask_t)[0]
        keep = mask_t.to(context.device.type)[0].reshape(-1, 1) != 0
        context = lucid.where(keep, context, 0.0)
        lucid.eval(context)
    return context


class PromptEncoder:
    """Encodes prompt schedules, loading UMT5 only when the cache misses.

    Parameters
    ----------
    device, dtype
        Where and in which dtype UMT5 runs (the reference: bf16).
    use_cache : bool
        Read and write ``checkpoints/prompt_cache``.
    """

    def __init__(self, device: str = "metal", dtype=lucid.bfloat16, use_cache: bool = True) -> None:  # fmt: skip
        self.device, self.dtype, self.use_cache = device, dtype, use_cache
        self.model: UMT5Encoder | None = None
        self.tokenizer: UMT5Tokenizer | None = None
        self.misses = 0

    def _ensure(self) -> None:
        if self.model is None:
            self.tokenizer = UMT5Tokenizer.load(catalog.tokenizer_dir(), SEQ_LEN)
            self.model = load_text_encoder(self.device, self.dtype)

    def cached(self, prompts: Sequence[str]) -> bool:
        """Whether every prompt's features are on disk (UMT5 would not be loaded)."""
        name = str(self.dtype).rsplit(".", 1)[-1]
        return self.use_cache and all(_cache_path(str(p), name).exists() for p in prompts)

    def release(self) -> None:
        """Drop the encoder (staged residency: it is not needed once prompts are encoded)."""
        self.model = None
        if self.device == "metal":
            lucid.metal.empty_cache()

    def encode(self, prompts: Sequence[str], on_prompt: Callable[[int, int], None] | None = None) -> Tensor:  # fmt: skip
        """Chunk prompts -> ``(1, len(prompts), 512, 4096)`` on ``device``."""
        unique = list(dict.fromkeys(str(p) for p in prompts))
        name = str(self.dtype).rsplit(".", 1)[-1]
        features: dict[str, Tensor] = {}
        for i, text in enumerate(unique):
            path = _cache_path(text, name)
            if self.use_cache and path.exists():
                features[text] = lucid.load_safetensors(str(path))["context"]
            else:
                self._ensure()
                self.misses += 1
                value = encode_one(self.model, self.tokenizer, text).to("cpu")
                if self.use_cache:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    lucid.save_safetensors({"context": value}, str(path))
                features[text] = value
            if on_prompt is not None:
                on_prompt(i + 1, len(unique))
        stacked = lucid.stack([features[str(p)] for p in prompts], dim=0)
        return stacked.unsqueeze(0).to(self.device)
