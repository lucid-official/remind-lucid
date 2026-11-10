"""The released files this port reads, where they land, and the model's fixed settings.

``checkpoints/`` mirrors the two folders the reference is pointed at
(``--model-folder`` = ``Wan2.2-TI2V-5B``, ``--base-checkpoint`` /
``--ema-checkpoint`` = ``ReMind``), holding only what inference reads:

==============================================  =========  =========================================
file                                            on disk    source
==============================================  =========  =========================================
ReMind/ReMind-5B.safetensors                    11.1 GB    as released (bf16), SHA-256 verified
ReMind/ReMind-5b-dmd-ema.safetensors            1.5 GB     as released (fp32), SHA-256 verified
Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16...   11.4 GB    ``.pth`` -> SafeTensors, bf16 as released
Wan2.2-TI2V-5B/Wan2.2_VAE.safetensors           2.8 GB     ``.pth`` -> SafeTensors, fp32 as released
Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer.json   17 MB      as released
==============================================  =========  =========================================

``ReMind-5B.safetensors`` carries every one of the generator's 975 tensors,
so none of the 20 GB fp32 Wan2.2 DiT shards is needed; :func:`base_keys_needed`
re-derives that from the remote headers instead of assuming it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REMIND_REPO = "AppliedIntuitionResearch/ReMind"
WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B"
GENERATOR_FILE = "ReMind-5B.safetensors"
LORA_FILE = "ReMind-5b-dmd-ema.safetensors"
T5_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
VAE_FILE = "Wan2.2_VAE.pth"
WAN_SHARDS = [
    f"diffusion_pytorch_model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)
]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = Path(os.environ.get("REMIND_LUCID_CHECKPOINTS", ROOT / "checkpoints"))

# configs/model_5b.yaml + configs/default_config.yaml + Wan2.2-TI2V-5B/config.json
DENOISING_STEPS = (1000, 938, 833, 625)
TIMESTEP_SHIFT = 5.0
NUM_TRAIN_TIMESTEPS = 1000
NUM_FRAME_PER_BLOCK = 3
FILTER_LIST = (
    "img_emb.",
    "cross_attn.k_img",
    "cross_attn.v_img",
    "cross_attn.norm_k_img",
)
LORA_RANK = 128
LORA_ALPHA = 128
# Present in the config but never read on the DMD inference path (see docs/PORTING_PLAN.md).
GUIDANCE_SCALE = 3.0
NEGATIVE_PROMPT = (
    "blurry, low quality, worst quality, ugly, deformed, watermark, text, static"
)


def remind_dir() -> Path:
    return CHECKPOINTS / "ReMind"


def wan_dir() -> Path:
    return CHECKPOINTS / "Wan2.2-TI2V-5B"


def generator_path() -> Path:
    return remind_dir() / GENERATOR_FILE


def lora_path() -> Path:
    return remind_dir() / LORA_FILE


def merged_path(dtype: str = "bf16") -> Path:
    """Cache of the generator with the DMD EMA LoRA merged (written on first load)."""
    return remind_dir() / f"ReMind-5B-dmd-ema-merged-{dtype}.safetensors"


def t5_path() -> Path:
    return wan_dir() / "models_t5_umt5-xxl-enc-bf16.safetensors"


def vae_path() -> Path:
    return wan_dir() / "Wan2.2_VAE.safetensors"


def base_extra_path() -> Path:
    return wan_dir() / "wan_base_extra.safetensors"


def tokenizer_dir() -> Path:
    return wan_dir() / "google" / "umt5-xxl"


def prompt_cache_dir() -> Path:
    return CHECKPOINTS / "prompt_cache"


@dataclass(frozen=True)
class Item:
    """One file the pipeline needs."""

    key: str
    label: str
    path: Path
    size: int  # bytes on disk when complete (released sizes)
    source: str

    def state(self) -> str:
        if self.path.exists():
            return "verified" if self.sha_ok() is not False else "unverified"
        part = self.path.with_suffix(self.path.suffix + ".part")
        return "partial" if part.exists() else "missing"

    def sha_ok(self) -> bool | None:
        """True/False when the file has a recorded SHA-256 check, None when it has no published hash."""
        if self.key not in ("generator", "lora"):
            return None
        return self.path.with_suffix(self.path.suffix + ".sha256").exists()

    def have(self) -> int:
        if self.path.exists():
            return self.path.stat().st_size
        part = self.path.with_suffix(self.path.suffix + ".part")
        return part.stat().st_size if part.exists() else 0


def items() -> list[Item]:
    return [
        Item("generator", "ReMind-5B generator (bf16)", generator_path(), 11_132_561_488, f"{REMIND_REPO}/{GENERATOR_FILE}"),
        Item("lora", "DMD EMA LoRA, rank 128 (fp32)", lora_path(), 1_478_584_376, f"{REMIND_REPO}/{LORA_FILE}"),
        Item("t5", "UMT5-XXL encoder (bf16)", t5_path(), 11_361_920_418, f"{WAN_REPO}/{T5_FILE}"),
        Item("vae", "Wan2.2 VAE (fp32)", vae_path(), 2_818_839_170, f"{WAN_REPO}/{VAE_FILE}"),
        Item("tokenizer", "UMT5 tokenizer", tokenizer_dir() / "tokenizer.json", 16_837_417, f"{WAN_REPO}/google/umt5-xxl/tokenizer.json"),
    ]  # fmt: skip


def status() -> list[dict]:
    return [
        {
            "key": it.key,
            "label": it.label,
            "path": str(it.path),
            "source": it.source,
            "state": it.state(),
            "have": it.have(),
            "size": it.size,
        }
        for it in items()
    ]


def ready() -> bool:
    return all(it.path.exists() for it in items())


def clean_key(key: str) -> str:
    """``pipeline/checkpoints.py::_clean_key``."""
    prefixes = ("module.", "_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod.", "model.")  # fmt: skip
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def base_keys_needed(generator_keys, base_keys) -> list[str]:
    """Wan base tensors that survive ``filter_list`` and that the ReMind generator does not overwrite."""
    have = {clean_key(k) for k in generator_keys}
    return [
        k for k in base_keys if not any(s in k for s in FILTER_LIST) and k not in have
    ]
