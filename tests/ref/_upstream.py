"""Import the upstream ReMind code (``reference/``) on a machine without CUDA.

Runs in ``.venv-ref`` (torch). The only changes to upstream behaviour are the
two shims the port's parity plan allows:

* ``cuda -> cpu``: ``torch.cuda.current_device()`` is asked for a device in a
  default argument and in ``sinusoidal_embedding_1d``; it returns ``"cpu"``.
* flash-attn absent -> SDPA: nothing to do, upstream already falls back to
  ``torch.nn.functional.scaled_dot_product_attention`` in bf16.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "reference"
FIXTURES = ROOT / "tests" / "fixtures"
CHECKPOINTS = Path(os.environ.get("REMIND_LUCID_CHECKPOINTS", ROOT / "checkpoints"))

os.environ.setdefault(
    "DISABLE_FLEX_COMPILE", "1"
)  # training-only kernel; keep it eager
os.environ.setdefault("TORCH_NUM_THREADS", str(os.cpu_count() or 8))

import torch  # noqa: E402

torch.cuda.current_device = lambda: "cpu"

if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))


def fp32_attention() -> None:
    """Test-only: run upstream attention in the tensors' own dtype instead of bf16.

    Upstream casts q/k/v to bf16 inside every attention call; for the tiny
    random-weight fp32 architecture checks that cast is replaced so the
    comparison can be made at fp32 precision. Never used with real weights.
    """
    import torch.nn.functional as F
    import wan.modules.causal_model as causal_model
    import wan.modules.model as model

    def attention(q, k, v, *args, **kwargs):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return out.transpose(1, 2).contiguous()

    causal_model.attention = attention
    model.flash_attention = attention
    model.FLASH_ATTN_2_AVAILABLE = True
