"""Secondary UMT5 reference: Hugging Face ``transformers`` ``UMT5EncoderModel``, first layers, fp32.

    .venv-ref/bin/python tests/ref/umt5_hf.py

An implementation independent of both upstream's ``t5.py`` and this port.
The released Wan weights are renamed to the HF layout and loaded into a
two-layer ``UMT5EncoderModel`` (the full fp32 encoder would not fit beside
everything else); its hidden state after two blocks is compared with this
port's ``UMT5Encoder(..., num_layers=2)``. Writes
``tests/fixtures/umt5_hf_large.safetensors`` (not in git).
"""

import _upstream  # noqa: F401
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

LAYERS = 2
PROMPTS = [
    "An espresso machine steadily pours into a short glass cup: at the start the cup shows a dark coffee base topped by a thin crema disc with latte-art.",
    "こんにちは世界 🌍 12345",
]
wan = _upstream.CHECKPOINTS / "Wan2.2-TI2V-5B"
config = UMT5Config(vocab_size=256384, d_model=4096, d_kv=64, d_ff=10240, num_layers=LAYERS, num_heads=64,
                    relative_attention_num_buckets=32, relative_attention_max_distance=128,
                    feed_forward_proj="gated-gelu", dropout_rate=0.0)
model = UMT5EncoderModel(config).eval()


def hf_name(key: str) -> str | None:
    if key == "token_embedding.weight":
        return "shared.weight"
    if not key.startswith("blocks."):
        return None  # the final norm belongs to layer 24
    _, index, rest = key.split(".", 2)
    if int(index) >= LAYERS:
        return None
    prefix = f"encoder.block.{index}.layer"
    table = {
        "norm1.weight": "0.layer_norm.weight", "attn.q.weight": "0.SelfAttention.q.weight",
        "attn.k.weight": "0.SelfAttention.k.weight", "attn.v.weight": "0.SelfAttention.v.weight",
        "attn.o.weight": "0.SelfAttention.o.weight",
        "pos_embedding.embedding.weight": "0.SelfAttention.relative_attention_bias.weight",
        "norm2.weight": "1.layer_norm.weight", "ffn.gate.0.weight": "1.DenseReluDense.wi_0.weight",
        "ffn.fc1.weight": "1.DenseReluDense.wi_1.weight", "ffn.fc2.weight": "1.DenseReluDense.wo.weight",
    }
    return f"{prefix}.{table[rest]}"


state = {}
with safe_open(str(wan / "models_t5_umt5-xxl-enc-bf16.safetensors"), framework="pt") as fh:
    for key in fh.keys():
        name = hf_name(key)
        if name is not None:
            state[name] = fh.get_tensor(key).float()
state["encoder.embed_tokens.weight"] = state["shared.weight"]
missing, unexpected = model.load_state_dict(state, strict=False)
assert not unexpected and all("final_layer_norm" in m for m in missing), (missing, unexpected)

tok = AutoTokenizer.from_pretrained(str(wan / "google" / "umt5-xxl"))
enc = tok(PROMPTS, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
with torch.no_grad():
    hidden = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True).hidden_states[LAYERS]
save_file({"ids": enc.input_ids, "mask": enc.attention_mask, "hidden": hidden.contiguous()}, str(_upstream.FIXTURES / "umt5_hf_large.safetensors"))
print("hidden", tuple(hidden.shape), float(hidden.abs().mean()))
