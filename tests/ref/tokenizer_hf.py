"""Tokenizer reference: upstream's ``HuggingfaceTokenizer`` (HF ``tokenizers`` + ftfy cleaning).

    .venv-ref/bin/python tests/ref/tokenizer_hf.py

Writes ``tests/fixtures/tokenizer_hf.json``: for a varied prompt set (every
chunk prompt of the 11 presets, CJK, emoji, digits, special tokens, curly
quotes, HTML entities, a prompt long enough to truncate) the cleaned text,
the 512 ids and the mask length.
"""

import _upstream  # noqa: F401
import json

from wan.modules.tokenizers import HuggingfaceTokenizer

tok = HuggingfaceTokenizer(name=str(_upstream.CHECKPOINTS / "Wan2.2-TI2V-5B" / "google" / "umt5-xxl") + "/", seq_len=512, clean="whitespace")
dry = json.loads((_upstream.FIXTURES / "presets_dry_run.json").read_text(encoding="utf-8"))
prompts = sorted({p for case in dry.values() for p in case["prompt_schedule"]})
prompts += [
    "",
    "A video.",
    "  Leading   and \t trailing\n whitespace  ",
    "こんにちは世界、これは日本語のテストです。猫が窓辺で眠っている。",
    "中文提示：一只白色的猫在雪地里奔跑，镜头缓慢推进。",
    "한국어 프롬프트: 강아지가 공원에서 뛰어놀고 있다.",
    "Emoji 🌍🚀 and ZWJ 👩‍🚀 sequences, plus ✨ sparkles.",
    "Digits 0123456789 3.14159 1,000,000 2026-11-11 08:30 100% #42 $9.99",
    "Specials </s> <pad> <unk> <extra_id_0> <extra_id_99> ▁underscore",
    "“Curly quotes” and ‘single’ — em dash – en dash … ellipsis",
    "HTML &amp; entities &lt;b&gt; &amp;amp; &quot;quoted&quot; &#39;apos&#39; &nbsp;nbsp",
    "Ligatures: ﬁne ﬂow oﬃce; full-width: ＡＢＣ１２３！",
    "naïve café déjà vu Ångström ñandú Straße œuvre",
    "Mixed UPPER lower CamelCase snake_case kebab-case path/to/file.txt http://example.com?q=1&r=2",
    "Tabs\tand\nnewlines\r\nare collapsed",
    "word " * 700,
    "The quick brown fox jumps over the lazy dog. " * 60,
]
rows = []
for text in prompts:
    ids, mask = tok([text], return_mask=True, add_special_tokens=True)
    rows.append({"text": text, "clean": tok._clean(text), "ids": ids[0].tolist()[: int(mask[0].sum())], "length": int(mask[0].sum())})
(_upstream.FIXTURES / "tokenizer_hf.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
print(len(rows), "prompts; max length", max(r["length"] for r in rows))
