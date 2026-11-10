"""The UMT5 tokenizer and the reference's prompt cleaning.

Reproduces what ``HuggingfaceTokenizer(name='google/umt5-xxl', seq_len=512,
clean='whitespace')`` hands the text encoder:

* cleaning: ``whitespace_clean(basic_clean(text))`` — ``ftfy.fix_text``, two
  rounds of HTML unescaping, whitespace collapsing. ftfy is not available
  here; the subset of its default fixes that can change a prompt that is not
  mojibake (curly quotes, Latin ligatures, full-width forms, control
  characters, NFC) is re-implemented. Mojibake repair is not.
* tokenisation: the SentencePiece Unigram model (Viterbi) of
  ``tokenizer.json`` with its Metaspace pre-tokenizer and ``$A </s>``
  template.
* ``padding='max_length', truncation=True, max_length=512``: content is cut
  to 511 pieces so ``</s>`` survives, then right-padded with ``<pad>`` (0).
"""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path
from typing import Sequence

from lucid.utils.tokenizer import UnigramTokenizerFast

_QUOTES = str.maketrans(
    {"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"', "‟": '"'}
)
_LIGATURES = str.maketrans(
    {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st", "Ĳ": "IJ", "ĳ": "ij"}
)  # fmt: skip


def _fix_width(text: str) -> str:
    """``ftfy.fix_character_width``: full-width ASCII and the ideographic space."""
    return "".join(
        unicodedata.normalize("NFKC", c) if ("！" <= c <= "～" or c == "　") else c
        for c in text
    )


def clean_prompt(text: str) -> str:
    """``whitespace_clean(basic_clean(text))`` of ``wan/modules/tokenizers.py``."""
    text = text.translate(_QUOTES).translate(_LIGATURES)
    text = _fix_width(text)
    text = "".join(c for c in text if c in "\n\t\r" or unicodedata.category(c) != "Cc")
    text = unicodedata.normalize("NFC", text)
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


class UMT5Tokenizer(UnigramTokenizerFast):
    """Lucid's Unigram tokenizer with the reference's cleaning, truncation and padding.

    Subclasses :class:`lucid.utils.tokenizer.UnigramTokenizerFast` (the
    Viterbi segmentation and the ``tokenizer.json`` pipeline are used as they
    are); only the fixed-length batch call is added.
    """

    seq_len = 512
    pad_id = 0
    eos_id = 1

    @classmethod
    def load(cls, directory: str | Path, seq_len: int = 512) -> "UMT5Tokenizer":
        tok = cls.from_file(str(directory))
        tok.seq_len = seq_len
        return tok

    def encode_clean(self, text: str) -> list[int]:
        """Ids of one prompt, ``</s>`` included, at most ``seq_len`` long."""
        ids = self.encode(clean_prompt(text), add_special_tokens=False)
        return ids[: self.seq_len - 1] + [self.eos_id]

    def batch(self, texts: Sequence[str]) -> tuple[list[list[int]], list[list[int]]]:
        """``(input_ids, attention_mask)`` rows of length ``seq_len``."""
        rows, masks = [], []
        for text in texts:
            ids = self.encode_clean(text)
            pad = self.seq_len - len(ids)
            rows.append(ids + [self.pad_id] * pad)
            masks.append([1] * len(ids) + [0] * pad)
        return rows, masks
