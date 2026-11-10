"""The tokenizer is upstream's ``HuggingfaceTokenizer`` id for id, cleaning and truncation included."""

import json

import pytest

from conftest import FIXTURES
from remind_lucid import catalog
from remind_lucid.text.tokenizer import UMT5Tokenizer, clean_prompt

PATH = FIXTURES / "tokenizer_hf.json"


@pytest.fixture(scope="module")
def rows():
    if not PATH.exists():
        pytest.skip("run tests/ref/tokenizer_hf.py in .venv-ref")
    return json.loads(PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tokenizer():
    if not (catalog.tokenizer_dir() / "tokenizer.json").exists():
        pytest.skip("tokenizer.json not downloaded")
    return UMT5Tokenizer.load(catalog.tokenizer_dir())


def test_cleaning(rows):
    assert [clean_prompt(r["text"]) for r in rows] == [r["clean"] for r in rows]


def test_ids(rows, tokenizer):
    bad = [r["text"][:40] for r in rows if tokenizer.encode_clean(r["text"]) != r["ids"]]
    assert not bad, bad


def test_padding_and_mask(rows, tokenizer):
    ids, mask = tokenizer.batch([r["text"] for r in rows])
    assert all(len(row) == 512 for row in ids)
    assert [sum(m) for m in mask] == [r["length"] for r in rows]
    assert max(r["length"] for r in rows) == 512  # the set includes truncated prompts
