"""Rust extension vs the pure-Python reference."""

from __future__ import annotations

from pathlib import Path

import pytest

import fast_tokenizer
from reference_bpe import Tokenizer as PyTokenizer

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = (ROOT / "data" / "sample.txt").read_text(encoding="utf-8")


def test_import_and_version():
    assert fast_tokenizer.__version__ == "0.1.0"
    assert "Tokenizer" in repr(fast_tokenizer.Tokenizer.load(str(ROOT / "data" / "gpt2")))


def test_training_matches_the_reference():
    specials = ["<|endoftext|>"]
    rust = fast_tokenizer.Tokenizer.train(SAMPLE, 500, specials)
    py = PyTokenizer.train(SAMPLE, vocab_size=500, special_tokens=specials)
    assert rust.merges() == py.merges()
    assert rust.vocab_size == py.vocab_size
    assert list(rust.special_tokens()) == py.special_tokens()
    held = "low low wider — café <|endoftext|> done\nit's a 19th try"
    assert list(rust.encode(held)) == py.encode(held)
    assert rust.decode(rust.encode(held)) == held == py.decode(py.encode(held))


def test_gpt2_encode_matches_the_reference():
    rust = fast_tokenizer.Tokenizer.load(str(ROOT / "data" / "gpt2"))
    py = PyTokenizer.load(ROOT / "data" / "gpt2")
    text = (ROOT / "data" / "corpora" / "pg11.txt").read_text(encoding="utf-8")[:20000]
    assert list(rust.encode(text)) == py.encode(text)
    assert list(rust.encode_ordinary("<|endoftext|>")) == py.encode_ordinary("<|endoftext|>")
    batch = ["hello world", "café", "<|endoftext|>"]
    assert rust.encode_batch(batch, num_threads=1) == [py.encode(item) for item in batch]
    assert rust.encode_batch(batch, num_threads=2) == [py.encode(item) for item in batch]


def test_cross_language_save_load(tmp_path: Path):
    rust = fast_tokenizer.Tokenizer.train(SAMPLE, 400, ["<|endoftext|>"])
    rust_dir = tmp_path / "rust"
    rust.save(str(rust_dir))
    py = PyTokenizer.load(rust_dir)
    assert py.merges() == rust.merges()
    assert py.encode(SAMPLE[:500]) == list(rust.encode(SAMPLE[:500]))

    py_model = PyTokenizer.train("ab\ncd\nab", vocab_size=270, special_tokens=[])
    py_dir = tmp_path / "py"
    py_model.save(py_dir)
    loaded = fast_tokenizer.Tokenizer.load(str(py_dir))
    assert loaded.merges() == py_model.merges()
    assert list(loaded.encode("ab cd")) == py_model.encode("ab cd")


def test_unknown_id_raises():
    model = fast_tokenizer.Tokenizer.train("hello", 260, [])
    with pytest.raises(ValueError):
        model.decode([10_000_000])
