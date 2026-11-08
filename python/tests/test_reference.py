"""Tests for the pure-Python reference that do not need the Rust extension."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reference_bpe import Tokenizer, pretokenize

ROOT = Path(__file__).resolve().parents[2]


def test_pretokenizer_matches_the_gpt2_cases():
    assert pretokenize("") == []
    assert pretokenize("hello world") == ["hello", " world"]
    assert pretokenize("it's") == ["it", "'s"]
    assert pretokenize("  leading") == [" ", " leading"]
    assert pretokenize("hello\n\n") == ["hello", "\n\n"]
    assert pretokenize("hello\n\nworld") == ["hello", "\n", "\n", "world"]
    assert pretokenize("hello \n world") == ["hello", " \n", " world"]
    assert pretokenize("<|endoftext|>") == ["<|", "endoftext", "|>"]
    assert pretokenize("the 19th century") == ["the", " 19", "th", " century"]
    assert "".join(pretokenize("café naïve")) == "café naïve"


def test_tie_break_and_round_trip():
    model = Tokenizer.train("ab\ncd", vocab_size=258, special_tokens=[])
    assert model.merges() == [("a", "b"), ("c", "d")]
    assert model.decode(model.encode("ab\ncd")) == "ab\ncd"

    text = "hello <|endoftext|> café 東京 😀\nit's fine"
    model = Tokenizer.train(text, vocab_size=400, special_tokens=["<|endoftext|>"])
    assert model.encode("<|endoftext|>") == [model.token_to_id["<|endoftext|>"]]
    assert len(model.encode_ordinary("<|endoftext|>")) > 1
    assert model.decode(model.encode(text)) == text
    assert model.decode(model.encode("unseen naïve — done")) == "unseen naïve — done"
    assert model.encode("") == []
    assert model.decode([]) == ""


def test_rejects_bad_training_arguments():
    with pytest.raises(ValueError):
        Tokenizer.train("abc", vocab_size=10)
    with pytest.raises(ValueError):
        Tokenizer.train("abc", vocab_size=300, special_tokens=[""])
    with pytest.raises(ValueError):
        Tokenizer.train("abc", vocab_size=300, special_tokens=["<|x|>", "<|x|>"])


def test_gpt2_fixture_and_save_round_trip(tmp_path: Path):
    model = Tokenizer.load(ROOT / "data" / "gpt2")
    assert model.vocab_size == 50257
    assert len(model.merges()) == 50000
    assert model.special_tokens() == ["<|endoftext|>"]
    fixture = json.loads((ROOT / "data" / "fixtures" / "gpt2_parity.json").read_text(encoding="utf-8"))
    for sample in fixture["samples"]:
        assert model.encode(sample["text"]) == sample["ids"]
        assert model.encode_ordinary(sample["text"]) == sample["ordinary"]
        assert model.decode(sample["ids"]) == sample["text"]

    trained = Tokenizer.train("low low low wider", vocab_size=280, special_tokens=["<|endoftext|>"])
    trained.save(tmp_path)
    loaded = Tokenizer.load(tmp_path)
    assert loaded.merges() == trained.merges()
    assert loaded.encode("low wider <|endoftext|>") == trained.encode("low wider <|endoftext|>")
