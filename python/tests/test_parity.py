"""The committed fixture must still be what tiktoken produces."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

tiktoken = pytest.importorskip("tiktoken")

ROOT = Path(__file__).resolve().parents[2]


def _hash_ids(ids: list[int]) -> str:
    digest = hashlib.sha256()
    for ident in ids:
        digest.update(struct.pack("<I", ident))
    return digest.hexdigest()


def test_fixture_matches_installed_tiktoken():
    enc = tiktoken.get_encoding("gpt2")
    fixture = json.loads((ROOT / "data" / "fixtures" / "gpt2_parity.json").read_text(encoding="utf-8"))
    for sample in fixture["samples"]:
        assert enc.encode(sample["text"], allowed_special="all") == sample["ids"]
        assert enc.encode_ordinary(sample["text"]) == sample["ordinary"]
    for corpus in fixture["corpora"]:
        text = (ROOT / "data" / "corpora" / corpus["file"]).read_bytes().decode("utf-8")
        ids = enc.encode(text, allowed_special="all")
        assert len(ids) == corpus["tokens"]
        assert ids[: len(corpus["head_ids"])] == corpus["head_ids"]
        assert _hash_ids(ids) == corpus["ids_sha256"]
