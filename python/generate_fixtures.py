"""Write tiktoken GPT-2 expectations used by the Rust and Python tests.

The fixture is the reference. Implementations are checked against it, and a
separate test checks the fixture still matches the installed tiktoken.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
CORPORA = ROOT / "data" / "corpora"
OUT = ROOT / "data" / "fixtures" / "gpt2_parity.json"

SAMPLES = [
    "",
    "a",
    "hello world",
    "Hello, world!",
    "  leading",
    "trailing  ",
    "double  space",
    "line\nbreak",
    "it's a test don't you're I'll",
    "café",
    "日本語",
    "emoji 😀",
    "<|endoftext|>",
    "before <|endoftext|> after",
    "numbers 12345",
    "!!!",
    "\n\n",
    "hello\n\nworld",
    "hello\n\n",
    "a\u0301",
    "\u00a0nbsp",
    "  ",
    "the 19th century",
    "hello \n world",
    " \n\n ",
    "emoji👨‍👩‍👧‍👦family",
    "dollar $5.00",
    "#### hash ## tag",
    "path/to/file.txt",
]


def hash_ids(ids: list[int]) -> str:
    digest = hashlib.sha256()
    for ident in ids:
        digest.update(struct.pack("<I", ident))
    return digest.hexdigest()


def main() -> None:
    enc = tiktoken.get_encoding("gpt2")
    samples = []
    for text in SAMPLES:
        samples.append(
            {
                "text": text,
                "ids": enc.encode(text, allowed_special="all"),
                "ordinary": enc.encode_ordinary(text),
            }
        )
    corpora = []
    for path in sorted(CORPORA.glob("*.txt")):
        data = path.read_bytes()
        text = data.decode("utf-8")
        ids = enc.encode(text, allowed_special="all")
        corpora.append(
            {
                "file": path.name,
                "file_sha256": hashlib.sha256(data).hexdigest(),
                "tokens": len(ids),
                "ids_sha256": hash_ids(ids),
                "head_ids": ids[:16],
            }
        )
        print(f"{path.name}: {len(ids)} tokens")
    payload = {
        "source": f"tiktoken {tiktoken.__version__} gpt2, allowed_special=all and encode_ordinary",
        "samples": samples,
        "corpora": corpora,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
