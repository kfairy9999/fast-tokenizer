"""Pure-Python byte-level BPE, using the same rules as the Rust core.

The implementation is the correctness reference for ``fast_tokenizer``. It is
deliberately straightforward: a scanner for the GPT-2 pretokenizer, a pair
counter, and the min-rank merge loop. A cache inside one ``encode`` call
avoids repeating work for a pretoken that shows up again in the same text,
which is also what the Rust encoder does.

Encode and train are different passes on purpose.

* Training repeatedly selects the most frequent pair. Ties break toward the
  lexicographically smaller pair of raw byte strings. Every non-overlapping
  occurrence in a word is replaced in that same pass.
* Encode never looks at frequencies. It merges the adjacent pair with the
  lowest rank (the order written in ``merges.txt``), one pair at a time.

The on-disk format is GPT-2's ``vocab.json`` plus ``merges.txt``.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

# Unicode White_Space. Kept explicit so the splitter does not drift with the
# Python build's Unicode tables for the characters GPT-2 actually cares about.
_WHITESPACE = set(
    [
        0x0009,
        0x000A,
        0x000B,
        0x000C,
        0x000D,
        0x0020,
        0x0085,
        0x00A0,
        0x1680,
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
    ]
    + list(range(0x2000, 0x200B))
)


def _is_ws(ch: str) -> bool:
    return ord(ch) in _WHITESPACE


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_number(ch: str) -> bool:
    return unicodedata.category(ch).startswith("N")


def _is_other(ch: str) -> bool:
    return not _is_ws(ch) and not _is_letter(ch) and not _is_number(ch)


def pretokenize(text: str) -> list[str]:
    """Split ``text`` the way the GPT-2 / tiktoken r50k regex does."""
    if not text:
        return []
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = _match_contraction(text, i)
        if end is None:
            end = _match_class(text, i, _is_letter)
        if end is None:
            end = _match_class(text, i, _is_number)
        if end is None:
            end = _match_class(text, i, _is_other)
        if end is None:
            end = _match_ws(text, i)
        if end is None or end <= i:
            raise RuntimeError(f"pretokenizer stuck at index {i}")
        out.append(text[i:end])
        i = end
    return out


def _match_contraction(text: str, i: int) -> int | None:
    if text[i] != "'":
        return None
    if i + 1 < len(text) and text[i + 1] in "sdmt":
        return i + 2
    pair = text[i + 1 : i + 3]
    if pair in ("ll", "ve", "re"):
        return i + 3
    return None


def _match_class(text: str, i: int, pred) -> int | None:
    n = len(text)
    if text[i] == " " and i + 1 < n and pred(text[i + 1]):
        start = i + 1
    elif pred(text[i]):
        start = i
    else:
        return None
    j = start + 1
    while j < n and pred(text[j]):
        j += 1
    return j


def _match_ws(text: str, i: int) -> int | None:
    n = len(text)
    if not _is_ws(text[i]):
        return None
    j = i + 1
    while j < n and _is_ws(text[j]):
        j += 1
    if j == n:
        return j
    if j - i >= 2:
        return j - 1
    return j


def _byte_tables() -> tuple[dict[int, str], list[int]]:
    """GPT-2 byte display characters and the canonical id of each byte."""
    ordered = list(range(ord("!"), ord("~") + 1))
    ordered += list(range(0xA1, 0xAD))
    ordered += list(range(0xAE, 0x100))
    present = set(ordered)
    base = len(ordered)
    extras: list[int] = []
    for byte in range(256):
        if byte not in present:
            extras.append(byte)
    encode: dict[int, str] = {}
    byte_to_id = [0] * 256
    for index, byte in enumerate(ordered):
        encode[byte] = chr(byte)
        byte_to_id[byte] = index
    for offset, byte in enumerate(extras):
        encode[byte] = chr(256 + offset)
        byte_to_id[byte] = base + offset
    return encode, byte_to_id


_ENCODE, _BYTE_TO_ID = _byte_tables()


def _to_unicode(data: bytes) -> str:
    return "".join(_ENCODE[byte] for byte in data)


class Tokenizer:
    """A trained or loaded byte-level BPE model."""

    def __init__(self) -> None:
        self.byte_to_id = [0] * 256
        self.id_to_bytes: list[bytes] = []
        self.id_to_token: list[str] = []
        self.token_to_id: dict[str, int] = {}
        self._merges: dict[tuple[int, int], tuple[int, int]] = {}
        self._merge_pairs: list[tuple[str, str]] = []
        self._specials_in_order: list[tuple[str, int]] = []
        self._specials_match: list[tuple[str, int]] = []

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 8000,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        specials = list(special_tokens or [])
        _validate_specials(specials)
        minimum = 256 + len(specials)
        if vocab_size < minimum:
            raise ValueError(
                f"vocab_size {vocab_size} is below the minimum {minimum} "
                f"(256 byte tokens + {len(specials)} special tokens)"
            )
        n_merges = vocab_size - minimum
        words = _count_words(text, specials)
        id_to_bytes: list[bytes] = [b""] * 256
        for byte in range(256):
            id_to_bytes[_BYTE_TO_ID[byte]] = bytes([byte])
        learned = _learn(words, id_to_bytes, n_merges)
        id_to_token = [_to_unicode(piece) for piece in id_to_bytes]
        merge_pairs = [
            (_to_unicode(id_to_bytes[left]), _to_unicode(id_to_bytes[right]))
            for left, right in learned
        ]
        return cls._assemble(id_to_bytes, id_to_token, merge_pairs, specials, list(_BYTE_TO_ID))

    @classmethod
    def from_gpt2_files(cls, vocab_path: str | Path, merges_path: str | Path) -> Tokenizer:
        vocab_path = Path(vocab_path)
        merges_path = Path(merges_path)
        vocab: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
        if not vocab:
            raise ValueError("vocabulary is empty")
        max_id = max(vocab.values())
        if len(vocab) != max_id + 1 or set(vocab.values()) != set(range(max_id + 1)):
            raise ValueError("token ids are not a contiguous range starting at 0")
        merge_pairs = _parse_merges(merges_path.read_text(encoding="utf-8"))
        id_to_token = [""] * len(vocab)
        for token, ident in vocab.items():
            id_to_token[ident] = token
        byte_to_id = [0] * 256
        id_to_bytes: list[bytes] = [b""] * len(vocab)
        for byte in range(256):
            display = _ENCODE[byte]
            if display not in vocab:
                raise ValueError(f"missing byte token for 0x{byte:02x}")
            ident = vocab[display]
            byte_to_id[byte] = ident
            id_to_bytes[ident] = bytes([byte])
        for left, right in merge_pairs:
            if left not in vocab or right not in vocab:
                raise ValueError(f"merge references unknown token {left!r} {right!r}")
            merged = left + right
            if merged not in vocab:
                raise ValueError(f"merge result {merged!r} is missing from the vocabulary")
            left_id = vocab[left]
            right_id = vocab[right]
            result_id = vocab[merged]
            if not id_to_bytes[result_id]:
                if not id_to_bytes[left_id] or not id_to_bytes[right_id]:
                    raise ValueError(f"merge {left!r} {right!r} is out of order")
                id_to_bytes[result_id] = id_to_bytes[left_id] + id_to_bytes[right_id]
        for token, ident in vocab.items():
            if not id_to_bytes[ident]:
                id_to_bytes[ident] = token.encode("utf-8")
        return cls._from_loaded(id_to_bytes, id_to_token, merge_pairs, byte_to_id)

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        directory = Path(path)
        vocab = _find(directory, ("vocab.json", "encoder.json"))
        merges = _find(directory, ("merges.txt", "vocab.bpe"))
        return cls.from_gpt2_files(vocab, merges)

    def save(self, path: str | Path) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self.token_to_id.items(), key=lambda item: item[1])
        payload = {token: ident for token, ident in ordered}
        vocab_text = json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
        (directory / "vocab.json").write_text(vocab_text, encoding="utf-8")
        lines = ["#version: 0.2"]
        lines.extend(f"{left} {right}" for left, right in self._merge_pairs)
        (directory / "merges.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def encode(self, text: str, special_tokens: bool = True) -> list[int]:
        cache: dict[str, list[int]] = {}
        ids: list[int] = []
        if not text:
            return ids
        if not special_tokens or not self._specials_match:
            self._encode_ordinary(text, cache, ids)
            return ids
        cursor = 0
        for start, end, ident in _special_spans(text, self._specials_match):
            if start > cursor:
                self._encode_ordinary(text[cursor:start], cache, ids)
            ids.append(ident)
            cursor = end
        if cursor < len(text):
            self._encode_ordinary(text[cursor:], cache, ids)
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        return self.encode(text, special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        for ident in ids:
            if ident < 0 or ident >= len(self.id_to_bytes):
                raise ValueError(f"unknown token id {ident}")
            out += self.id_to_bytes[ident]
        return out.decode("utf-8", errors="replace")

    def merges(self) -> list[tuple[str, str]]:
        return list(self._merge_pairs)

    def special_tokens(self) -> list[str]:
        return [token for token, _ident in self._specials_in_order]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def __repr__(self) -> str:
        return (
            f"Tokenizer(vocab_size={self.vocab_size}, merges={len(self._merge_pairs)}, "
            f"specials={len(self._specials_in_order)})"
        )

    def _encode_ordinary(self, text: str, cache: dict[str, list[int]], out: list[int]) -> None:
        for part in pretokenize(text):
            cached = cache.get(part)
            if cached is None:
                cached = self._bpe(part.encode("utf-8"))
                cache[part] = cached
            out.extend(cached)

    def _bpe(self, data: bytes) -> list[int]:
        ids = [self.byte_to_id[byte] for byte in data]
        merges = self._merges
        while len(ids) >= 2:
            best_rank = None
            best_index = 0
            best_result = 0
            for index in range(len(ids) - 1):
                found = merges.get((ids[index], ids[index + 1]))
                if found is not None and (best_rank is None or found[0] < best_rank):
                    best_rank = found[0]
                    best_index = index
                    best_result = found[1]
            if best_rank is None:
                break
            ids[best_index] = best_result
            del ids[best_index + 1]
        return ids

    @classmethod
    def _assemble(
        cls,
        id_to_bytes: list[bytes],
        id_to_token: list[str],
        merge_pairs: list[tuple[str, str]],
        specials: list[str],
        byte_to_id: list[int],
    ) -> Tokenizer:
        """Build a model trained in memory. Special tokens are appended."""
        token_to_id = {token: ident for ident, token in enumerate(id_to_token)}
        for special in specials:
            if special in token_to_id:
                raise ValueError(
                    f"special token {special!r} collides with a byte or merge token"
                )
        id_to_bytes = list(id_to_bytes)
        id_to_token = list(id_to_token)
        specials_in_order: list[tuple[str, int]] = []
        for special in specials:
            ident = len(id_to_token)
            token_to_id[special] = ident
            id_to_bytes.append(special.encode("utf-8"))
            id_to_token.append(special)
            specials_in_order.append((special, ident))
        merges: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (left, right) in enumerate(merge_pairs):
            merges[(token_to_id[left], token_to_id[right])] = (rank, token_to_id[left + right])
        model = cls()
        model.byte_to_id = byte_to_id
        model.id_to_bytes = id_to_bytes
        model.id_to_token = id_to_token
        model.token_to_id = token_to_id
        model._merges = merges
        model._merge_pairs = list(merge_pairs)
        model._specials_in_order = specials_in_order
        model._specials_match = sorted(specials_in_order, key=lambda item: (-len(item[0]), item[1]))
        return model

    @classmethod
    def _from_loaded(
        cls,
        id_to_bytes: list[bytes],
        id_to_token: list[str],
        merge_pairs: list[tuple[str, str]],
        byte_to_id: list[int],
    ) -> Tokenizer:
        model = cls()
        token_to_id = {token: ident for ident, token in enumerate(id_to_token)}
        merges: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (left, right) in enumerate(merge_pairs):
            merges[(token_to_id[left], token_to_id[right])] = (rank, token_to_id[left + right])
        derived = set(_ENCODE.values())
        for left, right in merge_pairs:
            derived.add(left + right)
        specials = [(token, ident) for token, ident in token_to_id.items() if token not in derived]
        specials.sort(key=lambda item: item[1])
        model.byte_to_id = list(byte_to_id)
        model.id_to_bytes = id_to_bytes
        model.id_to_token = id_to_token
        model.token_to_id = token_to_id
        model._merges = merges
        model._merge_pairs = list(merge_pairs)
        model._specials_in_order = specials
        model._specials_match = sorted(specials, key=lambda item: (-len(item[0]), item[1]))
        return model


def _validate_specials(specials: list[str]) -> None:
    seen: set[str] = set()
    for token in specials:
        if token == "":
            raise ValueError("special token must be non-empty")
        if token in seen:
            raise ValueError(f"duplicate special token {token!r}")
        seen.add(token)


def _parse_merges(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    if lines and lines[0].startswith("#version"):
        lines = lines[1:]
    pairs: list[tuple[str, str]] = []
    for index, line in enumerate(lines, start=1):
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"line {index}: expected exactly two tokens")
        pairs.append((parts[0], parts[1]))
    return pairs


def _find(directory: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"tokenizer files not found in {directory}")


def _special_spans(text: str, specials: list[tuple[str, int]]) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    if not specials or not text:
        return spans
    i = 0
    n = len(text)
    while i < n:
        best: tuple[int, int, int] | None = None
        for token, ident in specials:
            if not token:
                continue
            found = text.find(token, i)
            if found == -1:
                continue
            end = found + len(token)
            if best is None or found < best[0] or (found == best[0] and end > best[1]):
                best = (found, end, ident)
        if best is None:
            break
        spans.append(best)
        i = best[1]
    return spans


def _count_words(text: str, specials: list[str]) -> list[list]:
    tagged = [(token, index) for index, token in enumerate(specials)]
    spans = _special_spans(text, tagged)
    counts: Counter[str] = Counter()
    cursor = 0
    for start, end, _ident in spans:
        if start > cursor:
            counts.update(pretokenize(text[cursor:start]))
        cursor = end
    if cursor < len(text):
        counts.update(pretokenize(text[cursor:]))
    words: list[list] = []
    for part, freq in counts.items():
        symbols = [_BYTE_TO_ID[byte] for byte in part.encode("utf-8")]
        words.append([symbols, freq])
    return words


def _learn(words: list[list], id_to_bytes: list[bytes], n_merges: int) -> list[tuple[int, int]]:
    learned: list[tuple[int, int]] = []
    for _step in range(n_merges):
        stats: dict[tuple[int, int], int] = {}
        for symbols, freq in words:
            for left, right in zip(symbols, symbols[1:]):
                pair = (left, right)
                stats[pair] = stats.get(pair, 0) + freq
        best = _best_pair(stats, id_to_bytes)
        if best is None:
            break
        left, right = best
        new_id = len(id_to_bytes)
        id_to_bytes.append(id_to_bytes[left] + id_to_bytes[right])
        for word in words:
            symbols = word[0]
            if not any(symbols[i] == left and symbols[i + 1] == right for i in range(len(symbols) - 1)):
                continue
            word[0] = _rewrite(symbols, left, right, new_id)
        learned.append((left, right))
    return learned


def _best_pair(
    stats: dict[tuple[int, int], int], id_to_bytes: list[bytes]
) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    best_key: tuple[int, bytes, bytes] | None = None
    for (left, right), count in stats.items():
        if count <= 0:
            continue
        key = (-count, id_to_bytes[left], id_to_bytes[right])
        if best_key is None or key < best_key:
            best_key = key
            best = (left, right)
    return best


def _rewrite(symbols: list[int], left: int, right: int, new_id: int) -> list[int]:
    out: list[int] = []
    index = 0
    n = len(symbols)
    while index < n:
        if index + 1 < n and symbols[index] == left and symbols[index + 1] == right:
            out.append(new_id)
            index += 2
        else:
            out.append(symbols[index])
            index += 1
    return out
