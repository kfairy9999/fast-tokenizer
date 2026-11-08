# Benchmark results

Measured by `python python/benchmark.py` on this machine. MB/s divides UTF-8
bytes by the median wall time and by 1,000,000. Each figure is the median of
5 runs after 1 warmup. Model load time is not included.

- CPU: Intel(R) Xeon(R) Processor
- Threads: 4
- RAM: 15.63 GiB
- OS: Linux-6.12.113+-x86_64-with-glibc2.39
- rustc: rustc 1.99.1 (58cf8f9c3 2026-10-13)
- Python: 3.12.3
- tiktoken: 0.14.0
- tokenizers: 0.23.3
- fast-tokenizer: 0.1.0 (release build, thin LTO, codegen-units = 1)

The single-document rows call each library from Python and include the cost of
building the Python list of token ids. The Rust-core rows are the release CLI
on the same files, with no Python conversion. Batch rows split the combined
corpus on blank lines (CRLF-aware) into about 64 KiB chunks.

## Single document

| implementation | mode | bytes | tokens | median s | MB/s | tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pure Python | single document | 3279503 | 905180 | 0.901827 | 3.64 | 1003718 |
| HF tokenizers | single document | 3279503 | 905180 | 1.466438 | 2.24 | 617264 |
| tiktoken | single document | 3279503 | 905180 | 0.167277 | 19.61 | 5411275 |
| fast-tokenizer | single document | 3279503 | 905180 | 0.067537 | 48.56 | 13402683 |

### Per corpus, single document (MB/s)

| corpus | fast-tokenizer MB/s | tiktoken MB/s | HF tokenizers MB/s | pure Python MB/s | tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| pg11.txt | 38.83 | 17.64 | 3.47 | 3.11 | 52948 |
| pg1342.txt | 46.80 | 19.96 | 2.91 | 3.15 | 211562 |
| pg1661.txt | 47.27 | 19.59 | 3.16 | 3.42 | 172125 |
| pg2701.txt | 42.96 | 17.69 | 2.64 | 3.30 | 354334 |
| pg84.txt | 43.71 | 20.72 | 3.52 | 3.31 | 114211 |
| combined | 48.56 | 19.61 | 2.24 | 3.64 | 905180 |

## Rust CLI (no Python conversion)

| implementation | mode | bytes | tokens | median s | MB/s | tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Rust CLI, 1 thread | single-thread | 3279503 | 905180 | 0.057431 | 57.10 | 15761137 |
| Rust CLI, 4 threads | rayon batch (4 threads, 50 chunks) | 3279503 | 905180 | 0.021063 | 155.70 | 42974809 |

## Batch of ~64 KiB chunks, called from Python (combined corpus)

Chunk count: 49. Concatenating the chunks reproduces the corpus.
Each chunk is tokenized as its own document, so the token total can differ
from the single-document encode.

| implementation | mode | bytes | tokens | median s | MB/s | tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pure Python | batch, 1 thread, 49 chunks | 3279503 | 905180 | 1.363143 | 2.41 | 664039 |
| HF tokenizers | encode_batch, 49 chunks | 3279503 | 905180 | 0.335664 | 9.77 | 2696681 |
| tiktoken | encode_ordinary_batch, 49 chunks | 3279503 | 905180 | 0.079486 | 41.26 | 11387926 |
| fast-tokenizer 1 thread | batch, 1 thread, 49 chunks | 3279503 | 905180 | 0.113049 | 29.01 | 8006950 |
| fast-tokenizer 4 threads | rayon batch, 4 threads, 49 chunks | 3279503 | 905180 | 0.036884 | 88.91 | 24541111 |

## Token parity

- fast-tokenizer vs tiktoken on the combined corpus: exact, 905180 tokens
- pure Python vs tiktoken: exact
- HF tokenizers vs tiktoken: exact

On this corpus the 49-chunk encode produced the same token ids as encoding
the whole text at once (905,180 tokens). That is a property of these files
and this split, not a guarantee for every boundary.
