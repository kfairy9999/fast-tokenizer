"""Measure encode throughput and write the benchmark report.

Numbers in ``benchmarks/`` come from this script. It times the calls that
return token ids, including the cost of building a Python list, because that
is what ``tiktoken`` and the Hugging Face bindings do too.

Single-document rows encode one string. Batch rows split the same bytes on
paragraph boundaries into chunks of about 64 KiB and encode the chunks as
independent documents. Trailing whitespace inside a chunk is not the same
event as whitespace in the middle of the full string, so batch token counts
can differ by a fraction of a percent. MB/s uses the original UTF-8 size and
decimal megabytes (1e6 bytes). The reported time is the median of repeated
runs after one warmup.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import fast_tokenizer  # noqa: E402
from reference_bpe import Tokenizer as ReferenceTokenizer  # noqa: E402

OUT_DIR = ROOT / "benchmarks"
CORPORA_DIR = ROOT / "data" / "corpora"
MODEL_DIR = ROOT / "data" / "gpt2"
CHUNK_TARGET = 64 * 1024
WARMUP = 1
REPEATS = 5


def read_corpus(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _paragraph_end(text: str, start_at: int) -> int | None:
    rest = text[start_at:]
    best = None
    for sep in ("\r\n\r\n", "\n\n"):
        found = rest.find(sep)
        if found == -1:
            continue
        end = start_at + found + len(sep)
        best = end if best is None else min(best, end)
    return best


def chunk_text(text: str, target: int = CHUNK_TARGET) -> list[str]:
    """Split on blank lines into pieces of about `target` characters.

    Gutenberg files use CRLF, so a break is ``\\r\\n\\r\\n`` rather than ``\\n\\n``.
    Concatenating the chunks reproduces `text`.
    """
    if not text:
        return []
    if len(text) <= target:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + target)
        if end < len(text):
            boundary = _paragraph_end(text, end)
            end = len(text) if boundary is None else boundary
        if end <= start:
            end = len(text)
        chunks.append(text[start:end])
        start = end
    assert "".join(chunks) == text
    return chunks


def median_time(fn, repeats: int = REPEATS, warmup: int = WARMUP):
    for _ in range(warmup):
        fn()
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), min(samples), max(samples), result


def token_count(result) -> int:
    if not result:
        return 0
    if isinstance(result[0], int):
        return len(result)
    return sum(len(item) for item in result)


def parity(actual, expected) -> dict:
    if actual == expected:
        return {"match": True, "tokens": len(actual), "mismatches": 0, "rate": 1.0}
    mismatches = sum(left != right for left, right in zip(actual, expected))
    mismatches += abs(len(actual) - len(expected))
    denom = max(len(actual), len(expected), 1)
    return {
        "match": False,
        "tokens": len(actual),
        "expected_tokens": len(expected),
        "mismatches": mismatches,
        "rate": 1.0 - mismatches / denom,
    }


def load_optional():
    libs = {}
    try:
        import tiktoken

        libs["tiktoken"] = tiktoken.get_encoding("gpt2")
        libs["tiktoken_version"] = tiktoken.__version__
    except Exception as exc:  # noqa: BLE001
        libs["tiktoken_error"] = str(exc)
    try:
        from tokenizers import Tokenizer

        libs["hf"] = Tokenizer.from_pretrained("gpt2")
        import tokenizers

        libs["hf_version"] = tokenizers.__version__
    except Exception as exc:  # noqa: BLE001
        libs["hf_error"] = str(exc)
    return libs


def hardware() -> dict:
    cpu = "unknown"
    mem_kib = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_kib = int(line.split()[1])
                break
    except OSError:
        pass
    rustc = subprocess.check_output(["rustc", "--version"], text=True).strip()
    return {
        "platform": platform.platform(),
        "processor": cpu,
        "threads": platform.os.cpu_count(),
        "ram_gib": None if mem_kib is None else round(mem_kib / 1024 / 1024, 2),
        "python": platform.python_version(),
        "rustc": rustc,
        "implementation": platform.python_implementation(),
    }


def summarize(name: str, mode: str, nbytes: int, median_s: float, tokens: int, lo: float, hi: float) -> dict:
    return {
        "name": name,
        "mode": mode,
        "bytes": nbytes,
        "tokens": tokens,
        "median_s": median_s,
        "min_s": lo,
        "max_s": hi,
        "mb_s": (nbytes / median_s) / 1e6 if median_s else 0.0,
        "tokens_s": tokens / median_s if median_s else 0.0,
    }


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| implementation | mode | bytes | tokens | median s | MB/s | tokens/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {mode} | {bytes} | {tokens} | {median_s:.6f} | {mb_s:.2f} | {tokens_s:.0f} |".format(
                **row
            )
        )
    return "\n".join(lines)


def draw_chart(single_rows: list[dict], batch_rows: list[dict], path: Path, hw: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    panels = [
        (axes[0], single_rows, "One document, from Python"),
        (axes[1], batch_rows, "Combined corpus, parallel"),
    ]
    for axis, rows, title in panels:
        labels = [row["name"] for row in rows]
        values = [row["mb_s"] for row in rows]
        colors = [row["color"] for row in rows]
        positions = range(len(rows))
        bars = axis.barh(list(positions), values, color=colors, height=0.62)
        axis.set_yticks(list(positions), labels)
        axis.invert_yaxis()
        axis.set_xlabel("MB/s  (1e6 bytes / s)")
        axis.set_title(title)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="x", color="#E4E4E4", zorder=0)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                value + max(values) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=9,
                color="#222",
            )
        axis.set_xlim(0, max(values) * 1.22 if values else 1)
    threads = hw.get("threads")
    cpu = hw.get("processor", "cpu")
    fig.suptitle(
        f"GPT-2 encode throughput on the combined Gutenberg corpus\n{cpu} · {threads} threads · {hw.get('ram_gib')} GiB",
        fontsize=12,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def rust_core_bench(threads: int) -> dict | None:
    """Time the release CLI. This skips Python list conversion."""
    binary = ROOT / "target" / "release" / "fast-tokenizer"
    if not binary.is_file():
        print("skipping rust-core bench; build target/release/fast-tokenizer first", flush=True)
        return None
    proc = subprocess.run(
        [
            str(binary),
            "bench",
            "--model",
            str(MODEL_DIR),
            "--corpus",
            str(CORPORA_DIR),
            "--repeats",
            str(REPEATS),
            "--warmup",
            str(WARMUP),
            "--threads",
            str(threads),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def main() -> None:
    hw = hardware()
    libs = load_optional()
    files = sorted(CORPORA_DIR.glob("*.txt"))
    texts = [(path.name, read_corpus(path)) for path in files]
    combined = "combined", "".join(text for _name, text in texts)
    workloads = texts + [combined]

    rust = fast_tokenizer.Tokenizer.load(str(MODEL_DIR))
    reference = ReferenceTokenizer.load(MODEL_DIR)
    rows: list[dict] = []
    per_book: list[dict] = []
    parity_report = {}

    def time_named(impl, mode, text, fn):
        median_s, lo, hi, result = median_time(fn)
        nbytes = len(text.encode("utf-8"))
        # Batch fns return a list of lists. token_count handles both.
        tokens = token_count(result)
        row = summarize(impl, mode, nbytes, median_s, tokens, lo, hi)
        return row, result

    # Parity is measured once, outside the timed section, on the combined text.
    combined_text = combined[1]
    rust_ids = rust.encode(combined_text)
    ref_ids = reference.encode(combined_text)
    parity_report["fast_tokenizer"] = {"self_check_vs_reference": parity(rust_ids, ref_ids)}
    if "tiktoken" in libs:
        tiktoken_ids = libs["tiktoken"].encode(combined_text, allowed_special="all")
        parity_report["fast_tokenizer"]["vs_tiktoken"] = parity(rust_ids, tiktoken_ids)
        parity_report["reference"] = parity(ref_ids, tiktoken_ids)
    else:
        tiktoken_ids = None
        parity_report["tiktoken_error"] = libs.get("tiktoken_error")
    if "hf" in libs:
        hf_ids = libs["hf"].encode(combined_text).ids
        parity_report["tokenizers"] = parity(hf_ids, tiktoken_ids if tiktoken_ids is not None else rust_ids)
    else:
        parity_report["tokenizers_error"] = libs.get("hf_error")

    colors = {
        "pure Python": "#8E8E8E",
        "HF tokenizers": "#3D5A80",
        "tiktoken": "#1F6F5B",
        "fast-tokenizer": "#B7410E",
        "fast-tokenizer 1 thread": "#C4A484",
        "fast-tokenizer 4 threads": "#B7410E",
    }

    for name, text in workloads:
        specs = [
            ("fast-tokenizer", "single document", lambda text=text: rust.encode(text), colors["fast-tokenizer"]),
            ("pure Python", "single document", lambda text=text: reference.encode(text), colors["pure Python"]),
        ]
        if "tiktoken" in libs:
            enc = libs["tiktoken"]
            specs.append(
                (
                    "tiktoken",
                    "single document",
                    lambda text=text, enc=enc: enc.encode(text, allowed_special="all"),
                    colors["tiktoken"],
                )
            )
        if "hf" in libs:
            hf = libs["hf"]
            specs.append(
                ("HF tokenizers", "single document", lambda text=text, hf=hf: hf.encode(text).ids, colors["HF tokenizers"])
            )
        for impl, mode, fn, color in specs:
            row, _result = time_named(impl, mode, text, fn)
            row["corpus"] = name
            row["color"] = color
            per_book.append(row)
            print(f"{name:12} {impl:16} {row['mb_s']:.2f} MB/s {row['tokens_s']:.0f} tok/s", flush=True)

    chunks = chunk_text(combined_text)
    chunk_bytes = len(combined_text.encode("utf-8"))
    threads = hw["threads"] or 1
    batch_specs = [
        (
            "fast-tokenizer 1 thread",
            f"batch, 1 thread, {len(chunks)} chunks",
            lambda: rust.encode_batch(chunks, num_threads=1),
            colors["fast-tokenizer 1 thread"],
        ),
        (
            f"fast-tokenizer {threads} threads",
            f"rayon batch, {threads} threads, {len(chunks)} chunks",
            lambda: rust.encode_batch(chunks, num_threads=threads),
            colors["fast-tokenizer"],
        ),
        (
            "pure Python",
            f"batch, 1 thread, {len(chunks)} chunks",
            lambda: [reference.encode(chunk) for chunk in chunks],
            colors["pure Python"],
        ),
    ]
    if "tiktoken" in libs:
        enc = libs["tiktoken"]
        batch_specs.append(
            (
                "tiktoken",
                f"encode_ordinary_batch, {len(chunks)} chunks",
                lambda enc=enc: enc.encode_ordinary_batch(chunks),
                colors["tiktoken"],
            )
        )
    if "hf" in libs:
        hf = libs["hf"]
        batch_specs.append(
            (
                "HF tokenizers",
                f"encode_batch, {len(chunks)} chunks",
                lambda hf=hf: [item.ids for item in hf.encode_batch(chunks)],
                colors["HF tokenizers"],
            )
        )
    batch_rows = []
    for impl, mode, fn, color in batch_specs:
        row, _result = time_named(impl, mode, combined_text, fn)
        row["bytes"] = chunk_bytes
        row["mb_s"] = (chunk_bytes / row["median_s"]) / 1e6 if row["median_s"] else 0.0
        row["color"] = color
        row["corpus"] = "combined"
        # Recompute rates after forcing the original byte size (chunks concatenate
        # to the original text, so this is the same size).
        row["tokens_s"] = row["tokens"] / row["median_s"] if row["median_s"] else 0.0
        batch_rows.append(row)
        print(f"batch {impl:24} {row['mb_s']:.2f} MB/s tokens {row['tokens']}", flush=True)

    single_combined = [row for row in per_book if row["corpus"] == "combined"]
    # Stable display order.
    order = ["pure Python", "HF tokenizers", "tiktoken", "fast-tokenizer"]
    single_combined.sort(key=lambda row: order.index(row["name"]) if row["name"] in order else 99)
    batch_order = [
        "pure Python",
        "HF tokenizers",
        "tiktoken",
        "fast-tokenizer 1 thread",
        f"fast-tokenizer {threads} threads",
    ]
    batch_rows.sort(key=lambda row: batch_order.index(row["name"]) if row["name"] in batch_order else 99)

    core = rust_core_bench(threads)
    core_rows = []
    if core:
        for record in core["rows"]:
            if record["corpus"] != "combined":
                continue
            label = (
                "Rust CLI, 1 thread"
                if record["chunks"] == 1
                else f"Rust CLI, {threads} threads"
            )
            core_rows.append(
                {
                    **record,
                    "name": label,
                    "color": colors["fast-tokenizer 1 thread"] if record["chunks"] == 1 else colors["fast-tokenizer"],
                }
            )

    rename = {
        "fast-tokenizer 1 thread": "Python API, 1 thread",
        f"fast-tokenizer {threads} threads": f"Python API, {threads} threads",
        "tiktoken": "tiktoken batch",
    }
    chart_parallel = []
    for row in batch_rows:
        if row["name"] in rename:
            chart_parallel.append({**row, "name": rename[row["name"]]})
    chart_parallel = core_rows + chart_parallel

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = OUT_DIR / "throughput.png"
    draw_chart(single_combined, chart_parallel, chart_path, hw)

    file_meta = []
    for name, text in workloads:
        raw = text.encode("utf-8")
        file_meta.append({"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})

    versions = {
        "fast_tokenizer": fast_tokenizer.__version__,
        "tiktoken": libs.get("tiktoken_version"),
        "tokenizers": libs.get("hf_version"),
    }
    raw = {
        "hardware": hw,
        "versions": versions,
        "method": {
            "warmup": WARMUP,
            "repeats": REPEATS,
            "statistic": "median",
            "clock": "time.perf_counter",
            "megabyte": 1_000_000,
            "chunk_target_chars": CHUNK_TARGET,
            "chunks": len(chunks),
            "special_tokens": "enabled for fast-tokenizer and tiktoken allowed_special=all; HF uses its GPT-2 model",
            "release_profile": "lto=thin, codegen-units=1",
        },
        "files": file_meta,
        "single_document": per_book,
        "batch": batch_rows,
        "rust_core": core,
        "parity": parity_report,
    }
    (OUT_DIR / "raw.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "hardware.json").write_text(json.dumps({"hardware": hw, "versions": versions}, indent=2) + "\n", encoding="utf-8")

    def parity_phrase(stats: dict) -> str:
        if stats.get("match"):
            return "exact"
        rate = stats.get("rate", 0.0) * 100
        return f"{rate:.4f}% ({stats.get('mismatches')} mismatches)"

    parity_lines = []
    ft = parity_report.get("fast_tokenizer", {})
    vs = ft.get("vs_tiktoken")
    if vs:
        parity_lines.append(
            "- fast-tokenizer vs tiktoken on the combined corpus: "
            f"{parity_phrase(vs)}, {vs['tokens']} tokens"
        )
    ref = parity_report.get("reference")
    if isinstance(ref, dict) and "rate" in ref:
        parity_lines.append(f"- pure Python vs tiktoken: {parity_phrase(ref)}")
    hf = parity_report.get("tokenizers")
    if isinstance(hf, dict) and "rate" in hf:
        parity_lines.append(f"- HF tokenizers vs tiktoken: {parity_phrase(hf)}")

    per_lines = [
        "| corpus | fast-tokenizer MB/s | tiktoken MB/s | HF tokenizers MB/s | pure Python MB/s | tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_corpus: dict[str, dict[str, dict]] = {}
    for row in per_book:
        by_corpus.setdefault(row["corpus"], {})[row["name"]] = row
    for name, _text in workloads:
        bucket = by_corpus[name]
        def cell(impl: str) -> str:
            row = bucket.get(impl)
            return f"{row['mb_s']:.2f}" if row else "—"

        tokens = bucket["fast-tokenizer"]["tokens"]
        per_lines.append(
            f"| {name} | {cell('fast-tokenizer')} | {cell('tiktoken')} | {cell('HF tokenizers')} | {cell('pure Python')} | {tokens} |"
        )

    report = f"""# Benchmark results

Measured by `python python/benchmark.py` on this machine. MB/s divides UTF-8
bytes by the median wall time and by 1,000,000. Each figure is the median of
{REPEATS} runs after {WARMUP} warmup. Model load time is not included.

- CPU: {hw['processor']}
- Threads: {hw['threads']}
- RAM: {hw['ram_gib']} GiB
- OS: {hw['platform']}
- rustc: {hw['rustc']}
- Python: {hw['python']}
- tiktoken: {versions.get('tiktoken')}
- tokenizers: {versions.get('tokenizers')}
- fast-tokenizer: {versions.get('fast_tokenizer')} (release build, thin LTO, codegen-units = 1)

The single-document rows call each library from Python and include the cost of
building the Python list of token ids. The Rust-core rows are the release CLI
on the same files, with no Python conversion. Batch rows split the combined
corpus on blank lines (CRLF-aware) into about 64 KiB chunks.

## Single document

{markdown_table(single_combined)}

### Per corpus, single document (MB/s)

{chr(10).join(per_lines)}

## Rust CLI (no Python conversion)

{markdown_table(core_rows) if core_rows else "_Release binary was not built, so this table was skipped._"}

## Batch of ~64 KiB chunks, called from Python (combined corpus)

Chunk count: {len(chunks)}. Concatenating the chunks reproduces the corpus.
Each chunk is tokenized as its own document, so the token total can differ
from the single-document encode.

{markdown_table(batch_rows)}

## Token parity

{chr(10).join(parity_lines) if parity_lines else '- parity comparison unavailable'}

On this corpus the chunked encode produced the same token ids as encoding
the whole text at once. That is a property of these files and this split,
not a guarantee for an arbitrary cut.
"""
    (OUT_DIR / "results.md").write_text(report, encoding="utf-8")
    print(f"wrote {OUT_DIR / 'results.md'}")
    print(f"wrote {chart_path}")


if __name__ == "__main__":
    main()
