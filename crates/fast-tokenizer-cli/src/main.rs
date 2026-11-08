//! `fast-tokenizer encode|decode|train|bench`

use std::error::Error;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, Subcommand};
use fast_tokenizer::Tokenizer;

#[derive(Parser)]
#[command(
    name = "fast-tokenizer",
    version,
    about = "Train and run a byte-level BPE tokenizer"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a byte-level BPE model and write vocab.json + merges.txt.
    Train {
        /// UTF-8 corpus to learn from.
        #[arg(long)]
        input: PathBuf,
        /// Vocabulary size, counting byte tokens, merges, and special tokens.
        #[arg(long, default_value_t = 8000)]
        vocab_size: usize,
        /// Directory to write the model into.
        #[arg(long)]
        output: PathBuf,
        /// Atomic special token. Repeat the flag for more than one.
        #[arg(long = "special")]
        special: Vec<String>,
    },
    /// Encode text to a JSON array of token ids.
    Encode {
        /// Model directory (vocab.json + merges.txt).
        #[arg(long)]
        model: PathBuf,
        /// Text to encode. Omit, with --file, to read stdin.
        #[arg(long)]
        text: Option<String>,
        /// UTF-8 file to encode.
        #[arg(long)]
        file: Option<PathBuf>,
        /// Do not treat special tokens as atomic.
        #[arg(long)]
        ordinary: bool,
    },
    /// Decode a JSON array of token ids.
    Decode {
        /// Model directory (vocab.json + merges.txt).
        #[arg(long)]
        model: PathBuf,
        /// Comma-separated ids, or a JSON array.
        #[arg(long)]
        ids: Option<String>,
        /// File containing a JSON array of token ids.
        #[arg(long)]
        file: Option<PathBuf>,
    },
    /// Measure single-thread and rayon batch encode throughput.
    Bench {
        /// Model directory (vocab.json + merges.txt).
        #[arg(long)]
        model: PathBuf,
        /// UTF-8 corpus file, or a directory of .txt files.
        #[arg(long)]
        corpus: PathBuf,
        /// Timed repetitions. The reported time is the median.
        #[arg(long, default_value_t = 5)]
        repeats: usize,
        /// Untimed warmup passes before measurement.
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// Threads for the batch row. 0 uses every core rayon sees.
        #[arg(long, default_value_t = 0)]
        threads: usize,
        /// Print a JSON object instead of a markdown table.
        #[arg(long)]
        json: bool,
    },
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    match Cli::parse().command {
        Command::Train {
            input,
            vocab_size,
            output,
            special,
        } => {
            let text = fs::read_to_string(&input)?;
            let refs: Vec<&str> = special.iter().map(String::as_str).collect();
            let started = Instant::now();
            let mut completed = 0usize;
            let model =
                Tokenizer::train_with_progress(&text, vocab_size, &refs, &mut |done, total| {
                    completed = done;
                    if done == total || done % 500 == 0 {
                        eprintln!("merge {done}/{total}");
                    }
                })?;
            model.save(&output)?;
            eprintln!(
                "trained vocab_size={} merges={} specials={} in {:.2}s (stopped at merge {completed}) -> {}",
                model.vocab_size(),
                model.merges().len(),
                model.special_tokens().count(),
                started.elapsed().as_secs_f64(),
                output.display()
            );
        }
        Command::Encode {
            model,
            text,
            file,
            ordinary,
        } => {
            let model = Tokenizer::load(&model)?;
            let text = read_text(text, file)?;
            let ids = if ordinary {
                model.encode_ordinary(&text)
            } else {
                model.encode(&text)
            };
            let stdout = io::stdout();
            let mut lock = stdout.lock();
            serde_json::to_writer(&mut lock, &ids)?;
            lock.write_all(b"\n")?;
        }
        Command::Decode { model, ids, file } => {
            let model = Tokenizer::load(&model)?;
            let raw = match (ids, file) {
                (Some(ids), None) => ids,
                (None, Some(path)) => fs::read_to_string(path)?,
                (None, None) => {
                    let mut buf = String::new();
                    io::stdin().read_to_string(&mut buf)?;
                    buf
                }
                (Some(_), Some(_)) => {
                    return Err("pass only one of --ids and --file".into());
                }
            };
            let parsed = parse_ids(&raw)?;
            let text = model.decode(&parsed)?;
            let mut stdout = io::stdout().lock();
            stdout.write_all(text.as_bytes())?;
        }
        Command::Bench {
            model,
            corpus,
            repeats,
            warmup,
            threads,
            json,
        } => {
            if repeats == 0 {
                return Err("--repeats must be at least 1".into());
            }
            let model = Tokenizer::load(&model)?;
            let jobs = corpora(&corpus)?;
            let threads = if threads == 0 {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            } else {
                threads
            };
            let mut rows = Vec::new();
            for (name, text) in &jobs {
                let single = time_single(&model, text, warmup, repeats);
                rows.push(row_record(name, "single-thread", text, &single, 1));
                let chunks = chunk_text(text, 64 * 1024);
                let batch = time_batch(&model, &chunks, warmup, repeats, threads);
                rows.push(row_record(
                    name,
                    &format!("rayon batch ({threads} threads, {} chunks)", chunks.len()),
                    text,
                    &batch,
                    chunks.len(),
                ));
            }
            if json {
                let payload = serde_json::json!({
                    "hardware": hardware_line(),
                    "vocab_size": model.vocab_size(),
                    "merges": model.merges().len(),
                    "warmup": warmup,
                    "repeats": repeats,
                    "threads": threads,
                    "rows": rows,
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else {
                println!("# fast-tokenizer bench\n");
                println!("{}", hardware_line());
                println!(
                    "model vocab_size={} merges={} | warmup={warmup} repeats={repeats} | batch threads={threads} | median of wall time | MB/s uses 1e6 bytes\n",
                    model.vocab_size(),
                    model.merges().len()
                );
                println!("| corpus | mode | bytes | tokens | median s | MB/s | tokens/s |");
                println!("| --- | --- | ---: | ---: | ---: | ---: | ---: |");
                for row in &rows {
                    println!(
                        "| {name} | {mode} | {bytes} | {tokens} | {median_s:.6} | {mb_s:.2} | {tokens_s:.0} |",
                        name = row["corpus"].as_str().unwrap_or(""),
                        mode = row["mode"].as_str().unwrap_or(""),
                        bytes = row["bytes"],
                        tokens = row["tokens"],
                        median_s = row["median_s"].as_f64().unwrap_or(0.0),
                        mb_s = row["mb_s"].as_f64().unwrap_or(0.0),
                        tokens_s = row["tokens_s"].as_f64().unwrap_or(0.0),
                    );
                }
            }
        }
    }
    Ok(())
}

struct Timing {
    median_s: f64,
    tokens: usize,
}

fn time_single(model: &Tokenizer, text: &str, warmup: usize, repeats: usize) -> Timing {
    for _ in 0..warmup {
        std::hint::black_box(model.encode(text));
    }
    let mut samples = Vec::with_capacity(repeats);
    let mut tokens = 0;
    for _ in 0..repeats {
        let started = Instant::now();
        let ids = model.encode(text);
        samples.push(started.elapsed().as_secs_f64());
        tokens = ids.len();
    }
    Timing {
        median_s: median(&mut samples),
        tokens,
    }
}

fn time_batch(
    model: &Tokenizer,
    chunks: &[&str],
    warmup: usize,
    repeats: usize,
    threads: usize,
) -> Timing {
    for _ in 0..warmup {
        std::hint::black_box(model.encode_batch(chunks, threads));
    }
    let mut samples = Vec::with_capacity(repeats);
    let mut tokens = 0;
    for _ in 0..repeats {
        let started = Instant::now();
        let ids = model.encode_batch(chunks, threads);
        samples.push(started.elapsed().as_secs_f64());
        tokens = ids.iter().map(Vec::len).sum();
    }
    Timing {
        median_s: median(&mut samples),
        tokens,
    }
}

fn row_record(
    name: &str,
    mode: &str,
    text: &str,
    timing: &Timing,
    chunks: usize,
) -> serde_json::Value {
    let bytes = text.len();
    let mb_s = if timing.median_s > 0.0 {
        (bytes as f64 / timing.median_s) / 1e6
    } else {
        0.0
    };
    let tokens_s = if timing.median_s > 0.0 {
        timing.tokens as f64 / timing.median_s
    } else {
        0.0
    };
    serde_json::json!({
        "corpus": name,
        "mode": mode,
        "bytes": bytes,
        "tokens": timing.tokens,
        "median_s": timing.median_s,
        "mb_s": mb_s,
        "tokens_s": tokens_s,
        "chunks": chunks,
    })
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mid = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[mid]
    } else {
        (samples[mid - 1] + samples[mid]) / 2.0
    }
}

/// Split on paragraph boundaries into chunks of about `target` bytes.
///
/// Concatenating the chunks reproduces `text`. A chunk is an independent
/// document: whitespace at its end is tokenized as trailing whitespace.
fn chunk_text(text: &str, target: usize) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    if text.len() <= target {
        return vec![text];
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + target).min(text.len());
        while end > start && !text.is_char_boundary(end) {
            end -= 1;
        }
        if end < text.len() {
            match next_paragraph_end(text, end) {
                Some(boundary) => end = boundary,
                None => end = text.len(),
            }
        }
        if end <= start {
            end = text.len();
        }
        chunks.push(&text[start..end]);
        start = end;
    }
    debug_assert_eq!(chunks.join(""), text);
    chunks
}

fn next_paragraph_end(text: &str, from: usize) -> Option<usize> {
    let rest = text.get(from..)?;
    let mut best: Option<usize> = None;
    for sep in ["\r\n\r\n", "\n\n"] {
        if let Some(rel) = rest.find(sep) {
            let end = from + rel + sep.len();
            best = Some(best.map_or(end, |current| current.min(end)));
        }
    }
    best
}

fn corpora(path: &Path) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    if path.is_file() {
        return Ok(vec![(
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("corpus")
                .to_string(),
            fs::read_to_string(path)?,
        )]);
    }
    if path.is_dir() {
        let mut files: Vec<PathBuf> = fs::read_dir(path)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()?;
        files.retain(|path| path.extension().and_then(|ext| ext.to_str()) == Some("txt"));
        files.sort();
        if files.is_empty() {
            return Err(format!("no .txt files in {}", path.display()).into());
        }
        let mut jobs = Vec::new();
        let mut combined = String::new();
        for file in files {
            let name = file
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("corpus")
                .to_string();
            let text = fs::read_to_string(&file)?;
            combined.push_str(&text);
            jobs.push((name, text));
        }
        jobs.push(("combined".to_string(), combined));
        return Ok(jobs);
    }
    Err(format!("corpus path not found: {}", path.display()).into())
}

fn read_text(text: Option<String>, file: Option<PathBuf>) -> Result<String, Box<dyn Error>> {
    match (text, file) {
        (Some(text), None) => Ok(text),
        (None, Some(path)) => Ok(fs::read_to_string(path)?),
        (None, None) => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            Ok(buf)
        }
        (Some(_), Some(_)) => Err("pass only one of --text and --file".into()),
    }
}

fn parse_ids(raw: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if trimmed.starts_with('[') {
        return Ok(serde_json::from_str(trimmed)?);
    }
    trimmed
        .split(',')
        .map(|piece| {
            piece
                .trim()
                .parse::<u32>()
                .map_err(|err| format!("invalid token id {piece:?}: {err}").into())
        })
        .collect()
}

fn hardware_line() -> String {
    let cpu = fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split(':').nth(1))
                .map(|name| name.trim().to_string())
        })
        .unwrap_or_else(|| "unknown cpu".to_string());
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mem_gib = fs::read_to_string("/proc/meminfo").ok().and_then(|text| {
        text.lines().find_map(|line| {
            let rest = line.strip_prefix("MemTotal:")?;
            let kb: f64 = rest.split_whitespace().next()?.parse().ok()?;
            Some(kb / 1024.0 / 1024.0)
        })
    });
    match mem_gib {
        Some(mem) => format!("hardware: {cpu} | {cores} threads | {mem:.1} GiB RAM"),
        None => format!("hardware: {cpu} | {cores} threads"),
    }
}
