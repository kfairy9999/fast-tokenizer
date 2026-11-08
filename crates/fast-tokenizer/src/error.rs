use std::io;
use std::path::PathBuf;

/// Errors returned by training, loading, and decoding.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to read {path}: {source}")]
    ReadFile { path: String, source: io::Error },

    #[error("failed to write {path}: {source}")]
    WriteFile { path: String, source: io::Error },

    #[error("failed to parse JSON in {path}: {source}")]
    Json {
        path: String,
        source: serde_json::Error,
    },

    #[error("invalid merges file {path}: {reason}")]
    InvalidMerges { path: String, reason: String },

    #[error("invalid vocabulary: {0}")]
    InvalidVocab(String),

    #[error(
        "vocab_size {vocab_size} is below the minimum {minimum} \
         (256 byte tokens + {specials} special tokens)"
    )]
    VocabTooSmall {
        vocab_size: usize,
        minimum: usize,
        specials: usize,
    },

    #[error("unknown token id {0}")]
    UnknownToken(u32),

    #[error("special token must be non-empty")]
    EmptySpecial,

    #[error("duplicate special token {0:?}")]
    DuplicateSpecial(String),

    #[error(
        "special token {0:?} collides with a byte or merge token; \
         choose a marker that byte-level BPE will not form, such as <|endoftext|>"
    )]
    SpecialCollides(String),

    #[error("tokenizer files not found in {dir} (expected vocab.json and merges.txt)")]
    ModelNotFound { dir: PathBuf },
}

pub type Result<T> = std::result::Result<T, Error>;
