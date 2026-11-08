//! Byte-level byte-pair encoding.
//!
//! The tokenizer trains a vocabulary on raw UTF-8 bytes, encodes text to token
//! ids, and decodes those ids back to text. The on-disk format is the GPT-2
//! pair of `vocab.json` and `merges.txt`, so a model trained here and the
//! original GPT-2 files load through the same path.
//!
//! # Algorithm
//!
//! 1. Split off special tokens such as `<|endoftext|>` (longest match, earliest
//!    start).
//! 2. Pretokenize the remainder with the GPT-2 regex, implemented as a direct
//!    scanner. Merges never cross a pretoken boundary.
//! 3. Inside a pretoken, repeatedly merge the adjacent pair with the lowest
//!    rank. Rank is the line order in `merges.txt` (lower is earlier), one
//!    pair at a time. This is the rule tiktoken uses for GPT-2.
//! 4. Decode by concatenating each token's raw bytes and UTF-8-decoding the
//!    result. Valid Unicode text round-trips exactly.
//!
//! Training picks the most frequent pair in the corpus, breaks ties by the
//! raw byte strings of the pair, and applies that pair to every non-overlapping
//! occurrence in one pass. That training rule is intentionally not the encode
//! rule: encode always consults the learned ranks.
//!
//! ```
//! use fast_tokenizer::Tokenizer;
//! let model = Tokenizer::train("low low low low wider", 300, &[]).unwrap();
//! let ids = model.encode("low wider");
//! assert_eq!(model.decode(&ids).unwrap(), "low wider");
//! ```

mod bytes;
mod error;
mod pretokenize;
mod tokenizer;
mod train;

pub use error::{Error, Result};
pub use pretokenize::pretokenize;
pub use tokenizer::Tokenizer;
