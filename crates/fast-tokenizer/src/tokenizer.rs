//! Loaded tokenizer: encode, decode, and the GPT-2 file format.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde_json::Value;

use crate::bytes::ByteMap;
use crate::error::{Error, Result};
use crate::pretokenize::{pretokenize, special_spans};

/// A byte-level BPE model.
///
/// The value is immutable after construction and is safe to share across threads.
#[derive(Debug)]
pub struct Tokenizer {
    byte_to_id: [u32; 256],
    id_to_bytes: Vec<Vec<u8>>,
    id_to_token: Vec<String>,
    token_to_id: FxHashMap<String, u32>,
    /// `(left_id, right_id)` packed to `(rank, result_id)` packed.
    merges: FxHashMap<u64, u64>,
    merge_pairs: Vec<(String, String)>,
    /// Special tokens in ascending id order.
    specials_in_order: Vec<(String, u32)>,
    /// Special tokens ordered for matching (longest first).
    specials_match: Vec<(String, u32)>,
}

impl Tokenizer {
    /// Train a model on `text`.
    ///
    /// `vocab_size` counts byte tokens, learned merges, and special tokens.
    /// Training stops early when the corpus has no pairs left, so the actual
    /// vocabulary can be smaller than requested. `special_tokens` are kept
    /// atomic and are not learned as merges. The default for callers that want
    /// a GPT-2-style marker is `&["<|endoftext|>"]`; pass `&[]` for none.
    pub fn train(text: &str, vocab_size: usize, special_tokens: &[&str]) -> Result<Self> {
        crate::train::train(text, vocab_size, special_tokens, &mut |_, _| {})
    }

    /// Train and receive `(completed_merges, requested_merges)` after each merge.
    pub fn train_with_progress(
        text: &str,
        vocab_size: usize,
        special_tokens: &[&str],
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<Self> {
        crate::train::train(text, vocab_size, special_tokens, progress)
    }

    /// Load `vocab.json` (or `encoder.json`) and `merges.txt` (or `vocab.bpe`).
    pub fn from_files(vocab_path: impl AsRef<Path>, merges_path: impl AsRef<Path>) -> Result<Self> {
        let vocab_path = vocab_path.as_ref();
        let merges_path = merges_path.as_ref();
        let vocab_text = read_to_string(vocab_path)?;
        let merges_text = read_to_string(merges_path)?;
        let vocab: FxHashMap<String, u32> =
            serde_json::from_str(&vocab_text).map_err(|source| Error::Json {
                path: vocab_path.display().to_string(),
                source,
            })?;
        let pairs = parse_merges(&merges_text).map_err(|reason| Error::InvalidMerges {
            path: merges_path.display().to_string(),
            reason,
        })?;
        Self::compile(vocab, pairs)
    }

    /// Load a model directory.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let vocab = find_file(dir, &["vocab.json", "encoder.json"])?;
        let merges = find_file(dir, &["merges.txt", "vocab.bpe"])?;
        Self::from_files(vocab, merges)
    }

    /// Write `vocab.json` and `merges.txt` into `dir`, creating it if needed.
    pub fn save(&self, dir: impl AsRef<Path>) -> Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir).map_err(|source| Error::WriteFile {
            path: dir.display().to_string(),
            source,
        })?;
        self.write_vocab(&dir.join("vocab.json"))?;
        self.write_merges(&dir.join("merges.txt"))?;
        Ok(())
    }

    /// Encode `text`, keeping configured special tokens atomic.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_with(text, true)
    }

    /// Encode `text` without treating special tokens as atomic.
    ///
    /// This matches tiktoken's `encode_ordinary`.
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.encode_with(text, false)
    }

    /// Encode many strings.
    ///
    /// `threads == 1` runs on the calling thread. Any other value uses a rayon
    /// pool: `0` keeps rayon's default pool size, and `n > 1` uses `n` threads.
    /// Each string builds its own pretoken cache. Chunking is the caller's job;
    /// token boundaries are not adjusted to match a single encode of the
    /// concatenation.
    pub fn encode_batch(&self, texts: &[&str], threads: usize) -> Vec<Vec<u32>> {
        if texts.is_empty() || threads == 1 || texts.len() == 1 {
            return texts.iter().map(|text| self.encode(text)).collect();
        }
        let requested = if threads == 0 {
            rayon::current_num_threads()
        } else {
            threads
        };
        if requested <= 1 {
            return texts.iter().map(|text| self.encode(text)).collect();
        }
        if requested == rayon::current_num_threads() {
            texts.par_iter().map(|text| self.encode(text)).collect()
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(requested)
                .build()
                .expect("rayon thread pool");
            pool.install(|| texts.par_iter().map(|text| self.encode(text)).collect())
        }
    }

    /// Decode token ids to text.
    ///
    /// Bytes are concatenated first and then UTF-8 decoded, so a character that
    /// was split across tokens comes back intact. Invalid UTF-8 becomes U+FFFD,
    /// matching tiktoken. Unknown ids are an error.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let piece = self
                .id_to_bytes
                .get(id as usize)
                .ok_or(Error::UnknownToken(id))?;
            bytes.extend_from_slice(piece);
        }
        Ok(String::from_utf8(bytes)
            .unwrap_or_else(|err| String::from_utf8_lossy(err.as_bytes()).into_owned()))
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Learned merges in rank order, as vocab strings.
    pub fn merges(&self) -> &[(String, String)] {
        &self.merge_pairs
    }

    /// Special tokens in ascending id order.
    pub fn special_tokens(&self) -> impl Iterator<Item = &str> {
        self.specials_in_order
            .iter()
            .map(|(token, _)| token.as_str())
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(String::as_str)
    }

    fn encode_with(&self, text: &str, specials_enabled: bool) -> Vec<u32> {
        let mut out = Vec::new();
        let mut cache: FxHashMap<String, Vec<u32>> = FxHashMap::default();
        if text.is_empty() {
            return out;
        }
        if !specials_enabled || self.specials_match.is_empty() {
            self.encode_ordinary_slice(text, &mut cache, &mut out);
            return out;
        }
        let spans = special_spans(text, &self.specials_match);
        let mut cursor = 0;
        for (start, end, id) in spans {
            if start > cursor {
                self.encode_ordinary_slice(&text[cursor..start], &mut cache, &mut out);
            }
            out.push(id);
            cursor = end;
        }
        if cursor < text.len() {
            self.encode_ordinary_slice(&text[cursor..], &mut cache, &mut out);
        }
        out
    }

    fn encode_ordinary_slice(
        &self,
        text: &str,
        cache: &mut FxHashMap<String, Vec<u32>>,
        out: &mut Vec<u32>,
    ) {
        for part in pretokenize(text) {
            if let Some(ids) = cache.get(part) {
                out.extend_from_slice(ids);
                continue;
            }
            let ids = self.bpe(part.as_bytes());
            out.extend_from_slice(&ids);
            cache.insert(part.to_owned(), ids);
        }
    }

    /// Merge the lowest-rank pair until no listed pair remains.
    fn bpe(&self, data: &[u8]) -> Vec<u32> {
        let mut ids: Vec<u32> = Vec::with_capacity(data.len());
        for &byte in data {
            ids.push(self.byte_to_id[byte as usize]);
        }
        if ids.len() < 2 {
            return ids;
        }
        loop {
            let mut best_rank = u32::MAX;
            let mut best_index = 0usize;
            let mut best_result = 0u32;
            for index in 0..ids.len() - 1 {
                if let Some(&packed) = self.merges.get(&pack(ids[index], ids[index + 1])) {
                    let (rank, result) = unpack(packed);
                    if rank < best_rank {
                        best_rank = rank;
                        best_index = index;
                        best_result = result;
                    }
                }
            }
            if best_rank == u32::MAX {
                break;
            }
            ids[best_index] = best_result;
            ids.remove(best_index + 1);
        }
        ids
    }

    pub(crate) fn from_trained(
        byte_to_id: [u32; 256],
        id_to_bytes: Vec<Vec<u8>>,
        id_to_token: Vec<String>,
        merge_pairs: Vec<(String, String)>,
        specials: Vec<String>,
    ) -> Result<Self> {
        let mut token_to_id =
            FxHashMap::with_capacity_and_hasher(id_to_token.len(), Default::default());
        for (id, token) in id_to_token.iter().enumerate() {
            if token_to_id.insert(token.clone(), id as u32).is_some() {
                return Err(Error::InvalidVocab(format!("duplicate token {token:?}")));
            }
        }
        for special in &specials {
            if token_to_id.contains_key(special) {
                return Err(Error::SpecialCollides(special.clone()));
            }
        }
        let mut id_to_token = id_to_token;
        let mut id_to_bytes = id_to_bytes;
        let mut specials_in_order = Vec::with_capacity(specials.len());
        for special in specials {
            let id = id_to_token.len() as u32;
            token_to_id.insert(special.clone(), id);
            id_to_bytes.push(special.as_bytes().to_vec());
            id_to_token.push(special.clone());
            specials_in_order.push((special, id));
        }
        let merges = compile_merges(&token_to_id, &merge_pairs)?;
        let specials_match = match_order(&specials_in_order);
        Ok(Self {
            byte_to_id,
            id_to_bytes,
            id_to_token,
            token_to_id,
            merges,
            merge_pairs,
            specials_in_order,
            specials_match,
        })
    }

    fn compile(vocab: FxHashMap<String, u32>, merge_pairs: Vec<(String, String)>) -> Result<Self> {
        if vocab.is_empty() {
            return Err(Error::InvalidVocab("vocabulary is empty".into()));
        }
        let max_id = vocab.values().copied().max().unwrap();
        if vocab.len() != max_id as usize + 1 {
            return Err(Error::InvalidVocab(
                "token ids are not a contiguous range starting at 0".into(),
            ));
        }
        let mut seen = vec![false; vocab.len()];
        let mut id_to_token = vec![String::new(); vocab.len()];
        for (token, id) in &vocab {
            let slot = &mut seen[*id as usize];
            if *slot {
                return Err(Error::InvalidVocab(format!("duplicate id {id}")));
            }
            *slot = true;
            id_to_token[*id as usize] = token.clone();
        }

        let byte_map = ByteMap::new();
        let mut byte_to_id = [0u32; 256];
        let mut id_to_bytes = vec![Vec::new(); vocab.len()];
        for (byte, slot) in byte_to_id.iter_mut().enumerate() {
            let display = byte_map.encode[byte].to_string();
            let id = *vocab.get(&display).ok_or_else(|| {
                Error::InvalidVocab(format!("missing byte token for 0x{byte:02x} ({display:?})"))
            })?;
            *slot = id;
            id_to_bytes[id as usize] = vec![byte as u8];
        }

        let merges = compile_merges_filling(&vocab, &merge_pairs, &mut id_to_bytes)?;

        let mut specials_in_order = Vec::new();
        for (token, id) in &vocab {
            if id_to_bytes[*id as usize].is_empty() {
                id_to_bytes[*id as usize] = token.as_bytes().to_vec();
                specials_in_order.push((token.clone(), *id));
            }
        }
        specials_in_order.sort_by_key(|(_, id)| *id);
        let specials_match = match_order(&specials_in_order);

        Ok(Self {
            byte_to_id,
            id_to_bytes,
            id_to_token,
            token_to_id: vocab,
            merges,
            merge_pairs,
            specials_in_order,
            specials_match,
        })
    }

    fn write_vocab(&self, path: &Path) -> Result<()> {
        let mut map = serde_json::Map::new();
        for (id, token) in self.id_to_token.iter().enumerate() {
            map.insert(token.clone(), Value::from(id as u32));
        }
        let body = serde_json::to_vec(&Value::Object(map)).map_err(|source| Error::Json {
            path: path.display().to_string(),
            source,
        })?;
        write_bytes(path, &body, true)
    }

    fn write_merges(&self, path: &Path) -> Result<()> {
        let mut body = String::from("#version: 0.2\n");
        for (left, right) in &self.merge_pairs {
            body.push_str(left);
            body.push(' ');
            body.push_str(right);
            body.push('\n');
        }
        write_bytes(path, body.as_bytes(), false)
    }
}

fn compile_merges(
    vocab: &FxHashMap<String, u32>,
    merge_pairs: &[(String, String)],
) -> Result<FxHashMap<u64, u64>> {
    let mut merges = FxHashMap::with_capacity_and_hasher(merge_pairs.len(), Default::default());
    for (rank, (left, right)) in merge_pairs.iter().enumerate() {
        let (left_id, right_id, result_id) = resolve_merge(vocab, left, right)?;
        if merges
            .insert(pack(left_id, right_id), pack(rank as u32, result_id))
            .is_some()
        {
            return Err(Error::InvalidVocab(format!(
                "duplicate merge pair {left:?} {right:?}"
            )));
        }
    }
    Ok(merges)
}

fn resolve_merge(
    vocab: &FxHashMap<String, u32>,
    left: &str,
    right: &str,
) -> Result<(u32, u32, u32)> {
    let left_id = *vocab
        .get(left)
        .ok_or_else(|| Error::InvalidVocab(format!("merge references unknown token {left:?}")))?;
    let right_id = *vocab
        .get(right)
        .ok_or_else(|| Error::InvalidVocab(format!("merge references unknown token {right:?}")))?;
    let merged = format!("{left}{right}");
    let result_id = *vocab.get(&merged).ok_or_else(|| {
        Error::InvalidVocab(format!(
            "merge result {merged:?} is missing from the vocabulary"
        ))
    })?;
    Ok((left_id, right_id, result_id))
}

fn compile_merges_filling(
    vocab: &FxHashMap<String, u32>,
    merge_pairs: &[(String, String)],
    id_to_bytes: &mut [Vec<u8>],
) -> Result<FxHashMap<u64, u64>> {
    let mut merges = FxHashMap::with_capacity_and_hasher(merge_pairs.len(), Default::default());
    for (rank, (left, right)) in merge_pairs.iter().enumerate() {
        let (left_id, right_id, result_id) = resolve_merge(vocab, left, right)?;
        if merges
            .insert(pack(left_id, right_id), pack(rank as u32, result_id))
            .is_some()
        {
            return Err(Error::InvalidVocab(format!(
                "duplicate merge pair {left:?} {right:?}"
            )));
        }
        if id_to_bytes[result_id as usize].is_empty() {
            if id_to_bytes[left_id as usize].is_empty() || id_to_bytes[right_id as usize].is_empty()
            {
                return Err(Error::InvalidVocab(format!(
                    "merge {left:?} {right:?} is out of order"
                )));
            }
            let mut bytes = id_to_bytes[left_id as usize].clone();
            bytes.extend_from_slice(&id_to_bytes[right_id as usize]);
            id_to_bytes[result_id as usize] = bytes;
        }
    }
    Ok(merges)
}

fn match_order(specials: &[(String, u32)]) -> Vec<(String, u32)> {
    let mut ordered = specials.to_vec();
    ordered.sort_by(|left, right| {
        right
            .0
            .len()
            .cmp(&left.0.len())
            .then_with(|| left.1.cmp(&right.1))
    });
    ordered
}

/// Skip a leading `#version` line. Later lines that start with `#` are real
/// merges (`# #` builds the `##` token in the GPT-2 vocabulary).
pub(crate) fn parse_merges(text: &str) -> std::result::Result<Vec<(String, String)>, String> {
    let mut lines = text.lines().peekable();
    if lines
        .peek()
        .is_some_and(|line| line.starts_with("#version"))
    {
        lines.next();
    }
    let mut pairs = Vec::new();
    for (index, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let left = parts
            .next()
            .ok_or_else(|| format!("line {}: expected two tokens", index + 1))?;
        let right = parts
            .next()
            .ok_or_else(|| format!("line {}: expected two tokens", index + 1))?;
        if parts.next().is_some() {
            return Err(format!("line {}: expected exactly two tokens", index + 1));
        }
        pairs.push((left.to_string(), right.to_string()));
    }
    Ok(pairs)
}

fn find_file(dir: &Path, names: &[&str]) -> Result<PathBuf> {
    for name in names {
        let path = dir.join(name);
        if path.is_file() {
            return Ok(path);
        }
    }
    Err(Error::ModelNotFound {
        dir: dir.to_path_buf(),
    })
}

fn read_to_string(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|source| Error::ReadFile {
        path: path.display().to_string(),
        source,
    })
}

fn write_bytes(path: &Path, bytes: &[u8], newline: bool) -> Result<()> {
    let mut file = File::create(path).map_err(|source| Error::WriteFile {
        path: path.display().to_string(),
        source,
    })?;
    file.write_all(bytes).map_err(|source| Error::WriteFile {
        path: path.display().to_string(),
        source,
    })?;
    if newline {
        file.write_all(b"\n").map_err(|source| Error::WriteFile {
            path: path.display().to_string(),
            source,
        })?;
    }
    Ok(())
}

#[inline]
fn pack(hi: u32, lo: u32) -> u64 {
    ((hi as u64) << 32) | lo as u64
}

#[inline]
fn unpack(value: u64) -> (u32, u32) {
    ((value >> 32) as u32, value as u32)
}
