//! BPE training.
//!
//! The learner counts adjacent pairs inside pretokens, weighted by how often
//! the pretoken occurs. Each step takes the highest count and, on a tie, the
//! lexicographically smaller pair of raw byte strings. That pair is replaced
//! everywhere it occurs, left to right, without overlapping inside a word.
//! Encode does not use this "merge every occurrence" pass; it uses the ranks
//! recorded here.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::bytes::ByteMap;
use crate::error::{Error, Result};
use crate::pretokenize::{pretokenize, special_spans};
use crate::tokenizer::Tokenizer;

struct Word {
    symbols: Vec<u32>,
    freq: u64,
}

pub(crate) fn train(
    text: &str,
    vocab_size: usize,
    special_tokens: &[&str],
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Tokenizer> {
    learn(text, vocab_size, special_tokens, progress, true)
}

fn learn(
    text: &str,
    vocab_size: usize,
    special_tokens: &[&str],
    progress: &mut dyn FnMut(usize, usize),
    incremental: bool,
) -> Result<Tokenizer> {
    let specials = validate_specials(special_tokens)?;
    let minimum = 256 + specials.len();
    if vocab_size < minimum {
        return Err(Error::VocabTooSmall {
            vocab_size,
            minimum,
            specials: specials.len(),
        });
    }
    let n_merges = vocab_size - minimum;
    let byte_map = ByteMap::new();

    let mut words = count_words(text, &specials, &byte_map);
    let mut id_to_bytes = initial_bytes(&byte_map);
    let learned = if incremental {
        learn_incremental(&mut words, &mut id_to_bytes, n_merges, progress)
    } else {
        learn_brute(&mut words, &mut id_to_bytes, n_merges, progress)
    };

    let id_to_token = id_to_bytes
        .iter()
        .map(|bytes| byte_map.to_unicode(bytes))
        .collect::<Vec<_>>();
    let merge_pairs = learned
        .iter()
        .map(|(left, right)| {
            (
                byte_map.to_unicode(&id_to_bytes[*left as usize]),
                byte_map.to_unicode(&id_to_bytes[*right as usize]),
            )
        })
        .collect::<Vec<_>>();

    Tokenizer::from_trained(
        byte_map.byte_to_id,
        id_to_bytes,
        id_to_token,
        merge_pairs,
        specials,
    )
}

fn validate_specials(special_tokens: &[&str]) -> Result<Vec<String>> {
    let mut seen = FxHashSet::default();
    let mut specials = Vec::with_capacity(special_tokens.len());
    for token in special_tokens {
        if token.is_empty() {
            return Err(Error::EmptySpecial);
        }
        if !seen.insert(*token) {
            return Err(Error::DuplicateSpecial((*token).to_string()));
        }
        specials.push((*token).to_string());
    }
    Ok(specials)
}

fn count_words(text: &str, specials: &[String], byte_map: &ByteMap) -> Vec<Word> {
    let tagged: Vec<(String, u32)> = specials
        .iter()
        .enumerate()
        .map(|(index, token)| (token.clone(), index as u32))
        .collect();
    let spans = special_spans(text, &tagged);
    let mut counts: FxHashMap<&str, u64> = FxHashMap::default();
    let mut cursor = 0;
    for (start, end, _) in spans {
        if start > cursor {
            add_pretokens(&text[cursor..start], &mut counts);
        }
        cursor = end;
    }
    if cursor < text.len() {
        add_pretokens(&text[cursor..], &mut counts);
    }
    counts
        .into_iter()
        .map(|(part, freq)| Word {
            symbols: part
                .as_bytes()
                .iter()
                .map(|&byte| byte_map.byte_to_id[byte as usize])
                .collect(),
            freq,
        })
        .collect()
}

fn add_pretokens<'a>(text: &'a str, counts: &mut FxHashMap<&'a str, u64>) {
    for part in pretokenize(text) {
        *counts.entry(part).or_default() += 1;
    }
}

fn initial_bytes(byte_map: &ByteMap) -> Vec<Vec<u8>> {
    let mut id_to_bytes = vec![Vec::new(); 256];
    for byte in 0..256 {
        let id = byte_map.byte_to_id[byte] as usize;
        id_to_bytes[id] = vec![byte as u8];
    }
    debug_assert!(id_to_bytes.iter().all(|bytes| bytes.len() == 1));
    id_to_bytes
}

fn learn_incremental(
    words: &mut [Word],
    id_to_bytes: &mut Vec<Vec<u8>>,
    n_merges: usize,
    progress: &mut dyn FnMut(usize, usize),
) -> Vec<(u32, u32)> {
    let mut counts = rebuild_counts(words);
    let mut learned = Vec::with_capacity(n_merges.min(4096));
    for step in 0..n_merges {
        let Some((left, right, _)) = best_pair(&counts, id_to_bytes) else {
            break;
        };
        let new_id = push_symbol(id_to_bytes, left, right);
        for word in words.iter_mut() {
            if has_pair(&word.symbols, left, right) {
                apply_counted(word, left, right, new_id, &mut counts);
            }
        }
        learned.push((left, right));
        progress(step + 1, n_merges);
    }
    learned
}

fn learn_brute(
    words: &mut [Word],
    id_to_bytes: &mut Vec<Vec<u8>>,
    n_merges: usize,
    progress: &mut dyn FnMut(usize, usize),
) -> Vec<(u32, u32)> {
    let mut learned = Vec::with_capacity(n_merges.min(4096));
    for step in 0..n_merges {
        let counts = rebuild_counts(words);
        let Some((left, right, _)) = best_pair(&counts, id_to_bytes) else {
            break;
        };
        let new_id = push_symbol(id_to_bytes, left, right);
        for word in words.iter_mut() {
            if has_pair(&word.symbols, left, right) {
                rewrite(&mut word.symbols, left, right, new_id);
            }
        }
        learned.push((left, right));
        progress(step + 1, n_merges);
    }
    learned
}

fn push_symbol(id_to_bytes: &mut Vec<Vec<u8>>, left: u32, right: u32) -> u32 {
    let new_id = id_to_bytes.len() as u32;
    let mut bytes = id_to_bytes[left as usize].clone();
    bytes.extend_from_slice(&id_to_bytes[right as usize]);
    id_to_bytes.push(bytes);
    new_id
}

fn rebuild_counts(words: &[Word]) -> FxHashMap<u64, i64> {
    let mut counts = FxHashMap::default();
    for word in words {
        if word.symbols.len() < 2 {
            continue;
        }
        for pair in word.symbols.windows(2) {
            *counts.entry(pack(pair[0], pair[1])).or_default() += word.freq as i64;
        }
    }
    counts
}

fn best_pair(counts: &FxHashMap<u64, i64>, id_to_bytes: &[Vec<u8>]) -> Option<(u32, u32, i64)> {
    let mut best: Option<(u32, u32, i64)> = None;
    for (&key, &count) in counts {
        if count <= 0 {
            continue;
        }
        let left = (key >> 32) as u32;
        let right = key as u32;
        let replace = match best {
            None => true,
            Some((best_left, best_right, best_count)) => pair_is_better(
                left,
                right,
                count,
                best_left,
                best_right,
                best_count,
                id_to_bytes,
            ),
        };
        if replace {
            best = Some((left, right, count));
        }
    }
    best
}

fn pair_is_better(
    left: u32,
    right: u32,
    count: i64,
    best_left: u32,
    best_right: u32,
    best_count: i64,
    id_to_bytes: &[Vec<u8>],
) -> bool {
    if count != best_count {
        return count > best_count;
    }
    match id_to_bytes[left as usize].cmp(&id_to_bytes[best_left as usize]) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => id_to_bytes[right as usize] < id_to_bytes[best_right as usize],
    }
}

fn has_pair(symbols: &[u32], left: u32, right: u32) -> bool {
    symbols
        .windows(2)
        .any(|pair| pair[0] == left && pair[1] == right)
}

fn apply_counted(
    word: &mut Word,
    left: u32,
    right: u32,
    new_id: u32,
    counts: &mut FxHashMap<u64, i64>,
) {
    let freq = word.freq as i64;
    let mut delta: FxHashMap<u64, i64> = FxHashMap::default();
    if word.symbols.len() >= 2 {
        for pair in word.symbols.windows(2) {
            *delta.entry(pack(pair[0], pair[1])).or_default() -= freq;
        }
    }
    rewrite(&mut word.symbols, left, right, new_id);
    if word.symbols.len() >= 2 {
        for pair in word.symbols.windows(2) {
            *delta.entry(pack(pair[0], pair[1])).or_default() += freq;
        }
    }
    for (key, change) in delta {
        if change == 0 {
            continue;
        }
        let entry = counts.entry(key).or_insert(0);
        *entry += change;
        debug_assert!(*entry >= 0, "pair count went negative");
        if *entry == 0 {
            counts.remove(&key);
        }
    }
}

fn rewrite(symbols: &mut Vec<u32>, left: u32, right: u32, new_id: u32) {
    let old = std::mem::take(symbols);
    symbols.reserve(old.len());
    let mut index = 0;
    while index < old.len() {
        if index + 1 < old.len() && old[index] == left && old[index + 1] == right {
            symbols.push(new_id);
            index += 2;
        } else {
            symbols.push(old[index]);
            index += 1;
        }
    }
}

#[inline]
fn pack(left: u32, right: u32) -> u64 {
    ((left as u64) << 32) | right as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(model: &Tokenizer, text: &str) {
        let ids = model.encode(text);
        assert_eq!(model.decode(&ids).unwrap(), text);
    }

    #[test]
    fn tie_break_prefers_smaller_bytes() {
        // "ab" and "cd" each occur once. Counts match, and 'a' < 'c'.
        let model = Tokenizer::train("ab\ncd", 258, &[]).unwrap();
        assert_eq!(
            model.merges(),
            &[
                ("a".to_string(), "b".to_string()),
                ("c".to_string(), "d".to_string())
            ]
        );
        assert_eq!(model.encode("ab").len(), 1);
        assert_eq!(model.encode("cd").len(), 1);
        assert_ne!(model.encode("ab"), model.encode("cd"));
    }

    #[test]
    fn overlapping_run_merges_left_to_right() {
        let model = Tokenizer::train("aaaa", 258, &[]).unwrap();
        assert_eq!(model.encode("aaaa").len(), 1);
        assert_eq!(model.encode("aa").len(), 1);
        round_trip(&model, "aaaa");
        round_trip(&model, "aa");
        round_trip(&model, "aaa");
    }

    #[test]
    fn specials_stay_atomic_and_unicode_round_trips() {
        let text = "hello <|endoftext|> café 東京 😀\nit's fine";
        let model = Tokenizer::train(text, 400, &["<|endoftext|>"]).unwrap();
        let ids = model.encode(text);
        let special = model.token_to_id("<|endoftext|>").unwrap();
        assert!(ids.contains(&special));
        assert_eq!(model.encode("<|endoftext|>"), vec![special]);
        // Without the special rule the marker is several pretokens.
        assert!(model.encode_ordinary("<|endoftext|>").len() > 1);
        round_trip(&model, text);
        round_trip(&model, "unseen naïve — 👨‍👩‍👧 \u{00a0}done");
        round_trip(&model, "");
        assert!(model.encode("").is_empty());
        assert_eq!(model.decode(&[]).unwrap(), "");
    }

    #[test]
    fn incremental_matches_brute() {
        let mut text = String::new();
        let mut state = 0x1234_5678_u64;
        let alphabet: Vec<char> = "abcdefghijklmnopqrstuvwxyz \n\t'.,!?".chars().collect();
        for _ in 0..2000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            text.push(alphabet[(state as usize) % alphabet.len()]);
        }
        text.push_str(" low low low low wider it's the café <|endoftext|> end");
        let mut noop = |_: usize, _: usize| {};
        let incremental = learn(&text, 320, &["<|endoftext|>"], &mut noop, true).unwrap();
        let brute = learn(&text, 320, &["<|endoftext|>"], &mut noop, false).unwrap();
        assert_eq!(incremental.merges(), brute.merges());
        assert_eq!(incremental.encode(&text), brute.encode(&text));
        assert_eq!(incremental.vocab_size(), brute.vocab_size());
    }

    #[test]
    fn rejects_a_too_small_vocab_and_bad_specials() {
        assert!(Tokenizer::train("abc", 255, &[]).is_err());
        assert!(Tokenizer::train("abc", 256, &[""]).is_err());
        assert!(Tokenizer::train("abc", 300, &["<|end|>", "<|end|>"]).is_err());
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = std::env::temp_dir().join(format!(
            "fast-tokenizer-train-{}-{}",
            std::process::id(),
            "save"
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let text = "low low low wider wider the cat sat on the mat";
        let model = Tokenizer::train(text, 300, &["<|endoftext|>"]).unwrap();
        model.save(&dir).unwrap();
        let loaded = Tokenizer::load(&dir).unwrap();
        assert_eq!(loaded.merges(), model.merges());
        assert_eq!(loaded.encode(text), model.encode(text));
        assert_eq!(
            loaded.token_to_id("<|endoftext|>"),
            model.token_to_id("<|endoftext|>")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
