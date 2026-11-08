//! Token-for-token parity with a tiktoken GPT-2 fixture.

use std::fs;
use std::path::PathBuf;

use fast_tokenizer::Tokenizer;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Fixture {
    samples: Vec<Sample>,
    corpora: Vec<Corpus>,
}

#[derive(Deserialize)]
struct Sample {
    text: String,
    ids: Vec<u32>,
    ordinary: Vec<u32>,
}

#[derive(Deserialize)]
struct Corpus {
    file: String,
    file_sha256: String,
    tokens: usize,
    ids_sha256: String,
    head_ids: Vec<u32>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn hash_ids(ids: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for id in ids {
        hasher.update(id.to_le_bytes());
    }
    hex_encode(&hasher.finalize())
}

fn hash_bytes(bytes: &[u8]) -> String {
    hex_encode(&Sha256::digest(bytes))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

#[test]
fn gpt2_matches_tiktoken_fixture() {
    let root = repo_root();
    let model = Tokenizer::load(root.join("data/gpt2")).expect("load gpt2");
    assert_eq!(model.vocab_size(), 50257);
    assert_eq!(model.merges().len(), 50000);
    assert_eq!(
        model.special_tokens().collect::<Vec<_>>(),
        vec!["<|endoftext|>"]
    );
    assert_eq!(model.token_to_id("<|endoftext|>"), Some(50256));
    assert_eq!(model.token_to_id("a"), Some(64));
    assert_eq!(model.token_to_id("Ġ"), Some(220));

    let fixture: Fixture = serde_json::from_str(
        &fs::read_to_string(root.join("data/fixtures/gpt2_parity.json")).unwrap(),
    )
    .unwrap();

    for sample in &fixture.samples {
        assert_eq!(
            model.encode(&sample.text),
            sample.ids,
            "encode {:?}",
            sample.text
        );
        assert_eq!(
            model.encode_ordinary(&sample.text),
            sample.ordinary,
            "ordinary {:?}",
            sample.text
        );
        assert_eq!(model.decode(&sample.ids).unwrap(), sample.text);
    }

    for corpus in &fixture.corpora {
        let path = root.join("data/corpora").join(&corpus.file);
        let bytes = fs::read(&path).unwrap();
        assert_eq!(hash_bytes(&bytes), corpus.file_sha256, "{}", corpus.file);
        let text = String::from_utf8(bytes).unwrap();
        let ids = model.encode(&text);
        assert_eq!(ids.len(), corpus.tokens, "{}", corpus.file);
        assert_eq!(&ids[..corpus.head_ids.len()], corpus.head_ids.as_slice());
        assert_eq!(hash_ids(&ids), corpus.ids_sha256, "{}", corpus.file);
        assert_eq!(model.decode(&ids).unwrap(), text);
    }
}
