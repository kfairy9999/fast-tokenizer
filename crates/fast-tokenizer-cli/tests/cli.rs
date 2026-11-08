use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn train_encode_decode_round_trip() {
    let bin = env!("CARGO_BIN_EXE_fast-tokenizer");
    let input = repo_root().join("data/sample.txt");
    let output = std::env::temp_dir().join(format!("fast-tokenizer-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&output);

    let train = Command::new(bin)
        .args([
            "train",
            "--input",
            input.to_str().unwrap(),
            "--vocab-size",
            "400",
            "--output",
            output.to_str().unwrap(),
            "--special",
            "<|endoftext|>",
        ])
        .output()
        .expect("run train");
    assert!(
        train.status.success(),
        "train failed: {}",
        String::from_utf8_lossy(&train.stderr)
    );

    let text = "hello <|endoftext|> café";
    let encoded = Command::new(bin)
        .args([
            "encode",
            "--model",
            output.to_str().unwrap(),
            "--text",
            text,
        ])
        .output()
        .expect("run encode");
    assert!(
        encoded.status.success(),
        "encode failed: {}",
        String::from_utf8_lossy(&encoded.stderr)
    );

    let decoded = Command::new(bin)
        .args([
            "decode",
            "--model",
            output.to_str().unwrap(),
            "--ids",
            std::str::from_utf8(&encoded.stdout).unwrap().trim(),
        ])
        .output()
        .expect("run decode");
    assert!(
        decoded.status.success(),
        "decode failed: {}",
        String::from_utf8_lossy(&decoded.stderr)
    );
    assert_eq!(std::str::from_utf8(&decoded.stdout).unwrap(), text);
    let _ = std::fs::remove_dir_all(&output);
}
