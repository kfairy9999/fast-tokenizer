.PHONY: develop test lint bench

develop:
	maturin develop --release

test:
	cargo test --workspace
	pytest -q

lint:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings

bench:
	python python/benchmark.py
