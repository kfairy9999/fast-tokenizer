//! Python module `fast_tokenizer`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use bpe::Tokenizer;

/// Byte-level BPE tokenizer backed by the Rust core.
#[pyclass(name = "Tokenizer", frozen)]
struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Train on `text`.
    ///
    /// `vocab_size` includes the 256 byte tokens. `special_tokens` are kept
    /// atomic and are not learned as merges. Training stops early when the
    /// corpus runs out of pairs.
    #[staticmethod]
    #[pyo3(signature = (text, vocab_size=8000, special_tokens=None))]
    fn train(text: &str, vocab_size: usize, special_tokens: Option<Vec<String>>) -> PyResult<Self> {
        let owned = special_tokens.unwrap_or_default();
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        let inner = Tokenizer::train(text, vocab_size, &refs).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Load a directory containing `vocab.json` and `merges.txt`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = Tokenizer::load(path).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Load GPT-2-style vocabulary and merges files.
    #[staticmethod]
    fn from_gpt2_files(vocab_path: &str, merges_path: &str) -> PyResult<Self> {
        let inner = Tokenizer::from_files(vocab_path, merges_path).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Encode `text` to token ids.
    ///
    /// Special tokens are atomic when `special_tokens` is true.
    #[pyo3(signature = (text, special_tokens=true))]
    fn encode(&self, text: &str, special_tokens: bool) -> Vec<u32> {
        if special_tokens {
            self.inner.encode(text)
        } else {
            self.inner.encode_ordinary(text)
        }
    }

    /// Encode without the special-token rule (tiktoken `encode_ordinary`).
    fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.inner.encode_ordinary(text)
    }

    /// Encode many strings. `num_threads` 0 uses the default rayon pool.
    ///
    /// The GIL is released for the duration of the batch.
    #[pyo3(signature = (texts, num_threads=0))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        num_threads: usize,
    ) -> Vec<Vec<u32>> {
        py.allow_threads(|| {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            self.inner.encode_batch(&refs, num_threads)
        })
    }

    /// Decode token ids. Unknown ids raise `ValueError`.
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode(&ids).map_err(to_py)
    }

    /// Write `vocab.json` and `merges.txt` into `path`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(to_py)
    }

    /// Learned merges in rank order, as vocabulary strings.
    fn merges(&self) -> Vec<(String, String)> {
        self.inner.merges().to_vec()
    }

    /// Special tokens in ascending id order.
    fn special_tokens(&self) -> Vec<String> {
        self.inner.special_tokens().map(str::to_owned).collect()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id).map(str::to_owned)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn __repr__(&self) -> String {
        format!(
            "Tokenizer(vocab_size={}, merges={}, specials={})",
            self.inner.vocab_size(),
            self.inner.merges().len(),
            self.inner.special_tokens().count()
        )
    }
}

fn to_py(err: bpe::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Fast byte-level BPE tokenizer implemented in Rust.
#[pymodule]
fn fast_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
