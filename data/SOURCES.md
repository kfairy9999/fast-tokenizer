# Data sources

These files are inputs for tests and the published benchmark. They are not
model weights.

## GPT-2 tokenizer

| file | sha256 | source |
| --- | --- | --- |
| `gpt2/vocab.json` | `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783` | [encoder.json](https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json) |
| `gpt2/merges.txt` | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` | [vocab.bpe](https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe) |

The hashes are the ones tiktoken 0.14 pins for the `gpt2` encoding. The files
were released with GPT-2 under the MIT license.

## Public-domain corpora

Plain text from Project Gutenberg. The benchmark hashes these exact bytes.

| file | work | gutenberg id | sha256 |
| --- | --- | ---: | --- |
| `corpora/pg11.txt` | Alice's Adventures in Wonderland | 11 | `01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e` |
| `corpora/pg84.txt` | Frankenstein | 84 | `2b97b7f8aad508649481d8dd4f6b54f9b965ddb3ba634e9151c815f537b49e42` |
| `corpora/pg1342.txt` | Pride and Prejudice | 1342 | `3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75` |
| `corpora/pg1661.txt` | The Adventures of Sherlock Holmes | 1661 | `922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0` |
| `corpora/pg2701.txt` | Moby Dick | 2701 | `b5e88b71a1cda068e9ab1d94344c9d932338ba6da1a66def97a1aecd9cfcc87d` |

Downloaded from `https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt`.
