//! GPT-2 byte-to-unicode display map.
//!
//! Byte-level BPE runs on raw bytes, but the vocab and merges files are text.
//! Bytes that are awkward in a text file (space, newline, controls) are shown
//! as a single Unicode scalar. The mapping is a bijection, so decoding is exact.

use rustc_hash::FxHashMap;

/// Display form of the 256-byte alphabet, plus the GPT-2 id of each byte.
#[derive(Clone, Debug)]
pub struct ByteMap {
    /// Display character for every raw byte.
    pub encode: [char; 256],
    /// Canonical GPT-2 token id of every raw byte (index in the printable-first order).
    pub byte_to_id: [u32; 256],
    decode: FxHashMap<char, u8>,
}

impl ByteMap {
    pub fn new() -> Self {
        // Same construction as GPT-2's `bytes_to_unicode` and tiktoken's
        // `data_gym_to_mergeable_bpe_ranks`: printable Latin-1 except space
        // first, then every remaining byte in numeric order.
        let mut included = [false; 256];
        let mut order: Vec<u8> = Vec::with_capacity(256);
        for b in (b'!'..=b'~').chain(0xA1..=0xAC).chain(0xAE..=0xFF) {
            included[b as usize] = true;
            order.push(b);
        }
        let base = order.len();
        for b in 0..256u16 {
            if !included[b as usize] {
                order.push(b as u8);
            }
        }
        debug_assert_eq!(order.len(), 256);

        let mut encode = ['\0'; 256];
        let mut byte_to_id = [0u32; 256];
        let mut decode = FxHashMap::with_capacity_and_hasher(256, Default::default());
        for (id, &byte) in order.iter().enumerate() {
            let ch = if id < base {
                char::from(byte)
            } else {
                char::from_u32(256 + (id - base) as u32).expect("byte display scalar")
            };
            encode[byte as usize] = ch;
            byte_to_id[byte as usize] = id as u32;
            decode.insert(ch, byte);
        }
        let map = Self {
            encode,
            byte_to_id,
            decode,
        };
        for byte in 0..256u16 {
            let token = map.encode[byte as usize].to_string();
            assert_eq!(
                map.unicode_to_bytes(&token),
                Some(vec![byte as u8]),
                "byte display map is not invertible"
            );
        }
        map
    }

    /// Map raw bytes to the vocab string used in `vocab.json` / `merges.txt`.
    pub fn to_unicode(&self, bytes: &[u8]) -> String {
        bytes.iter().map(|&b| self.encode[b as usize]).collect()
    }

    /// Invert a vocab string that was produced by [`to_unicode`](Self::to_unicode).
    pub fn unicode_to_bytes(&self, token: &str) -> Option<Vec<u8>> {
        let mut out = Vec::with_capacity(token.len());
        for ch in token.chars() {
            out.push(*self.decode.get(&ch)?);
        }
        Some(out)
    }
}

impl Default for ByteMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_byte_ids_match_known_tokens() {
        let map = ByteMap::new();
        assert_eq!(map.byte_to_id[b'a' as usize], 64);
        assert_eq!(map.byte_to_id[b' ' as usize], 220);
        assert_eq!(map.encode[b' ' as usize], '\u{0120}');
        assert_eq!(map.encode[b'!' as usize], '!');
        assert_eq!(map.unicode_to_bytes("Ġa").unwrap(), b" a");
        assert_eq!(map.to_unicode(b" a"), "Ġa");
    }

    #[test]
    fn mapping_is_bijective() {
        let map = ByteMap::new();
        let mut seen = [false; 256];
        for b in 0..256 {
            let ch = map.encode[b];
            let back = map.decode[&ch];
            assert_eq!(back, b as u8);
            seen[map.byte_to_id[b] as usize] = true;
        }
        assert!(seen.iter().all(|s| *s));
    }
}
