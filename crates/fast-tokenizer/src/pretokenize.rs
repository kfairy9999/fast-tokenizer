//! GPT-2 / tiktoken `r50k` pretokenizer.
//!
//! The pattern (tiktoken `r50k_pat_str`, equivalent to the original GPT-2 regex) is:
//!
//! ```text
//! '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+$|\s+(?!\S)|\s
//! ```
//!
//! Merges never cross these pieces. The scanner below follows the same
//! left-to-right, first-alternative-wins behavior, including the trailing
//! whitespace rule: a whitespace run at the end of the string stays one piece,
//! while a run in the middle keeps its last whitespace character so it can
//! attach to the next word.

use unicode_general_category::{get_general_category, GeneralCategory};

/// Unicode `White_Space` characters matched by `\s` in the GPT-2 pattern.
/// An explicit set keeps the splitter stable across Unicode versions.
#[inline]
fn is_ws(c: char) -> bool {
    matches!(
        c,
        '\u{0009}'
            | '\u{000A}'
            | '\u{000B}'
            | '\u{000C}'
            | '\u{000D}'
            | '\u{0020}'
            | '\u{0085}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'
            ..='\u{200A}' | '\u{2028}' | '\u{2029}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
    )
}

#[inline]
fn is_letter(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

#[inline]
fn is_number(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    )
}

#[inline]
fn is_other(c: char) -> bool {
    !is_ws(c) && !is_letter(c) && !is_number(c)
}

/// Split `text` the way the GPT-2 regex does.
///
/// The returned slices borrow `text` and concatenate back to it.
pub fn pretokenize(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let cps: Vec<(usize, char)> = text.char_indices().collect();
    let n = cps.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let end = match_contraction(&cps, i)
            .or_else(|| match_class(&cps, i, is_letter))
            .or_else(|| match_class(&cps, i, is_number))
            .or_else(|| match_class(&cps, i, is_other))
            .or_else(|| match_ws(&cps, i))
            .expect("every scalar is a letter, number, other, or whitespace");
        debug_assert!(end > i);
        let start_b = cps[i].0;
        let end_b = if end < n { cps[end].0 } else { text.len() };
        out.push(&text[start_b..end_b]);
        i = end;
    }
    out
}

fn match_contraction(cps: &[(usize, char)], i: usize) -> Option<usize> {
    if cps[i].1 != '\'' {
        return None;
    }
    let n = cps.len();
    if i + 1 < n && matches!(cps[i + 1].1, 's' | 'd' | 'm' | 't') {
        return Some(i + 2);
    }
    if i + 2 < n {
        let pair = (cps[i + 1].1, cps[i + 2].1);
        if matches!(pair, ('l', 'l') | ('v', 'e') | ('r', 'e')) {
            return Some(i + 3);
        }
    }
    None
}

/// Optional leading ASCII space, then one or more characters of `pred`.
fn match_class(cps: &[(usize, char)], i: usize, pred: fn(char) -> bool) -> Option<usize> {
    let n = cps.len();
    let start = if cps[i].1 == ' ' && i + 1 < n && pred(cps[i + 1].1) {
        i + 1
    } else if pred(cps[i].1) {
        i
    } else {
        return None;
    };
    let mut j = start + 1;
    while j < n && pred(cps[j].1) {
        j += 1;
    }
    Some(j)
}

fn match_ws(cps: &[(usize, char)], i: usize) -> Option<usize> {
    let n = cps.len();
    if !is_ws(cps[i].1) {
        return None;
    }
    let mut j = i + 1;
    while j < n && is_ws(cps[j].1) {
        j += 1;
    }
    if j == n {
        // `\s++$` consumes a trailing run whole.
        Some(j)
    } else if j - i >= 2 {
        // `\s+(?!\S)` backtracks one so the last whitespace can join the next word.
        Some(j - 1)
    } else {
        // A single non-trailing whitespace is the final `\s` alternative.
        Some(j)
    }
}

/// Non-overlapping special-token spans `(start_byte, end_byte, id)`.
///
/// The earliest match wins. Two matches that start at the same byte resolve
/// to the longer token.
pub(crate) fn special_spans(text: &str, specials: &[(String, u32)]) -> Vec<(usize, usize, u32)> {
    let mut spans = Vec::new();
    if specials.is_empty() || text.is_empty() {
        return spans;
    }
    let mut i = 0;
    while i < text.len() {
        let mut best: Option<(usize, usize, u32)> = None;
        for (token, id) in specials {
            if token.is_empty() {
                continue;
            }
            if let Some(rel) = text[i..].find(token.as_str()) {
                let start = i + rel;
                let end = start + token.len();
                let better = match best {
                    None => true,
                    Some((best_start, best_end, _)) => {
                        start < best_start || (start == best_start && end > best_end)
                    }
                };
                if better {
                    best = Some((start, end, *id));
                }
            }
        }
        match best {
            Some((start, end, id)) => {
                spans.push((start, end, id));
                i = end;
            }
            None => break,
        }
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(s: &str) -> Vec<&str> {
        pretokenize(s)
    }

    #[test]
    fn covers_the_gpt2_cases() {
        assert!(split("").is_empty());
        assert_eq!(split("hello world"), vec!["hello", " world"]);
        assert_eq!(split("it's"), vec!["it", "'s"]);
        assert_eq!(split("I'll"), vec!["I", "'ll"]);
        assert_eq!(split("you're"), vec!["you", "'re"]);
        assert_eq!(split("don't"), vec!["don", "'t"]);
        assert_eq!(split("  leading"), vec![" ", " leading"]);
        assert_eq!(split("double  space"), vec!["double", " ", " space"]);
        assert_eq!(split("hello\n\n"), vec!["hello", "\n\n"]);
        assert_eq!(split("hello\n\nworld"), vec!["hello", "\n", "\n", "world"]);
        assert_eq!(split("hello \n world"), vec!["hello", " \n", " world"]);
        assert_eq!(split(" \n\n "), vec![" \n\n "]);
        assert_eq!(split("  "), vec!["  "]);
        assert_eq!(split("<|endoftext|>"), vec!["<|", "endoftext", "|>"]);
        assert_eq!(split("emoji 😀"), vec!["emoji", " 😀"]);
        assert_eq!(split("café"), vec!["café"]);
        assert_eq!(
            split("the 19th century"),
            vec!["the", " 19", "th", " century"]
        );
        assert_eq!(split("a\u{0301}"), vec!["a", "\u{0301}"]);
        assert_eq!(split("####"), vec!["####"]);
        assert_eq!(split("line\nbreak"), vec!["line", "\n", "break"]);
    }

    #[test]
    fn pieces_cover_the_input() {
        let text = "It's a fine day.\n\nThe 19th-century café — \"hello\" — costs $5.00.\n";
        let parts = pretokenize(text);
        assert_eq!(parts.concat(), text);
    }

    #[test]
    fn longest_special_wins() {
        let specials = vec![("a".to_string(), 1), ("aaa".to_string(), 2)];
        let spans = special_spans("aaa", &specials);
        assert_eq!(spans, vec![(0, 3, 2)]);
    }
}
