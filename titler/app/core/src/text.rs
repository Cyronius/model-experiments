//! Port of `clean_title` from scripts/common/validation.py (spec: specs/title/spec.md).
//! The teacher-only steps for think tags are left out: the model never writes them.

pub const MAX_WORDS: usize = 8;

const QUOTE_CHARS: [char; 7] = ['"', '\'', '\u{201c}', '\u{201d}', '\u{2018}', '\u{2019}', '`'];
const SMALL_WORDS: [&str; 16] = ["a", "an", "the", "and", "or", "nor", "but", "for", "of", "in", "on", "at", "to", "vs", "via", "with"];

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn keeps_case(word: &str) -> bool {
    word.chars().any(|c| c.is_numeric()) || word.chars().skip(1).any(|c| c.is_uppercase())
}

// Python str.splitlines() line breaks.
fn is_line_break(c: char) -> bool {
    matches!(c, '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

fn strip_title_prefix(t: &str) -> &str {
    let rest = t.trim_start();
    if rest.get(..5).is_some_and(|p| p.eq_ignore_ascii_case("title")) {
        let after = rest[5..].trim_start();
        if let Some(r) = after.strip_prefix(':') {
            return r.trim_start();
        }
    }
    t
}

// `^\s*\d+\s*-\s*\d+\s*w(?:or)?ds?(\s+title)?\s*:?\s*`, e.g. "3-8 Words:" leaked from a prompt.
fn strip_format_range(t: &str) -> &str {
    let b: Vec<(usize, char)> = t.char_indices().collect();
    let mut i = 0;
    let skip_ws = |i: &mut usize| {
        while *i < b.len() && b[*i].1.is_whitespace() {
            *i += 1;
        }
    };
    let digits = |i: &mut usize| {
        let s = *i;
        while *i < b.len() && b[*i].1.is_numeric() {
            *i += 1;
        }
        *i > s
    };
    skip_ws(&mut i);
    if !digits(&mut i) {
        return t;
    }
    skip_ws(&mut i);
    if i >= b.len() || b[i].1 != '-' {
        return t;
    }
    i += 1;
    skip_ws(&mut i);
    if !digits(&mut i) {
        return t;
    }
    skip_ws(&mut i);
    let rest: String = b[i..].iter().map(|x| x.1).collect::<String>().to_lowercase();
    let word_len = if rest.starts_with("words") {
        5
    } else if rest.starts_with("word") {
        4
    } else if rest.starts_with("wds") {
        3
    } else if rest.starts_with("wd") {
        2
    } else {
        return t;
    };
    i += word_len;
    let mut j = i;
    while j < b.len() && b[j].1.is_whitespace() {
        j += 1;
    }
    if j > i && b[j..].iter().map(|x| x.1).take(5).collect::<String>().to_lowercase() == "title" {
        i = j + 5;
    }
    skip_ws(&mut i);
    if i < b.len() && b[i].1 == ':' {
        i += 1;
    }
    skip_ws(&mut i);
    if i >= b.len() {
        ""
    } else {
        &t[b[i].0..]
    }
}

pub fn clean_title(raw: &str) -> String {
    let mut t = raw.trim().replace('\u{2019}', "'");
    if let Some(end) = t.find(is_line_break) {
        t.truncate(end);
    }
    let t = t.trim().trim_matches(|c: char| QUOTE_CHARS.contains(&c) || c == ' ');
    let t = strip_title_prefix(t);
    let t = strip_format_range(t);
    let t = t.trim_end_matches(|c: char| matches!(c, '.' | '!' | '?' | ':' | ';' | ',') || c.is_whitespace());
    let t: String = t.chars().filter(|&c| c != '*').collect();

    // An apostrophe survives only between two word characters; other quote marks always go.
    let chars: Vec<char> = t.chars().collect();
    let mut unquoted = String::with_capacity(t.len());
    for (i, &c) in chars.iter().enumerate() {
        let drop = match c {
            '"' | '\u{201c}' | '\u{201d}' | '\u{2018}' | '`' => true,
            '\'' | '\u{2019}' => !(i > 0 && is_word_char(chars[i - 1]) && i + 1 < chars.len() && is_word_char(chars[i + 1])),
            _ => false,
        };
        if !drop {
            unquoted.push(c);
        }
    }

    // Runs of two or more whitespace characters become one space; a lone tab or space stays.
    let mut collapsed = String::with_capacity(unquoted.len());
    let cs: Vec<char> = unquoted.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        if cs[i].is_whitespace() {
            let mut j = i;
            while j < cs.len() && cs[j].is_whitespace() {
                j += 1;
            }
            if j - i >= 2 {
                collapsed.push(' ');
            } else {
                collapsed.push(cs[i]);
            }
            i = j;
        } else {
            collapsed.push(cs[i]);
            i += 1;
        }
    }

    let words: Vec<&str> = collapsed.split_whitespace().take(MAX_WORDS).collect();
    let last = words.len().saturating_sub(1);
    words
        .iter()
        .enumerate()
        .map(|(i, w)| {
            if keeps_case(w) {
                w.to_string()
            } else if i > 0 && i < last && SMALL_WORDS.contains(&w.to_lowercase().as_str()) {
                w.to_lowercase()
            } else {
                let mut cs = w.chars();
                match cs.next() {
                    Some(first) => first.to_uppercase().chain(cs).collect(),
                    None => String::new(),
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
