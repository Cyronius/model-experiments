//! Unigram tokenizer that reproduces the Hugging Face pipeline the model was trained with:
//! NFKC, whitespace collapse, Metaspace split, Viterbi, byte fallback.

use std::collections::HashMap;

use unicode_normalization::UnicodeNormalization;

use crate::error::{Error, Result};
use crate::gguf::Gguf;

const SPACE: char = '\u{2581}';
const UNK_PENALTY: f64 = 10.0;

// GGUF token types (llama.cpp): 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte.
const TYPE_UNKNOWN: i32 = 2;
const TYPE_CONTROL: i32 = 3;
const TYPE_UNUSED: i32 = 5;
const TYPE_BYTE: i32 = 6;

pub struct Tokenizer {
    pieces: Vec<String>,
    scores: Vec<f64>,
    kinds: Vec<i32>,
    lookup: HashMap<String, u32>,
    max_piece_bytes: usize,
    byte_ids: [Option<u32>; 256],
    unk_score: f64,
    pub unk: u32,
    pub bos: u32,
    pub eos: u32,
    pub sep: u32,
    pub pad: u32,
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer> {
        let pieces: Vec<String> = g
            .get("tokenizer.ggml.tokens")?
            .as_arr()
            .ok_or("tokenizer.ggml.tokens is not an array")?
            .iter()
            .map(|v| v.as_str().map(str::to_string).ok_or_else(|| Error::from("token is not a string")))
            .collect::<Result<_>>()?;
        let scores: Vec<f64> = g
            .get("tokenizer.ggml.scores")?
            .as_arr()
            .ok_or("tokenizer.ggml.scores is not an array")?
            .iter()
            .map(|v| v.as_f32().map(f64::from).ok_or_else(|| Error::from("score is not a float")))
            .collect::<Result<_>>()?;
        let kinds: Vec<i32> = match g.metadata.get("tokenizer.ggml.token_type") {
            Some(v) => v
                .as_arr()
                .ok_or("token_type is not an array")?
                .iter()
                .map(|t| t.as_u32().map(|x| x as i32).ok_or_else(|| Error::from("bad token type")))
                .collect::<Result<_>>()?,
            None => vec![1; pieces.len()],
        };
        if scores.len() != pieces.len() || kinds.len() != pieces.len() {
            return Err("tokenizer arrays differ in length".into());
        }
        let id_of = |key: &str| g.get(key).ok().and_then(|v| v.as_u32());
        let unk = id_of("tokenizer.ggml.unknown_token_id").ok_or("missing unknown_token_id")?;
        let bos = id_of("tokenizer.ggml.bos_token_id").ok_or("missing bos_token_id")?;
        let eos = id_of("tokenizer.ggml.eos_token_id").ok_or("missing eos_token_id")?;
        let pad = id_of("tokenizer.ggml.padding_token_id").unwrap_or(0);
        let sep = pieces.iter().position(|p| p == "<sep>").ok_or("vocab has no <sep>")? as u32;

        let mut lookup = HashMap::new();
        let mut byte_ids = [None; 256];
        let mut max_piece_bytes = 0;
        for (i, p) in pieces.iter().enumerate() {
            match kinds[i] {
                TYPE_BYTE => {
                    if let Some(b) = parse_byte_piece(p) {
                        byte_ids[b as usize] = Some(i as u32);
                    }
                }
                TYPE_UNKNOWN | TYPE_CONTROL | TYPE_UNUSED => {}
                _ => {
                    max_piece_bytes = max_piece_bytes.max(p.len());
                    lookup.entry(p.clone()).or_insert(i as u32);
                }
            }
        }
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        Ok(Tokenizer { pieces, scores, kinds, lookup, max_piece_bytes, byte_ids, unk_score: min_score - UNK_PENALTY, unk, bos, eos, sep, pad })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_word_start(&self, id: u32) -> bool {
        self.pieces[id as usize].starts_with(SPACE)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        // A run of whitespace becomes one space, as the "\s+" -> " " normalizer does.
        let mut s = String::with_capacity(text.len() + 3);
        let mut in_space = false;
        for c in text.nfkc() {
            if c.is_whitespace() {
                if !in_space {
                    s.push(SPACE);
                }
                in_space = true;
            } else {
                s.push(c);
                in_space = false;
            }
        }
        if s.is_empty() {
            return Vec::new();
        }
        if !s.starts_with(SPACE) {
            s.insert(0, SPACE);
        }
        let mut ids = Vec::new();
        let mut start = 0;
        for (i, c) in s.char_indices() {
            if c == SPACE && i > start {
                self.encode_word(&s[start..i], &mut ids);
                start = i;
            }
        }
        self.encode_word(&s[start..], &mut ids);
        ids
    }

    fn encode_word(&self, word: &str, out: &mut Vec<u32>) {
        if word.is_empty() {
            return;
        }
        struct Node {
            id: Option<u32>,
            score: f64,
            starts_at: Option<usize>,
        }
        let n = word.len();
        let mut best: Vec<Node> = (0..=n).map(|_| Node { id: None, score: 0.0, starts_at: None }).collect();
        let mut start = 0;
        while start < n {
            let here = best[start].score;
            let char_len = word[start..].chars().next().unwrap().len_utf8();
            let mut has_single = false;
            let limit = (start + self.max_piece_bytes).min(n);
            for end in start + 1..=limit {
                if !word.is_char_boundary(end) {
                    continue;
                }
                if let Some(&id) = self.lookup.get(&word[start..end]) {
                    let cand = here + self.scores[id as usize];
                    let t = &mut best[end];
                    if t.starts_at.is_none() || cand > t.score {
                        *t = Node { id: Some(id), score: cand, starts_at: Some(start) };
                    }
                    if end - start == char_len {
                        has_single = true;
                    }
                }
            }
            if !has_single {
                let cand = here + self.unk_score;
                let t = &mut best[start + char_len];
                if t.starts_at.is_none() || cand > t.score {
                    *t = Node { id: None, score: cand, starts_at: Some(start) };
                }
            }
            start += char_len;
        }
        let mut path: Vec<(usize, usize, Option<u32>)> = Vec::new();
        let mut end = n;
        while end > 0 {
            let node = &best[end];
            let s = node.starts_at.expect("every position is reachable");
            path.push((s, end, node.id));
            end = s;
        }
        path.reverse();
        // Adjacent unknown characters fuse into one span, then spell out as bytes.
        let mut i = 0;
        while i < path.len() {
            match path[i].2 {
                Some(id) => out.push(id),
                None => {
                    let from = path[i].0;
                    let mut j = i;
                    while j + 1 < path.len() && path[j + 1].2.is_none() {
                        j += 1;
                    }
                    let span = &word[from..path[j].1];
                    let ids: Option<Vec<u32>> = span.bytes().map(|b| self.byte_ids[b as usize]).collect();
                    match ids {
                        Some(ids) => out.extend(ids),
                        None => out.push(self.unk),
                    }
                    i = j;
                }
            }
            i += 1;
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        let mut bytes: Vec<u8> = Vec::new();
        let flush = |bytes: &mut Vec<u8>, out: &mut String| {
            if bytes.is_empty() {
                return;
            }
            match std::str::from_utf8(bytes) {
                Ok(s) => out.push_str(s),
                Err(_) => out.extend(std::iter::repeat('\u{fffd}').take(bytes.len())),
            }
            bytes.clear();
        };
        for &id in ids {
            let Some(kind) = self.kinds.get(id as usize) else { continue };
            if matches!(*kind, TYPE_UNKNOWN | TYPE_CONTROL | TYPE_UNUSED) {
                continue;
            }
            let piece = &self.pieces[id as usize];
            if *kind == TYPE_BYTE {
                if let Some(b) = parse_byte_piece(piece) {
                    bytes.push(b);
                    continue;
                }
            }
            flush(&mut bytes, &mut out);
            out.extend(piece.chars().map(|c| if c == SPACE { ' ' } else { c }));
        }
        flush(&mut bytes, &mut out);
        match out.strip_prefix(' ') {
            Some(rest) => rest.to_string(),
            None => out,
        }
    }
}

fn parse_byte_piece(p: &str) -> Option<u8> {
    let hex = p.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}
