// Traces: TITLE-PRODUCT-CLAMP, TITLE-APOSTROPHE, TITLE-CASING, TITLE-DECASE-ROUNDTRIP (canonical spec: specs/title/spec.md)
// Golden files come from scripts/24_make_app_golden.py (Python is the source of truth).

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use titler_core::beam::{beam_search, Backend, BeamConfig, Rules};
use titler_core::gguf::Gguf;
use titler_core::text::clean_title;
use titler_core::{Result, Tokenizer};

fn golden(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(name);
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn tokenizer() -> Tokenizer {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../model/titler-v3-q4_k_m.gguf");
    Tokenizer::from_gguf(&Gguf::parse(&fs::read(path).unwrap()).unwrap()).unwrap()
}

#[test]
fn tokenizer_matches_python() {
    let tok = tokenizer();
    let cases = golden("tokenizer.json");
    let mut bad = Vec::new();
    let mut ties = 0;
    for c in cases.as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let want: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let got = tok.encode(text);
        // Exact ties between equal-text splits (long "____" runs) can fall either way: the GGUF keeps f32 scores.
        if got != want && tok.decode(&got) == tok.decode(&want) {
            ties += 1;
        } else if got != want {
            bad.push(format!("{:?}\n  want {:?}\n  got  {:?}", text.chars().take(60).collect::<String>(), want, got));
        }
    }
    assert!(ties <= 3, "{ties} equal-text ties");
    assert!(bad.is_empty(), "{} of {} differ:\n{}", bad.len(), cases.as_array().unwrap().len(), bad.iter().take(8).cloned().collect::<Vec<_>>().join("\n"));
}

#[test]
fn clean_title_matches_python() {
    let cases = golden("clean.json");
    let mut bad = Vec::new();
    for c in cases.as_array().unwrap() {
        let raw = c["raw"].as_str().unwrap();
        let want = c["clean"].as_str().unwrap();
        let got = clean_title(raw);
        if got != want {
            bad.push(format!("{raw:?}\n  want {want:?}\n  got  {got:?}"));
        }
    }
    assert!(bad.is_empty(), "{} differ:\n{}", bad.len(), bad.iter().take(12).cloned().collect::<Vec<_>>().join("\n"));
}

struct Bigram {
    vocab: usize,
    table: Vec<f32>,
    bias: Vec<f32>,
    pos: Vec<usize>,
}

impl Bigram {
    fn logits(&self, token: u32, pos: usize) -> Vec<f32> {
        let row = &self.table[token as usize * self.vocab..][..self.vocab];
        let bias = &self.bias[pos * self.vocab..][..self.vocab];
        row.iter().zip(bias).map(|(a, b)| a + b).collect()
    }
}

impl Backend for Bigram {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        self.pos[slot] = tokens.len();
        Ok(self.logits(*tokens.last().unwrap(), tokens.len() - 1))
    }

    fn copy_slot(&mut self, src: usize, dst: usize) -> Result<()> {
        self.pos[dst] = self.pos[src];
        Ok(())
    }

    fn step(&mut self, batch: &[(usize, u32)]) -> Result<Vec<Vec<f32>>> {
        Ok(batch
            .iter()
            .map(|&(slot, t)| {
                let pos = self.pos[slot];
                self.pos[slot] += 1;
                self.logits(t, pos)
            })
            .collect())
    }
}

#[test]
fn beam_search_matches_huggingface() {
    let cases = golden("decode.json");
    let mut bad = Vec::new();
    for (i, c) in cases.as_array().unwrap().iter().enumerate() {
        let vocab = c["vocab"].as_u64().unwrap() as usize;
        let ids = |k: &str| -> Vec<u32> { c[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect() };
        let table: Vec<f32> = c["table"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let word_starts = c["word_starts"].as_array().map(|ws| {
            let mut flags = vec![false; vocab];
            ws.iter().for_each(|x| flags[x.as_u64().unwrap() as usize] = true);
            flags
        });
        let rules = Rules { word_starts, filler: ids("filler"), filler_penalty: c["penalty"].as_f64().unwrap() as f32 };
        let cfg = BeamConfig { beams: c["beams"].as_u64().unwrap() as usize, max_new: c["max_new"].as_u64().unwrap() as usize, eos: 2 };
        let bias: Vec<f32> = c["bias"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let mut backend = Bigram { vocab, table, bias, pos: vec![0; 2 * cfg.beams] };
        let got = beam_search(&mut backend, &ids("prompt"), &cfg, &rules).unwrap();
        let want = ids("expected");
        if got != want {
            bad.push(format!("case {i} (beams {}, max_new {}): want {want:?} got {got:?}", cfg.beams, cfg.max_new));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
