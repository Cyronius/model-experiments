//! Beam search that follows Hugging Face `generate(num_beams=N)` (transformers 4.57, early_stopping=False,
//! length_penalty=1.0) step for step, so Rust titles match the Python run the model was judged with.

use crate::error::Result;

const NEG_INF: f32 = f32::NEG_INFINITY;
const DEAD: f32 = -1.0e9;

/// The model side: slots hold one sequence's cache each. Core never looks inside a slot.
pub trait Backend {
    fn vocab_size(&self) -> usize;
    /// Empty `slot`, feed `tokens` from position 0, return the logits after the last token.
    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>>;
    /// Make `dst` an exact copy of `src` (cache and position).
    fn copy_slot(&mut self, src: usize, dst: usize) -> Result<()>;
    /// Feed one token to each (slot, token) pair and return the logits for each, in order.
    fn step(&mut self, batch: &[(usize, u32)]) -> Result<Vec<Vec<f32>>>;
}

#[derive(Clone, Debug, Default)]
pub struct Rules {
    /// Per token id: does it start a word? `None` turns the repeat ban off.
    pub word_starts: Option<Vec<bool>>,
    pub filler: Vec<u32>,
    pub filler_penalty: f32,
}

#[derive(Clone, Debug)]
pub struct BeamConfig {
    pub beams: usize,
    pub max_new: usize,
    pub eos: u32,
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(NEG_INF, f32::max);
    let sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_sum = sum.ln();
    logits.iter().map(|&x| x - max - log_sum).collect()
}

/// Repeat ban inside the title only, plus a soft penalty on filler words.
fn apply_rules(lp: &mut [f32], generated: &[u32], rules: &Rules) {
    if let (Some(starts), Some(&last)) = (&rules.word_starts, generated.last()) {
        if starts[last as usize] {
            lp[last as usize] = NEG_INF;
        }
        for pair in generated.windows(2) {
            if pair[0] == last {
                lp[pair[1] as usize] = NEG_INF;
            }
        }
    }
    if rules.filler_penalty != 0.0 {
        for &id in &rules.filler {
            lp[id as usize] -= rules.filler_penalty;
        }
    }
}

#[derive(Clone)]
struct Cand {
    score: f32,
    beam: usize,
    token: u32,
    stop: bool,
}

/// Larger score first; on equal scores the lower flat index wins, like `torch.topk`.
fn better(a: (f32, usize), b: (f32, usize)) -> bool {
    a.0 > b.0 || (a.0 == b.0 && a.1 < b.1)
}

fn top_k(rows: &[Vec<f32>], k: usize) -> Vec<(f32, usize)> {
    let v = rows[0].len();
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for (r, row) in rows.iter().enumerate() {
        for (t, &s) in row.iter().enumerate() {
            let item = (s, r * v + t);
            if best.len() == k && !better(item, best[k - 1]) {
                continue;
            }
            let pos = best.iter().position(|&b| better(item, b)).unwrap_or(best.len());
            best.insert(pos, item);
            best.truncate(k);
        }
    }
    best
}

pub fn beam_search<B: Backend>(backend: &mut B, prompt: &[u32], cfg: &BeamConfig, rules: &Rules) -> Result<Vec<u32>> {
    let n = cfg.beams.max(1);
    let k = 2 * n;
    let v = backend.vocab_size();

    let first = backend.prefill(0, prompt)?;
    let mut logits: Vec<Vec<f32>> = vec![first; n];
    let mut slots: Vec<usize> = vec![0; n];
    let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut scores: Vec<f32> = (0..n).map(|i| if i == 0 { 0.0 } else { DEAD }).collect();

    let mut fin_scores: Vec<f32> = vec![DEAD; n];
    let mut fin_tokens: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut fin_flags: Vec<bool> = vec![false; n];
    let mut early_unsatisfied = true;

    for t in 0.. {
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|r| {
                let mut lp = log_softmax(&logits[r]);
                apply_rules(&mut lp, &tokens[r], rules);
                lp.iter_mut().for_each(|x| *x += scores[r]);
                lp
            })
            .collect();
        let cands: Vec<Cand> = top_k(&rows, k)
            .into_iter()
            .map(|(score, idx)| {
                let token = (idx % v) as u32;
                Cand { score, beam: idx / v, token, stop: token == cfg.eos || t + 1 >= cfg.max_new }
            })
            .collect();

        // Next running beams: the best `n` candidates that did not stop (stopped ones sink by 1e9).
        let mut order: Vec<usize> = (0..cands.len()).collect();
        let run_key = |c: &Cand| c.score + if c.stop { DEAD } else { 0.0 };
        order.sort_by(|&a, &b| run_key(&cands[b]).partial_cmp(&run_key(&cands[a])).unwrap_or(std::cmp::Ordering::Equal));
        order.truncate(n);

        // Finished list: only stopped candidates among the first `n` get a real score.
        let mut merged: Vec<(f32, bool, Vec<u32>)> = (0..n).map(|i| (fin_scores[i], fin_flags[i], fin_tokens[i].clone())).collect();
        for (i, c) in cands.iter().enumerate() {
            let mut s = c.score / (t + 1) as f32;
            if !early_unsatisfied {
                s += DEAD;
            }
            let just_finished = c.stop && i < n;
            if !just_finished {
                s += DEAD;
            }
            let mut seq = tokens[c.beam].clone();
            seq.push(c.token);
            merged.push((s, just_finished, seq));
        }
        let mut idx: Vec<usize> = (0..merged.len()).collect();
        idx.sort_by(|&a, &b| merged[b].0.partial_cmp(&merged[a].0).unwrap_or(std::cmp::Ordering::Equal));
        idx.truncate(n);
        for (slot, &m) in idx.iter().enumerate() {
            fin_scores[slot] = merged[m].0;
            fin_flags[slot] = merged[m].1;
            fin_tokens[slot] = merged[m].2.clone();
        }

        let next_scores: Vec<f32> = order.iter().map(|&i| run_key(&cands[i])).collect();
        let best_possible = next_scores[0] / (t + 1) as f32;
        let worst_finished = fin_scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let improvable = fin_flags.iter().any(|&f| best_possible > if f { worst_finished } else { DEAD });
        early_unsatisfied = early_unsatisfied && improvable;
        if !early_unsatisfied || cands.iter().all(|c| c.stop) {
            break;
        }

        let base = if t % 2 == 0 { n } else { 0 };
        let mut batch = Vec::with_capacity(n);
        let mut next_tokens = Vec::with_capacity(n);
        for (i, &ci) in order.iter().enumerate() {
            let c = &cands[ci];
            backend.copy_slot(slots[c.beam], base + i)?;
            batch.push((base + i, c.token));
            let mut seq = tokens[c.beam].clone();
            seq.push(c.token);
            next_tokens.push(seq);
        }
        logits = backend.step(&batch)?;
        slots = (0..n).map(|i| base + i).collect();
        tokens = next_tokens;
        scores = next_scores;
    }

    let best = &fin_tokens[0];
    let end = best.iter().position(|&t| t == cfg.eos).unwrap_or(best.len());
    Ok(best[..end].to_vec())
}
