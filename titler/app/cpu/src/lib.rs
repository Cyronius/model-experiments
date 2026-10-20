//! Pure-Rust CPU backend for `titler-core`: loads the GGUF, dequantizes every weight to f32, and runs the
//! Llama forward pass itself. No C++ and no dependencies, so it also builds for WebAssembly.
//! Weights follow llama.cpp's GGUF layout (RoPE on adjacent pairs).

mod quant;

use std::sync::Arc;

use titler_core::gguf::{Gguf, Value};
use titler_core::{Backend, Error, Result};

struct Layer {
    attn_norm: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    ffn_norm: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
}

pub struct Model {
    n_embd: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    n_ff: usize,
    vocab: usize,
    eps: f32,
    inv_freq: Vec<f32>,
    tok_embd: Vec<f32>,
    out_norm: Vec<f32>,
    layers: Vec<Layer>,
}

impl Model {
    pub fn from_gguf_bytes(bytes: &[u8]) -> Result<Model> {
        let g = Gguf::parse(bytes)?;
        let arch = g.get("general.architecture")?.as_str().ok_or("bad architecture")?.to_string();
        if arch != "llama" {
            return Err(Error(format!("unsupported architecture {arch}")));
        }
        let uint = |key: &str| -> Result<usize> {
            g.get(&format!("llama.{key}"))?.as_u32().map(|v| v as usize).ok_or_else(|| Error(format!("bad {key}")))
        };
        let n_embd = uint("embedding_length")?;
        let n_head = uint("attention.head_count")?;
        let n_kv_head = uint("attention.head_count_kv").unwrap_or(n_head);
        let n_layer = uint("block_count")?;
        let n_ff = uint("feed_forward_length")?;
        let head_dim = uint("rope.dimension_count").unwrap_or(n_embd / n_head);
        let float = |key: &str, default: f32| g.metadata.get(&format!("llama.{key}")).and_then(Value::as_f32).unwrap_or(default);
        let eps = float("attention.layer_norm_rms_epsilon", 1e-5);
        let base = float("rope.freq_base", 10000.0);

        let load = |name: &str, want: &[usize]| -> Result<Vec<f32>> {
            let t = g.tensors.iter().find(|t| t.name == name).ok_or_else(|| Error(format!("missing tensor {name}")))?;
            let dims: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
            if dims != want {
                return Err(Error(format!("tensor {name} has shape {dims:?}, expected {want:?}")));
            }
            let n: usize = dims.iter().product();
            let start = g.data_start + t.offset as usize;
            let (per_block, block_bytes) = quant::block_size(t.dtype)?;
            let end = start + n / per_block * block_bytes;
            let data = bytes.get(start..end).ok_or("tensor lies outside the file")?;
            quant::dequantize(t.dtype, data, n)
        };

        let tok_info = g.tensors.iter().find(|t| t.name == "token_embd.weight").ok_or("missing token_embd.weight")?;
        let vocab = tok_info.dims[1] as usize;
        let (kv_dim, q_dim) = (n_kv_head * head_dim, n_head * head_dim);
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let n = |s: &str| format!("blk.{i}.{s}.weight");
            layers.push(Layer {
                attn_norm: load(&n("attn_norm"), &[n_embd])?,
                wq: load(&n("attn_q"), &[n_embd, q_dim])?,
                wk: load(&n("attn_k"), &[n_embd, kv_dim])?,
                wv: load(&n("attn_v"), &[n_embd, kv_dim])?,
                wo: load(&n("attn_output"), &[q_dim, n_embd])?,
                ffn_norm: load(&n("ffn_norm"), &[n_embd])?,
                w_gate: load(&n("ffn_gate"), &[n_embd, n_ff])?,
                w_up: load(&n("ffn_up"), &[n_embd, n_ff])?,
                w_down: load(&n("ffn_down"), &[n_ff, n_embd])?,
            });
        }
        if g.tensors.iter().any(|t| t.name == "output.weight") {
            return Err("models with a separate output.weight are not supported (this model ties it to token_embd)".into());
        }
        let inv_freq = (0..head_dim / 2).map(|i| base.powf(-2.0 * i as f32 / head_dim as f32)).collect();
        Ok(Model {
            n_embd,
            n_head,
            n_kv_head,
            head_dim,
            n_ff,
            vocab,
            eps,
            inv_freq,
            tok_embd: load("token_embd.weight", &[n_embd, vocab])?,
            out_norm: load("output_norm.weight", &[n_embd])?,
            layers,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// `out[t][o] = dot(w[o], x[t])`: each weight row is read once for all `t` tokens.
fn matmul(w: &[f32], in_dim: usize, x: &[f32], t: usize, out: &mut Vec<f32>) {
    let out_dim = w.len() / in_dim;
    out.clear();
    out.resize(t * out_dim, 0.0);
    for (o, row) in w.chunks_exact(in_dim).enumerate() {
        for ti in 0..t {
            out[ti * out_dim + o] = dot(row, &x[ti * in_dim..(ti + 1) * in_dim]);
        }
    }
}

fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
        *o = v * scale * w;
    }
}

#[derive(Default, Clone)]
struct Slot {
    pos: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

pub struct Session {
    model: Arc<Model>,
    slots: Vec<Slot>,
}

impl Session {
    /// `beams` is the beam count; beam search uses `2 * beams` slots.
    pub fn new(model: Arc<Model>, beams: usize) -> Session {
        let layers = model.layers.len();
        let blank = Slot { pos: 0, k: vec![Vec::new(); layers], v: vec![Vec::new(); layers] };
        Session { model, slots: vec![blank; 2 * beams.max(1)] }
    }

    /// Run each item's tokens through the model after what its slot already holds, all items in one pass so every
    /// weight row is read once. Slots in a batch must differ. Returns the logits of each item's last token.
    fn forward(&mut self, items: &[(usize, &[u32])]) -> Vec<Vec<f32>> {
        let m = &*self.model;
        let (e, hd, nh, nkv) = (m.n_embd, m.head_dim, m.n_head, m.n_kv_head);
        let (q_dim, kv_dim) = (nh * hd, nkv * hd);
        let t: usize = items.iter().map(|(_, toks)| toks.len()).sum();
        let starts: Vec<usize> = items.iter().scan(0, |acc, (_, toks)| { let s = *acc; *acc += toks.len(); Some(s) }).collect();
        let pos0: Vec<usize> = items.iter().map(|&(slot, _)| self.slots[slot].pos).collect();

        let mut x: Vec<f32> = items.iter().flat_map(|(_, toks)| toks.iter()).flat_map(|&id| m.tok_embd[id as usize * e..][..e].iter().copied()).collect();
        let (mut h, mut q, mut k, mut v, mut att, mut o) = (vec![0.0; t * e], vec![], vec![], vec![], vec![0.0; t * q_dim], vec![]);
        let (mut gate, mut up, mut down) = (vec![], vec![], vec![]);
        let mut scores = vec![0f32; items.iter().zip(&pos0).map(|((_, toks), p)| p + toks.len()).max().unwrap_or(0)];
        let scale = 1.0 / (hd as f32).sqrt();
        let group = nh / nkv;

        for (li, layer) in m.layers.iter().enumerate() {
            for ti in 0..t {
                rms_norm(&x[ti * e..(ti + 1) * e], &layer.attn_norm, m.eps, &mut h[ti * e..(ti + 1) * e]);
            }
            matmul(&layer.wq, e, &h, t, &mut q);
            matmul(&layer.wk, e, &h, t, &mut k);
            matmul(&layer.wv, e, &h, t, &mut v);
            for (i, (slot, toks)) in items.iter().enumerate() {
                let n = toks.len();
                for ti in 0..n {
                    let g = starts[i] + ti;
                    let pos = (pos0[i] + ti) as f32;
                    for (buf, heads, stride) in [(&mut q, nh, q_dim), (&mut k, nkv, kv_dim)] {
                        for hh in 0..heads {
                            let base = g * stride + hh * hd;
                            for (j, &f) in m.inv_freq.iter().enumerate() {
                                let (sin, cos) = (pos * f).sin_cos();
                                let (a, b) = (buf[base + 2 * j], buf[base + 2 * j + 1]);
                                buf[base + 2 * j] = a * cos - b * sin;
                                buf[base + 2 * j + 1] = a * sin + b * cos;
                            }
                        }
                    }
                }
                let s = &mut self.slots[*slot];
                s.k[li].extend_from_slice(&k[starts[i] * kv_dim..(starts[i] + n) * kv_dim]);
                s.v[li].extend_from_slice(&v[starts[i] * kv_dim..(starts[i] + n) * kv_dim]);
                let (kc, vc) = (&s.k[li], &s.v[li]);
                for ti in 0..n {
                    let g = starts[i] + ti;
                    let visible = pos0[i] + ti + 1;
                    for hh in 0..nh {
                        let kvh = hh / group;
                        let qv = &q[g * q_dim + hh * hd..][..hd];
                        for p in 0..visible {
                            scores[p] = dot(qv, &kc[p * kv_dim + kvh * hd..][..hd]) * scale;
                        }
                        let max = scores[..visible].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0;
                        for sc in &mut scores[..visible] {
                            *sc = (*sc - max).exp();
                            sum += *sc;
                        }
                        let out = &mut att[g * q_dim + hh * hd..][..hd];
                        out.fill(0.0);
                        for p in 0..visible {
                            let w = scores[p] / sum;
                            for (oi, &vv) in out.iter_mut().zip(&vc[p * kv_dim + kvh * hd..][..hd]) {
                                *oi += w * vv;
                            }
                        }
                    }
                }
            }
            matmul(&layer.wo, q_dim, &att, t, &mut o);
            x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);

            for ti in 0..t {
                rms_norm(&x[ti * e..(ti + 1) * e], &layer.ffn_norm, m.eps, &mut h[ti * e..(ti + 1) * e]);
            }
            matmul(&layer.w_gate, e, &h, t, &mut gate);
            matmul(&layer.w_up, e, &h, t, &mut up);
            gate.iter_mut().zip(&up).for_each(|(g, u)| *g = *g / (1.0 + (-*g).exp()) * u);
            matmul(&layer.w_down, m.n_ff, &gate, t, &mut down);
            x.iter_mut().zip(&down).for_each(|(a, b)| *a += b);
        }
        for (i, (slot, toks)) in items.iter().enumerate() {
            self.slots[*slot].pos = pos0[i] + toks.len();
        }

        let n = items.len();
        let mut last = vec![0.0; n * e];
        for i in 0..n {
            let g = starts[i] + items[i].1.len() - 1;
            rms_norm(&x[g * e..(g + 1) * e], &m.out_norm, m.eps, &mut last[i * e..(i + 1) * e]);
        }
        let mut logits = vec![vec![0f32; m.vocab]; n];
        for (r, row) in m.tok_embd.chunks_exact(e).enumerate() {
            for i in 0..n {
                logits[i][r] = dot(row, &last[i * e..(i + 1) * e]);
            }
        }
        logits
    }
}

impl Backend for Session {
    fn vocab_size(&self) -> usize {
        self.model.vocab
    }

    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        let s = &mut self.slots[slot];
        s.pos = 0;
        s.k.iter_mut().chain(s.v.iter_mut()).for_each(Vec::clear);
        Ok(self.forward(&[(slot, tokens)]).remove(0))
    }

    fn copy_slot(&mut self, src: usize, dst: usize) -> Result<()> {
        let from = self.slots[src].clone();
        self.slots[dst] = from;
        Ok(())
    }

    fn step(&mut self, batch: &[(usize, u32)]) -> Result<Vec<Vec<f32>>> {
        let tokens: Vec<[u32; 1]> = batch.iter().map(|&(_, t)| [t]).collect();
        let items: Vec<(usize, &[u32])> = batch.iter().zip(&tokens).map(|(&(slot, _), t)| (slot, &t[..])).collect();
        Ok(self.forward(&items))
    }
}
