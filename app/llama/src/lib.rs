//! `titler-core` backend on top of llama.cpp (through llama-cpp-2). One context holds `2 * beams` sequences:
//! beam search alternates between two sets of slots and copies the cache of the parent beam into the new slot.

use std::num::NonZeroU32;
use std::path::Path;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use titler_core::{Backend, Error, Result};

const N_CTX: u32 = 1024;
const N_BATCH: u32 = 512;

fn err(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}

pub struct Engine {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl Engine {
    pub fn load(path: impl AsRef<Path>) -> Result<Engine> {
        let mut backend = LlamaBackend::init().map_err(err)?;
        backend.void_logs();
        let model = LlamaModel::load_from_file(&backend, path, &LlamaModelParams::default()).map_err(err)?;
        Ok(Engine { backend, model })
    }

    /// `beams` is the beam count; the session keeps `2 * beams` sequence slots.
    pub fn session(&self, beams: usize, threads: i32) -> Result<Session<'_>> {
        let slots = 2 * beams.max(1);
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(N_CTX))
            .with_n_batch(N_BATCH)
            .with_n_ubatch(N_BATCH)
            .with_n_seq_max(slots as u32)
            .with_kv_unified(true)
            .with_n_threads(threads)
            .with_n_threads_batch(threads)
            .with_no_perf(true);
        let ctx = self.model.new_context(&self.backend, params).map_err(err)?;
        Ok(Session { ctx, batch: LlamaBatch::new(N_BATCH as usize, 1), pos: vec![0; slots], vocab: self.model.n_vocab() as usize })
    }
}

pub struct Session<'a> {
    ctx: LlamaContext<'a>,
    batch: LlamaBatch<'a>,
    pos: Vec<i32>,
    vocab: usize,
}

impl Backend for Session<'_> {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        self.ctx.kv_cache_seq_rm(slot as i32, None, None).map_err(err)?;
        self.batch.clear();
        let last = tokens.len() - 1;
        for (i, &t) in tokens.iter().enumerate() {
            self.batch.add(LlamaToken(t as i32), i as i32, &[slot as i32], i == last).map_err(err)?;
        }
        self.ctx.decode(&mut self.batch).map_err(err)?;
        self.pos[slot] = tokens.len() as i32;
        Ok(self.ctx.get_logits_ith(last as i32).to_vec())
    }

    fn copy_slot(&mut self, src: usize, dst: usize) -> Result<()> {
        self.ctx.kv_cache_seq_rm(dst as i32, None, None).map_err(err)?;
        self.ctx.kv_cache_seq_cp(src as i32, dst as i32, None, None).map_err(err)?;
        self.pos[dst] = self.pos[src];
        Ok(())
    }

    fn step(&mut self, batch: &[(usize, u32)]) -> Result<Vec<Vec<f32>>> {
        self.batch.clear();
        for &(slot, token) in batch {
            self.batch.add(LlamaToken(token as i32), self.pos[slot], &[slot as i32], true).map_err(err)?;
            self.pos[slot] += 1;
        }
        self.ctx.decode(&mut self.batch).map_err(err)?;
        Ok((0..batch.len()).map(|i| self.ctx.get_logits_ith(i as i32).to_vec()).collect())
    }
}
