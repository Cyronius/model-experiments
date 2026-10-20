//! titler-core: prompt in, short title out. Everything except the neural network: tokenizer, beam search with the
//! repeat ban and filler penalty, and title cleanup. A `Backend` supplies the logits.

pub mod beam;
pub mod error;
pub mod gguf;
pub mod text;
pub mod tokenizer;

pub use beam::{Backend, BeamConfig, Rules};
pub use error::{Error, Result};
pub use tokenizer::Tokenizer;

#[derive(Clone, Debug)]
pub struct Options {
    pub beams: usize,
    pub max_new_tokens: usize,
    pub max_prompt_tokens: usize,
    pub filler_penalty: f32,
    pub no_repeat: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { beams: 4, max_new_tokens: 40, max_prompt_tokens: 256, filler_penalty: 3.0, no_repeat: true }
    }
}

pub struct Titler {
    tok: Tokenizer,
    opts: Options,
    rules: Rules,
}

impl Titler {
    pub fn from_gguf_bytes(bytes: &[u8], opts: Options) -> Result<Titler> {
        let tok = Tokenizer::from_gguf(&gguf::Gguf::parse(bytes)?)?;
        let mut filler: Vec<u32> = ["request", "inquiry"].iter().filter_map(|w| tok.encode(w).first().copied()).collect();
        filler.sort_unstable();
        filler.dedup();
        let word_starts = opts.no_repeat.then(|| (0..tok.vocab_size() as u32).map(|i| tok.is_word_start(i)).collect());
        let rules = Rules { word_starts, filler, filler_penalty: opts.filler_penalty };
        Ok(Titler { tok, opts, rules })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    pub fn options(&self) -> &Options {
        &self.opts
    }

    /// BOS, the first `max_prompt_tokens` prompt tokens, then the separator the model was trained with.
    pub fn prompt_ids(&self, prompt: &str) -> Vec<u32> {
        let mut ids = vec![self.tok.bos];
        ids.extend(self.tok.encode(prompt).into_iter().take(self.opts.max_prompt_tokens));
        ids.push(self.tok.sep);
        ids
    }

    pub fn title<B: Backend>(&self, backend: &mut B, prompt: &str) -> Result<String> {
        self.title_with_beams(backend, prompt, self.opts.beams)
    }

    /// The backend must have at least `2 * beams` slots.
    pub fn title_with_beams<B: Backend>(&self, backend: &mut B, prompt: &str, beams: usize) -> Result<String> {
        let cfg = BeamConfig { beams, max_new: self.opts.max_new_tokens, eos: self.tok.eos };
        let ids = beam::beam_search(backend, &self.prompt_ids(prompt), &cfg, &self.rules)?;
        Ok(text::clean_title(&self.tok.decode(&ids)))
    }
}
