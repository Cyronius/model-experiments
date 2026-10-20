//! Short titles for prompts, from a 4 MB model that runs on the CPU.
//!
//! ```no_run
//! let titler = titler::Titler::from_path("titler-v3-q4_k_m.gguf")?;
//! println!("{}", titler.title("how do I sort a list of dicts by a key in python")?);
//! # Ok::<(), titler::Error>(())
//! ```
//!
//! Load the model once and share the `Titler` between threads: `title` takes `&self`, so any number of threads can
//! title at the same time.

use std::path::Path;
use std::sync::Arc;

use titler_cpu::{Model, Session};

pub use titler_core::{Error, Options, Result};

pub const DEFAULT_BEAMS: usize = 4;
const MAX_BEAMS: usize = 8;

pub struct Titler {
    core: titler_core::Titler,
    model: Arc<Model>,
}

impl Titler {
    pub fn from_gguf_bytes(bytes: &[u8]) -> Result<Titler> {
        Titler::with_options(bytes, Options::default())
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Titler> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        Titler::from_gguf_bytes(&bytes)
    }

    pub fn with_options(bytes: &[u8], options: Options) -> Result<Titler> {
        Ok(Titler { core: titler_core::Titler::from_gguf_bytes(bytes, options)?, model: Arc::new(Model::from_gguf_bytes(bytes)?) })
    }

    /// Title with the default 4 beams, the setting the model was judged with.
    pub fn title(&self, prompt: &str) -> Result<String> {
        self.title_with_beams(prompt, self.core.options().beams)
    }

    /// 1 beam is fastest; more beams cost time and give slightly better titles. Clamped to 1..=8.
    pub fn title_with_beams(&self, prompt: &str, beams: usize) -> Result<String> {
        let beams = beams.clamp(1, MAX_BEAMS);
        let mut session = Session::new(self.model.clone(), beams);
        self.core.title_with_beams(&mut session, prompt, beams)
    }
}
