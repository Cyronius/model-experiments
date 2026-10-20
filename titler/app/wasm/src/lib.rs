use std::sync::Arc;

use titler_core::{Options, Titler};
use titler_cpu::{Model, Session};
use wasm_bindgen::prelude::*;

/// A loaded model. Create it once from the GGUF bytes, then call `title` for each prompt.
#[wasm_bindgen]
pub struct WasmTitler {
    titler: Titler,
    model: Arc<Model>,
}

#[wasm_bindgen]
impl WasmTitler {
    #[wasm_bindgen(constructor)]
    pub fn new(gguf: &[u8]) -> Result<WasmTitler, JsError> {
        let titler = Titler::from_gguf_bytes(gguf, Options::default()).map_err(|e| JsError::new(&e.to_string()))?;
        let model = Arc::new(Model::from_gguf_bytes(gguf).map_err(|e| JsError::new(&e.to_string()))?);
        Ok(WasmTitler { titler, model })
    }

    /// `beams` of 4 is the default the model was judged with; 1 is the fastest.
    pub fn title(&self, prompt: &str, beams: usize) -> Result<String, JsError> {
        let beams = beams.clamp(1, 8);
        let mut session = Session::new(self.model.clone(), beams);
        self.titler.title_with_beams(&mut session, prompt, beams).map_err(|e| JsError::new(&e.to_string()))
    }
}
