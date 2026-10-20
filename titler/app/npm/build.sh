#!/usr/bin/env bash
# Builds the npm package in app/npm: the WebAssembly module, its JavaScript glue and the model. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -p titler-wasm --target wasm32-unknown-unknown
rm -rf npm/pkg
wasm-bindgen --target web --out-dir npm/pkg --out-name titler_wasm target/wasm32-unknown-unknown/release/titler_wasm.wasm
cp model/titler-v2-q4_k_m.gguf npm/titler.gguf
echo "npm package ready in app/npm"
