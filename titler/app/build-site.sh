#!/usr/bin/env bash
# Builds the website into app/website/dist: page, worker, WebAssembly module and model. Run from app/.
set -euo pipefail
cd "$(dirname "$0")"
cargo build --release -p titler-wasm --target wasm32-unknown-unknown
rm -rf website/dist
mkdir -p website/dist
wasm-bindgen --target web --out-dir website/dist/pkg --out-name titler_wasm target/wasm32-unknown-unknown/release/titler_wasm.wasm
cp website/index.html website/worker.js website/dist/
cp model/titler-v3-q4_k_m.gguf website/dist/titler.gguf
echo "site ready in app/website/dist"
