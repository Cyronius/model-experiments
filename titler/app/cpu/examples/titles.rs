//! Titles for a JSONL file of {"id", "prompt"} rows, written as {"id": "title"} JSON (for the Python judge scripts).
//!
//!     cargo run --release -p titler-cpu --example titles -- MODEL.gguf IN.jsonl OUT.json [--beams N] [--threads N] [--limit N]

use std::collections::BTreeMap;
use std::fs;
use std::time::Instant;

use titler_core::{Options, Titler};
use std::sync::Arc;
use titler_cpu::{Model, Session};

fn flag(args: &[String], name: &str, default: usize) -> usize {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (model, input, output) = (&args[1], &args[2], &args[3]);
    let beams = flag(&args, "--beams", 4);
    let threads = flag(&args, "--threads", 2) as i32;
    let limit = flag(&args, "--limit", usize::MAX);

    let titler = Titler::from_gguf_bytes(&fs::read(model).unwrap(), Options { beams, ..Options::default() }).unwrap();
    let _ = threads;
    let mut session = Session::new(Arc::new(Model::from_gguf_bytes(&fs::read(model).unwrap()).unwrap()), beams);

    let mut titles = BTreeMap::new();
    let mut times = Vec::new();
    for line in fs::read_to_string(input).unwrap().lines().filter(|l| !l.trim().is_empty()).take(limit) {
        let row: serde_json::Value = serde_json::from_str(line).unwrap();
        let start = Instant::now();
        let title = titler.title(&mut session, row["prompt"].as_str().unwrap()).unwrap();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        titles.insert(row["id"].as_str().unwrap().to_string(), title);
    }
    fs::write(output, serde_json::to_string(&titles).unwrap()).unwrap();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |q: f64| times[((times.len() - 1) as f64 * q) as usize];
    eprintln!("{} titles, beams {beams}: p50 {:.1} ms  p95 {:.1} ms  max {:.1} ms", times.len(), pct(0.5), pct(0.95), times[times.len() - 1]);
}
