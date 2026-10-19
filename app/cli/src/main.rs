use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use titler_core::{Options, Titler};
use titler_llama::Engine;

const MODEL: &[u8] = include_bytes!("../../model/titler-v3-q4_k_m.gguf");

const HELP: &str = "titler: short titles for a prompt, from a 4 MB model running on your CPU

usage:
  titler \"how do I sort a list of dicts by a key in python\"
  echo \"some long text\" | titler
  titler -i                 one prompt per line, one title per line

options:
  -i, --interactive   read prompts line by line until end of input
  --beams N           beam width (default 4)
  --threads N         CPU threads (default 2)
  --model PATH        use this GGUF instead of the built-in model
  --time              print the time for each title
  -h, --help          show this help";

struct Args {
    text: Vec<String>,
    interactive: bool,
    beams: usize,
    threads: i32,
    model: Option<PathBuf>,
    time: bool,
}

fn parse() -> Result<Args, String> {
    let mut a = Args { text: Vec::new(), interactive: false, beams: 4, threads: 2, model: None, time: false };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            "-i" | "--interactive" => a.interactive = true,
            "--time" => a.time = true,
            "--beams" => a.beams = value("--beams")?.parse().map_err(|_| "--beams needs a number")?,
            "--threads" => a.threads = value("--threads")?.parse().map_err(|_| "--threads needs a number")?,
            "--model" => a.model = Some(PathBuf::from(value("--model")?)),
            _ => a.text.push(arg),
        }
    }
    if a.beams == 0 || a.threads < 1 {
        return Err("--beams and --threads must be at least 1".into());
    }
    Ok(a)
}

// llama.cpp loads models from a path, so the built-in model is written out once.
fn builtin_model_path() -> io::Result<PathBuf> {
    let path = std::env::temp_dir().join(format!("titler-model-{}.gguf", MODEL.len()));
    if std::fs::metadata(&path).map_or(true, |m| m.len() as usize != MODEL.len()) {
        std::fs::write(&path, MODEL)?;
    }
    Ok(path)
}

fn run() -> Result<(), String> {
    let args = parse()?;
    let (path, bytes) = match &args.model {
        Some(p) => (p.clone(), std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?),
        None => (builtin_model_path().map_err(|e| e.to_string())?, MODEL.to_vec()),
    };
    let opts = Options { beams: args.beams, ..Options::default() };
    let titler = Titler::from_gguf_bytes(&bytes, opts).map_err(|e| e.to_string())?;
    let engine = Engine::load(&path).map_err(|e| e.to_string())?;
    let mut session = engine.session(args.beams, args.threads).map_err(|e| e.to_string())?;

    let mut title_of = |prompt: &str| -> Result<(), String> {
        let start = Instant::now();
        let title = titler.title(&mut session, prompt).map_err(|e| e.to_string())?;
        if args.time {
            println!("{title}  ({:.0} ms)", start.elapsed().as_secs_f64() * 1000.0);
        } else {
            println!("{title}");
        }
        io::stdout().flush().ok();
        Ok(())
    };

    if !args.text.is_empty() {
        return title_of(&args.text.join(" "));
    }
    let stdin = io::stdin();
    if args.interactive {
        if stdin.is_terminal() {
            eprintln!("type a prompt and press Enter; Ctrl-Z then Enter (Ctrl-D on Linux) to quit");
        }
        for line in stdin.lock().lines() {
            let line = line.map_err(|e| e.to_string())?;
            if !line.trim().is_empty() {
                title_of(&line)?;
            }
        }
        return Ok(());
    }
    if stdin.is_terminal() {
        println!("{HELP}");
        return Ok(());
    }
    let mut text = String::new();
    stdin.lock().read_to_string(&mut text).map_err(|e| e.to_string())?;
    title_of(&text)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("titler: {e}");
        std::process::exit(1);
    }
}
