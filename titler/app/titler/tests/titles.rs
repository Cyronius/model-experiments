use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use titler::Titler;

fn model() -> Titler {
    Titler::from_path(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../model/titler-v2-q4_k_m.gguf")).unwrap()
}

#[test]
fn titles_a_prompt() {
    let title = model().title("how do I sort a list of dicts by a key in python").unwrap();
    assert!(title.to_lowercase().contains("python") && title.to_lowercase().contains("sort"), "{title}");
    assert!(title.split_whitespace().count() <= 8, "{title}");
}

#[test]
fn threads_can_title_at_once_and_agree_with_one_thread() {
    let titler = Arc::new(model());
    let prompts = ["write a poem about autumn rain", "what is the capital of australia", "fix this rust borrow checker error", "summarize the plot of hamlet"];
    let alone: Vec<String> = prompts.iter().map(|p| titler.title(p).unwrap()).collect();
    let handles: Vec<_> = prompts
        .iter()
        .map(|p| {
            let t = titler.clone();
            let p = p.to_string();
            thread::spawn(move || t.title(&p).unwrap())
        })
        .collect();
    let together: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(alone, together);
}

#[test]
fn bad_input_is_an_error_not_a_crash() {
    assert!(Titler::from_gguf_bytes(b"not a gguf file").is_err());
    assert!(Titler::from_path("does-not-exist.gguf").is_err());
}

#[test]
fn empty_and_huge_prompts_work() {
    let t = model();
    t.title("").unwrap();
    t.title(&"word ".repeat(3000)).unwrap();
    t.title_with_beams("hello there", 0).unwrap();
    t.title_with_beams("hello there", 99).unwrap();
}
