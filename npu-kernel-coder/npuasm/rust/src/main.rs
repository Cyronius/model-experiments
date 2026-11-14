//! npuasm-core: AIE2P core programs without the vendor toolchain.
//!
//!   npuasm-core lift  CORE.elf        one line per bundle: addr <TAB> hex <TAB> text
//!   npuasm-core asm   CORE.s OUT.bin  assemble an npuasm listing to raw .text bytes
//!   npuasm-core check CORE.elf...     decode -> text -> assemble, compare bytes
//!   npuasm-core dis   CODE.bin        decode raw program bytes (from address 0)
//!   npuasm-core bundles < LINES       assemble one bundle per line: ok <TAB> hex | err <TAB> reason
//!
//! Decoding and byte encoding are hipfire's `peacemaker-npu` (Apache-2.0; its
//! generated tables are Apache-2.0 WITH LLVM-exception, derived from llvm-aie
//! TableGen). The text parser is `asm.rs`.

mod asm;
mod decode_ext;

use decode_ext::decode;
use peacemaker_npu::isa::decode::executable_sections;
use std::collections::BTreeMap;
use std::process::ExitCode;

fn text_of(data: &[u8]) -> Result<Vec<u8>, String> {
    let secs = executable_sections(data).map_err(|e| e.to_string())?;
    match secs.as_slice() {
        [s] if s.address == 0 => Ok(s.bytes.to_vec()),
        _ => Err(format!("expected one executable section at 0, found {}", secs.len())),
    }
}

fn lift(path: &str) -> Result<(), String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    let text = text_of(&data)?;
    let mut pc = 0;
    while pc < text.len() {
        let b = decode(&text[pc..], pc as u64)?;
        let raw = &text[pc..pc + b.len];
        let hex: String = raw.iter().map(|x| format!("{x:02x}")).collect();
        println!("{pc:x}\t{hex}\t{}", b.assembly());
        pc += b.len;
    }
    Ok(())
}

#[derive(Default)]
struct Check {
    files: u64,
    files_ok: u64,
    bundles: u64,
    decode_fail: u64,
    reencode_fail: u64,
    asm_fail: u64,
    asm_wrong: u64,
    ambiguous: u64,
    errors: BTreeMap<String, (u64, String)>,
}

fn check(path: &str, c: &mut Check) {
    c.files += 1;
    let Ok(data) = std::fs::read(path) else { return };
    let Ok(text) = text_of(&data) else {
        c.errors.entry("no single .text at 0".into()).or_insert((0, path.into())).0 += 1;
        return;
    };
    let mut pc = 0;
    let mut ok = true;
    let mut st = asm::Stats::default();
    while pc < text.len() {
        let b = match decode(&text[pc..], pc as u64) {
            Ok(b) => b,
            Err(e) => {
                c.decode_fail += 1;
                c.errors.entry("decode".into()).or_insert((0, e)).0 += 1;
                ok = false;
                break;
            }
        };
        c.bundles += 1;
        let raw = &text[pc..pc + b.len];
        if b.encode().map(|v| v != raw).unwrap_or(true) {
            c.reencode_fail += 1;
            ok = false;
        }
        let t = b.assembly();
        match asm::assemble_all(&t) {
            Ok(all) if all.is_empty() => {
                c.asm_fail += 1;
                ok = false;
                c.errors.entry("no format".into()).or_insert((0, t.clone())).0 += 1;
            }
            Ok(all) => {
                if all.len() > 1 {
                    st.ambiguous += 1;
                }
                if all[0] != raw {
                    c.asm_wrong += 1;
                    ok = false;
                    let key = format!("wrong bytes ({} candidates)", all.len());
                    let e = c.errors.entry(key).or_insert((0, String::new()));
                    e.0 += 1;
                    if e.1.is_empty() {
                        let h = |v: &[u8]| v.iter().map(|x| format!("{x:02x}")).collect::<String>();
                        e.1 = format!("{t} | want {} | got {}", h(raw), all.iter().map(|v| h(v)).collect::<Vec<_>>().join(","));
                    }
                }
            }
            Err(e) => {
                c.asm_fail += 1;
                ok = false;
                let key = e.split('`').next().unwrap_or("").to_string() + t.split_whitespace().next().unwrap_or("");
                c.errors.entry(key).or_insert((0, t.clone())).0 += 1;
            }
        }
        pc += b.len;
    }
    c.ambiguous += st.ambiguous;
    if ok {
        c.files_ok += 1;
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("lift") if args.len() == 2 => lift(&args[1]),
        Some("asm") if args.len() == 3 => (|| {
            let src = std::fs::read_to_string(&args[1]).map_err(|e| e.to_string())?;
            let mut st = asm::Stats::default();
            let out = asm::assemble_file(&src, &mut st)?;
            std::fs::write(&args[2], out).map_err(|e| e.to_string())?;
            if st.ambiguous > 0 {
                eprintln!("{} bundles had more than one encoding; used the shortest", st.ambiguous);
            }
            Ok(())
        })(),
        Some("bundles") if args.len() == 1 => (|| {
            // One bundle per stdin line -> "ok <TAB> hex" (shortest encoding) or "err <TAB> reason".
            // Branch targets must be numbers: there are no labels here.
            use std::io::{BufRead, Write};
            let mut out = std::io::BufWriter::new(std::io::stdout().lock());
            for line in std::io::stdin().lock().lines() {
                let line = line.map_err(|e| e.to_string())?;
                let r = match asm::assemble_all(line.trim()) {
                    Ok(all) if !all.is_empty() => {
                        format!("ok\t{}", all[0].iter().map(|x| format!("{x:02x}")).collect::<String>())
                    }
                    Ok(_) => format!("err\tno format fits `{}`", line.trim()),
                    Err(e) => format!("err\t{e}"),
                };
                writeln!(out, "{r}").map_err(|e| e.to_string())?;
            }
            Ok(())
        })(),
        Some("dis") if args.len() == 2 => (|| {
            let text = std::fs::read(&args[1]).map_err(|e| e.to_string())?;
            let mut pc = 0;
            while pc < text.len() {
                let b = decode(&text[pc..], pc as u64)?;
                let hex: String = text[pc..pc + b.len].iter().map(|x| format!("{x:02x}")).collect();
                println!("{pc:x}\t{hex}\t{}", b.assembly());
                pc += b.len;
            }
            Ok(())
        })(),
        Some("rules") => {
            // hipfire's silicon-proven core rules (first violation per program).
            let mut counts: BTreeMap<String, (u64, String)> = BTreeMap::new();
            let mut files = 0;
            for p in &args[1..] {
                files += 1;
                let Ok(data) = std::fs::read(p) else { continue };
                let Ok(text) = text_of(&data) else { continue };
                let key = match peacemaker_npu::isa::rules::check(&text) {
                    Ok(()) => ("pass".to_string(), String::new()),
                    Err(v) => (format!("{:?}", v.rule), format!("{p} @ {:#x}: {}", v.pc, v.detail)),
                };
                let e = counts.entry(key.0).or_insert((0, key.1));
                e.0 += 1;
            }
            println!("programs: {files}");
            for (k, (n, ex)) in &counts {
                println!("{n:8}  {k}  {ex}");
            }
            Ok(())
        }
        Some("check") => {
            let mut c = Check::default();
            for p in &args[1..] {
                check(p, &mut c);
            }
            println!(
                "{{\"files\": {}, \"files_ok\": {}, \"bundles\": {}, \"decode_fail\": {}, \"reencode_fail\": {}, \"asm_fail\": {}, \"asm_wrong\": {}, \"ambiguous\": {}}}",
                c.files, c.files_ok, c.bundles, c.decode_fail, c.reencode_fail, c.asm_fail, c.asm_wrong, c.ambiguous
            );
            for (k, (n, ex)) in &c.errors {
                eprintln!("{n:8}  {k}  e.g. {ex}");
            }
            Ok(())
        }
        _ => Err("usage: npuasm-core lift CORE.elf | asm CORE.s OUT.bin | check CORE.elf...".into()),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("npuasm-core: {e}");
            ExitCode::FAILURE
        }
    }
}
