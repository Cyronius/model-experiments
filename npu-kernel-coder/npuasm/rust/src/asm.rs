//! Text assembler for AIE2P bundles, the inverse of hipfire's
//! `DecodedInst::assembly` (which matches llvm-objdump's syntax).
//!
//! A bundle is `slot; slot; ...`. Each slot is matched against every
//! encoding with the same mnemonic: the encoding's `asm` template is split
//! into literal text and `$operand` placeholders and matched with whitespace
//! ignored. Register operands resolve through the encoding's register table;
//! immediates are the printed (architectural) value. The slot assignment and
//! composite format are then searched, and a candidate is accepted only if
//! hipfire's decoder turns its bytes back into the same text. So every byte
//! sequence this returns re-disassembles to its input.

use peacemaker_npu::isa::bundle::Bundle;
use crate::decode_ext::decode;
use peacemaker_npu::isa::decode::{DecodedBundle, DecodedInst, DecodedOperand};
use peacemaker_npu::isa::gen::{self, Encoding, Format, Operand};
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
enum Tok {
    Lit(String),
    Op(&'static str),
}

fn squash(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

fn template(asm: &'static str) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut rest = asm;
    let mut lit = String::new();
    while let Some(i) = rest.find('$') {
        lit.push_str(&rest[..i]);
        rest = &rest[i + 1..];
        let braced = rest.starts_with('{');
        if braced {
            rest = &rest[1..];
        }
        let end = rest
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .unwrap_or(rest.len());
        let name = &rest[..end];
        rest = &rest[end..];
        if braced {
            rest = rest.strip_prefix('}').unwrap_or(rest);
        }
        let l = squash(&lit);
        if !l.is_empty() {
            out.push(Tok::Lit(l));
        }
        lit.clear();
        out.push(Tok::Op(name));
    }
    lit.push_str(rest);
    let l = squash(&lit);
    if !l.is_empty() {
        out.push(Tok::Lit(l));
    }
    out
}

struct Table {
    by_mnemonic: HashMap<&'static str, Vec<(&'static Encoding, Vec<Tok>)>>,
}

fn table() -> &'static Table {
    static T: OnceLock<Table> = OnceLock::new();
    T.get_or_init(|| {
        let mut by_mnemonic: HashMap<&'static str, Vec<(&'static Encoding, Vec<Tok>)>> = HashMap::new();
        for e in gen::ENCODINGS {
            by_mnemonic.entry(e.mnemonic).or_default().push((e, template(e.asm)));
        }
        Table { by_mnemonic }
    })
}

/// Match `s` (whitespace removed) against tokens; push (operand, text) pairs.
fn match_toks<'a>(toks: &[Tok], s: &'a str, out: &mut Vec<(&'static str, &'a str)>) -> bool {
    match toks.first() {
        None => s.is_empty(),
        Some(Tok::Lit(l)) => s.starts_with(l.as_str()) && match_toks(&toks[1..], &s[l.len()..], out),
        Some(Tok::Op(name)) => {
            // An operand is a non-empty run; try every split, shortest first.
            for end in 1..=s.len() {
                if !s.is_char_boundary(end) {
                    continue;
                }
                out.push((name, &s[..end]));
                if match_toks(&toks[1..], &s[end..], out) {
                    return true;
                }
                out.pop();
            }
            false
        }
    }
}

fn parse_int(t: &str) -> Option<i64> {
    let (neg, body) = match t.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, t),
    };
    let v = if let Some(h) = body.strip_prefix("0x") {
        i64::from_str_radix(h, 16).ok()?
    } else {
        body.parse::<i64>().ok()?
    };
    Some(if neg { -v } else { v })
}

fn operand_value(op: &'static Operand, text: &str) -> Option<i64> {
    if !op.registers.is_empty() {
        // resolve_register compares the field's low bits, so store the
        // architectural index masked to the field width when known.
        let r = op.registers.iter().find(|r| r.asm == text)?;
        let mask = if op.width > 0 { (1u64 << op.width) - 1 } else { u64::MAX };
        return Some((r.value & mask) as i64);
    }
    let v = parse_int(text)?;
    if let Some(off) = op.printer.strip_prefix("printImmOffset<") {
        let off: i64 = off.trim_end_matches('>').trim_start_matches("/*offset=*/").parse().ok()?;
        return Some(v - off);
    }
    Some(v)
}

/// All instruction readings of one slot's text.
fn slot_candidates(text: &str) -> Vec<DecodedInst> {
    let s = squash(text);
    let mnemonic = text.split_whitespace().next().unwrap_or("");
    let mut out = Vec::new();
    let Some(list) = table().by_mnemonic.get(mnemonic) else { return out };
    for (e, toks) in list {
        let mut pairs = Vec::new();
        if !match_toks(toks, &s, &mut pairs) {
            continue;
        }
        let mut ops = Vec::with_capacity(e.operands.len());
        let mut ok = true;
        for op in e.operands {
            let value = match pairs.iter().find(|(n, _)| *n == op.name) {
                Some((_, t)) => match operand_value(op, t) {
                    Some(v) => v,
                    None => {
                        ok = false;
                        break;
                    }
                },
                // Not printed (implicit or tied); the encoder ignores operands
                // without fields, and tied fields come from the printed twin.
                None => 0,
            };
            ops.push(DecodedOperand { name: op.name, kind: op.kind, value, output: op.output });
        }
        // Tied operands ($a = $b) share one printed value.
        if ok {
            for c in e.constraints.split(',') {
                if let Some((a, b)) = c.split_once('=') {
                    let a = a.trim().trim_start_matches('$');
                    let b = b.trim().trim_start_matches('$');
                    let va = ops.iter().find(|o| o.name == a).map(|o| o.value);
                    let vb = ops.iter().find(|o| o.name == b).map(|o| o.value);
                    let a_printed = pairs.iter().any(|(n, _)| *n == a);
                    let b_printed = pairs.iter().any(|(n, _)| *n == b);
                    if let (Some(va), Some(vb)) = (va, vb) {
                        for o in ops.iter_mut() {
                            if o.name == b && a_printed && !b_printed {
                                o.value = va;
                            } else if o.name == a && b_printed && !a_printed {
                                o.value = vb;
                            }
                        }
                    }
                }
            }
            out.push(DecodedInst { encoding: e, raw: 0, operands: ops });
        }
    }
    out
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Stats {
    pub ambiguous: u64,
}

fn slot_mask(f: &Format) -> u8 {
    f.slots.iter().fold(0u8, |m, s| m | 1 << s.slot as usize)
}

/// Assemble one bundle's text. Returns every distinct byte encoding that
/// disassembles back to the same text, shortest first.
pub fn assemble_all(text: &str) -> Result<Vec<Vec<u8>>, String> {
    let slots: Vec<&str> = text.split(';').map(str::trim).filter(|s| !s.is_empty()).collect();
    if slots.is_empty() {
        return Err("empty bundle".into());
    }
    let cands: Vec<Vec<DecodedInst>> = slots.iter().map(|s| slot_candidates(s)).collect();
    if let Some(i) = cands.iter().position(|c| c.is_empty()) {
        return Err(format!("no encoding matches `{}`", slots[i]));
    }
    let want = squash(text);
    let mut found: Vec<Vec<u8>> = Vec::new();
    let mut choice = vec![0usize; slots.len()];
    loop {
        let mut mask = 0u8;
        let mut dup = false;
        for (i, &c) in choice.iter().enumerate() {
            let bit = 1u8 << cands[i][c].encoding.slot as usize;
            dup |= mask & bit != 0;
            mask |= bit;
        }
        if !dup {
            for f in gen::FORMATS.iter().filter(|f| slot_mask(f) == mask) {
                let insts: Vec<DecodedInst> = choice
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| {
                        let d = &cands[i][c];
                        DecodedInst { encoding: d.encoding, raw: 0, operands: d.operands.clone() }
                    })
                    .collect();
                let b = DecodedBundle { pc: 0, len: f.bits as usize / 8, raw: Bundle::new(f), instructions: insts };
                let Ok(bytes) = b.encode() else { continue };
                // Accept only if it disassembles to the same text, in the
                // same slot order the format prints.
                if let Ok(back) = decode(&bytes, 0) {
                    if back.len == bytes.len() && squash(&back.assembly()) == want && !found.contains(&bytes) {
                        found.push(bytes);
                    }
                }
            }
        }
        // next combination
        let mut k = 0;
        loop {
            if k == choice.len() {
                found.sort_by_key(|b| b.len());
                return Ok(found);
            }
            choice[k] += 1;
            if choice[k] < cands[k].len() {
                break;
            }
            choice[k] = 0;
            k += 1;
        }
    }
}

/// Assemble one bundle: the shortest encoding that round-trips.
pub fn assemble(text: &str, stats: &mut Stats) -> Result<Vec<u8>, String> {
    let all = assemble_all(text)?;
    if all.len() > 1 {
        stats.ambiguous += 1;
    }
    all.into_iter().next().ok_or_else(|| format!("no format fits `{text}`"))
}

fn is_label_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '$'
}

/// Replace `#name` references with `#0x<addr>`. Numeric immediates (`#0x..`,
/// `#-..`, `#12`) are left alone.
fn substitute(line: &str, labels: &HashMap<String, u64>, fill: Option<u64>) -> Result<String, String> {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(i) = rest.find('#') {
        out.push_str(&rest[..=i]);
        rest = &rest[i + 1..];
        let first = rest.chars().next().unwrap_or(' ');
        if first.is_ascii_digit() || first == '-' || !is_label_char(first) {
            continue;
        }
        let end = rest.find(|c: char| !is_label_char(c)).unwrap_or(rest.len());
        let name = &rest[..end];
        let v = match fill {
            Some(v) => v,
            None => *labels.get(name).ok_or_else(|| format!("undefined label `{name}`"))?,
        };
        out.push_str(&format!("{v:#x}"));
        rest = &rest[end..];
    }
    out.push_str(rest);
    Ok(out)
}

enum Item<'a> {
    Bytes(Vec<u8>),
    Bundle(usize, &'a str),
    Label(&'a str),
}

/// Assemble an npuasm `.s` file: `.text`, `// comments`, `name:` labels,
/// `.byte` lines and one bundle per line, where an immediate may be `#label`.
/// Two passes: sizes with every label at 0, then real addresses; a bundle
/// whose size changed between the passes is an error. Returns section bytes.
pub fn assemble_file(src: &str, stats: &mut Stats) -> Result<Vec<u8>, String> {
    let mut items = Vec::new();
    for (n, raw) in src.lines().enumerate() {
        let line = match raw.find("//") {
            Some(i) => &raw[..i],
            None => raw,
        }
        .trim();
        if line.is_empty() || line == ".text" {
            continue;
        }
        if let Some(rest) = line.strip_prefix(".byte") {
            let mut b = Vec::new();
            for t in rest.split(',') {
                let t = t.trim();
                b.push(u8::from_str_radix(t.trim_start_matches("0x"), 16).map_err(|_| format!("line {}: bad byte {t}", n + 1))?);
            }
            items.push(Item::Bytes(b));
        } else if let Some(name) = line.strip_suffix(':').filter(|s| s.chars().all(is_label_char)) {
            items.push(Item::Label(name));
        } else {
            items.push(Item::Bundle(n + 1, line));
        }
    }
    // Pass 1: sizes and label addresses.
    let none = HashMap::new();
    let mut labels: HashMap<String, u64> = HashMap::new();
    let mut sizes = Vec::new();
    let mut pc = 0u64;
    let mut scratch = Stats::default();
    for it in &items {
        match it {
            Item::Bytes(b) => pc += b.len() as u64,
            Item::Label(name) => {
                if labels.insert(name.to_string(), pc).is_some() {
                    return Err(format!("label `{name}` defined twice"));
                }
            }
            Item::Bundle(n, line) => {
                let t = substitute(line, &none, Some(0))?;
                let b = assemble(&t, &mut scratch).map_err(|e| format!("line {n}: {e}"))?;
                sizes.push(b.len());
                pc += b.len() as u64;
            }
        }
    }
    // Pass 2: real addresses.
    let mut out = Vec::new();
    let mut k = 0;
    for it in &items {
        match it {
            Item::Bytes(b) => out.extend_from_slice(b),
            Item::Label(_) => {}
            Item::Bundle(n, line) => {
                let t = substitute(line, &labels, None).map_err(|e| format!("line {n}: {e}"))?;
                let b = assemble(&t, stats).map_err(|e| format!("line {n}: {e}"))?;
                if b.len() != sizes[k] {
                    return Err(format!("line {n}: bundle size changed once labels resolved"));
                }
                k += 1;
                out.extend_from_slice(&b);
            }
        }
    }
    Ok(out)
}
