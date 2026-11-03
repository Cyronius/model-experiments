// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Adapted from hipfire `crates/peacemaker-npu/src/isa/decode.rs` (commit
// efa8fbb; Copyright (c) 2026 Kaden Schutt and hipfire contributors; immediate
// and printer semantics derived from llvm-aie AIEBaseDisassembler.h). See
// hipfire's LICENSE and NOTICE. Changed: a fallback for encodings whose
// register operands share one selector field (below).
//! Decode with one extension over hipfire's decoder.
//!
//! `vldb.fill.512` / `vldb.fillx.512` print `[pN, lfN, r2N]` but encode only
//! the pointer bit: the FIFO register and position register have no field,
//! so hipfire's decoder rejects the bundle. In the fallback, an unfielded
//! register operand takes the same index as a fielded operand whose register
//! class has the same size. Checked against llvm-objdump's text on every
//! bundle in the local corpus that needs it.

use peacemaker_npu::isa::bundle::{self, matches_encoding};
use peacemaker_npu::isa::decode::{self as hf, resolve_register, DecodedBundle, DecodedInst, DecodedOperand};
use peacemaker_npu::isa::gen::{self, Encoding, Operand};

pub fn decode(bytes: &[u8], pc: u64) -> Result<DecodedBundle, String> {
    match hf::decode(bytes, pc) {
        Ok(b) => Ok(b),
        Err(e) => decode_tied(bytes, pc).ok_or_else(|| e.to_string()),
    }
}

fn operand_bits(e: &Encoding, name: &str, bits: u64) -> u64 {
    e.fields.iter().filter(|f| f.name == name).fold(0, |v, f| {
        v | (((bits >> f.lsb) & ((1u64 << f.width) - 1)) << f.source_lsb)
    })
}

fn specificity(e: &Encoding) -> u32 {
    let mut count = e.mask.count_ones();
    for (index, field) in e.fields.iter().enumerate() {
        let mask = ((1u64 << field.width) - 1) << field.source_lsb;
        let prior = e.fields[..index]
            .iter()
            .filter(|f| f.name == field.name)
            .fold(0, |m, f| m | (((1u64 << f.width) - 1) << f.source_lsb));
        count += (mask & prior).count_ones();
    }
    count
}

fn transform(op: &Operand) -> (u8, i64, bool, bool) {
    let mut args = op.decoder.split_once('<').map(|(_, s)| s.trim_end_matches('>').split(','));
    let width = args.as_mut().and_then(|a| a.next()).and_then(|s| s.trim().parse().ok()).unwrap_or(op.width);
    if op.decoder.starts_with("decodeSImmOperandXStep") {
        let mut args = args.unwrap();
        let step = args.next().unwrap().trim().parse().unwrap();
        let negative = matches!(args.next().unwrap().trim(), "1" | "true");
        let unsigned = matches!(args.next().unwrap().trim(), "1" | "true");
        (width, step, negative, !unsigned && !negative)
    } else {
        (width, 1, false, op.decoder.starts_with("decodeSImm"))
    }
}

fn immediate(op: &Operand, raw: u64) -> i64 {
    let (width, step, negative, signed) = transform(op);
    let v = if signed && width > 0 { ((raw << (64 - width)) as i64) >> (64 - width) } else { raw as i64 };
    let v = if negative { v - (1i64 << width) } else { v };
    v * step
}

/// hipfire's operand rule, then the shared-selector fallback.
fn operand_value(e: &Encoding, op: &Operand, bits: u64) -> u64 {
    if e.fields.iter().any(|f| f.name == op.name) {
        return operand_bits(e, op.name, bits);
    }
    if op.registers.len() == 1 {
        return op.registers[0].value;
    }
    for constraint in e.constraints.split(',') {
        if let Some((a, b)) = constraint.split_once('=') {
            let a = a.trim().trim_start_matches('$');
            let b = b.trim().trim_start_matches('$');
            if a == op.name {
                return operand_bits(e, b, bits);
            }
            if b == op.name {
                return operand_bits(e, a, bits);
            }
        }
    }
    if op.registers.len() > 1 {
        for sel in e.operands {
            if sel.registers.len() == op.registers.len() && e.fields.iter().any(|f| f.name == sel.name) {
                let v = operand_bits(e, sel.name, bits);
                if let Some(r) = resolve_register(sel, v) {
                    if let Some(i) = sel.registers.iter().position(|x| std::ptr::eq(x, r)) {
                        return op.registers[i].value;
                    }
                }
            }
        }
    }
    0
}

fn decode_tied(bytes: &[u8], pc: u64) -> Option<DecodedBundle> {
    let raw = bundle::unpack(bytes).ok()?;
    let mut instructions = Vec::new();
    for field in raw.format.slots {
        let bits = raw.slots[field.slot as usize]?;
        let encoding = gen::ENCODINGS
            .iter()
            .filter(|e| e.slot == field.slot && matches_encoding(e, bits))
            .filter(|e| {
                e.operands.iter().all(|op| {
                    op.registers.is_empty() || resolve_register(op, operand_value(e, op, bits)).is_some()
                })
            })
            .max_by_key(|e| specificity(e))?;
        let operands = encoding
            .operands
            .iter()
            .map(|op| {
                let value = operand_value(encoding, op, bits);
                DecodedOperand {
                    name: op.name,
                    kind: op.kind,
                    value: if op.registers.is_empty() { immediate(op, value) } else { value as i64 },
                    output: op.output,
                }
            })
            .collect();
        instructions.push(DecodedInst { encoding, raw: bits, operands });
    }
    Some(DecodedBundle { pc, len: raw.format.bits as usize / 8, raw, instructions })
}
