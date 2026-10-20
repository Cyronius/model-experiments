//! Dequantize GGML tensor types to f32. Block layouts follow ggml's `dequantize_row_*`.

use titler_core::{Error, Result};

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q4_0: u32 = 2;
pub const Q8_0: u32 = 8;
pub const Q4_K: u32 = 12;
pub const Q5_K: u32 = 13;
pub const Q6_K: u32 = 14;

const QK_K: usize = 256;

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // Subnormal half: normalize into a normal float.
            let shift = frac.leading_zeros() - 21;
            let frac = (frac << shift) & 0x3ff;
            (sign << 31) | ((113 - shift) << 23) | (frac << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

fn half(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

// Scale and min of sub-block `j` of a Q4_K / Q5_K block (6 bits each, packed into 12 bytes).
fn scale_min(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        ((s[j + 4] & 0xf) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4))
    }
}

pub fn block_size(dtype: u32) -> Result<(usize, usize)> {
    Ok(match dtype {
        F32 => (1, 4),
        F16 => (1, 2),
        Q4_0 => (32, 18),
        Q8_0 => (32, 34),
        Q4_K => (QK_K, 144),
        Q5_K => (QK_K, 176),
        Q6_K => (QK_K, 210),
        _ => return Err(Error(format!("unsupported tensor type {dtype}"))),
    })
}

pub fn dequantize(dtype: u32, data: &[u8], n: usize) -> Result<Vec<f32>> {
    let (per_block, bytes) = block_size(dtype)?;
    if n % per_block != 0 || data.len() < n / per_block * bytes {
        return Err("tensor data is too short".into());
    }
    let mut out = vec![0f32; n];
    for (bi, block) in data.chunks_exact(bytes).take(n / per_block).enumerate() {
        let y = &mut out[bi * per_block..(bi + 1) * per_block];
        match dtype {
            F32 => y[0] = f32::from_le_bytes(block.try_into().unwrap()),
            F16 => y[0] = half(block, 0),
            Q4_0 => {
                let d = half(block, 0);
                for j in 0..16 {
                    y[j] = ((block[2 + j] & 0xf) as i32 - 8) as f32 * d;
                    y[j + 16] = ((block[2 + j] >> 4) as i32 - 8) as f32 * d;
                }
            }
            Q8_0 => {
                let d = half(block, 0);
                for j in 0..32 {
                    y[j] = block[2 + j] as i8 as f32 * d;
                }
            }
            Q4_K => {
                let (d, dmin) = (half(block, 0), half(block, 2));
                let (scales, qs) = (&block[4..16], &block[16..144]);
                for g in 0..4 {
                    let (sc1, m1) = scale_min(2 * g, scales);
                    let (sc2, m2) = scale_min(2 * g + 1, scales);
                    let (d1, mm1, d2, mm2) = (d * sc1 as f32, dmin * m1 as f32, d * sc2 as f32, dmin * m2 as f32);
                    for l in 0..32 {
                        let q = qs[32 * g + l];
                        y[64 * g + l] = d1 * (q & 0xf) as f32 - mm1;
                        y[64 * g + 32 + l] = d2 * (q >> 4) as f32 - mm2;
                    }
                }
            }
            Q5_K => {
                let (d, dmin) = (half(block, 0), half(block, 2));
                let (scales, qh, qs) = (&block[4..16], &block[16..48], &block[48..176]);
                for g in 0..4 {
                    let (sc1, m1) = scale_min(2 * g, scales);
                    let (sc2, m2) = scale_min(2 * g + 1, scales);
                    let (d1, mm1, d2, mm2) = (d * sc1 as f32, dmin * m1 as f32, d * sc2 as f32, dmin * m2 as f32);
                    let (u1, u2) = (1u8 << (2 * g), 2u8 << (2 * g));
                    for l in 0..32 {
                        let q = qs[32 * g + l];
                        let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                        let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                        y[64 * g + l] = d1 * ((q & 0xf) + hi1) as f32 - mm1;
                        y[64 * g + 32 + l] = d2 * ((q >> 4) + hi2) as f32 - mm2;
                    }
                }
            }
            Q6_K => {
                let d = half(block, 208);
                let (ql, qh, sc) = (&block[0..128], &block[128..192], &block[192..208]);
                for half_idx in 0..2 {
                    let (ql, qh, sc) = (&ql[64 * half_idx..], &qh[32 * half_idx..], &sc[8 * half_idx..]);
                    let y = &mut y[128 * half_idx..];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    Ok(out)
}
