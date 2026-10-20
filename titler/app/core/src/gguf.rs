//! Reads only the metadata block of a GGUF file (the vocab lives there); tensors are the backend's job.

use std::collections::HashMap;

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u32(&self) -> Option<u32> {
        match *self {
            Value::U8(v) => Some(v as u32),
            Value::U16(v) => Some(v as u32),
            Value::U32(v) => Some(v),
            Value::I32(v) if v >= 0 => Some(v as u32),
            Value::U64(v) => u32::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: `dims[0]` is the contiguous (innermost) dimension.
    pub dims: Vec<u64>,
    pub dtype: u32,
    /// Byte offset from the start of the tensor data block.
    pub offset: u64,
}

pub struct Gguf {
    pub metadata: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    /// Byte offset of the tensor data block in the file.
    pub data_start: usize,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len()).ok_or("GGUF: unexpected end of file")?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| Error::from("GGUF: string is not UTF-8"))
    }

    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let elem = self.u32()?;
                let n = self.u64()? as usize;
                let mut out = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    out.push(self.value(elem)?);
                }
                Value::Arr(out)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            _ => return Err(Error(format!("GGUF: unknown value type {ty}"))),
        })
    }
}

impl Gguf {
    pub fn parse(bytes: &[u8]) -> Result<Gguf> {
        let mut r = Reader { buf: bytes, pos: 0 };
        if r.take(4)? != b"GGUF" {
            return Err("not a GGUF file".into());
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error(format!("GGUF: unsupported version {version}")));
        }
        let tensor_count = r.u64()?;
        let kv_count = r.u64()?;
        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let key = r.string()?;
            let ty = r.u32()?;
            metadata.insert(key, r.value(ty)?);
        }
        let mut tensors = Vec::new();
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let dtype = r.u32()?;
            let offset = r.u64()?;
            tensors.push(TensorInfo { name, dims, dtype, offset });
        }
        let align = metadata.get("general.alignment").and_then(Value::as_u32).unwrap_or(32) as usize;
        let data_start = r.pos.div_ceil(align) * align;
        Ok(Gguf { metadata, tensors, data_start })
    }

    pub fn get(&self, key: &str) -> Result<&Value> {
        self.metadata.get(key).ok_or_else(|| Error(format!("GGUF: missing key {key}")))
    }
}
