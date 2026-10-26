//! Read CrispASR / llama.cpp GGUF (Q4_K, Q6_K, Q8_0, F16, F32) and dequantize
//! one tensor at a time into f32 for the Burn INT8 packer.
//!
//! Tensor names are the shortened CrispASR spellings (`lm.layers.0.attn.q_proj.weight`).
//! Lookups apply the same `shorten()` rules as `convert-vibevoice-to-gguf.py`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// CrispASR `convert-vibevoice-to-gguf.py` name shortening (must stay in sync).
pub fn gguf_shorten(mut name: &str) -> String {
    if let Some(rest) = name.strip_prefix("model.") {
        name = rest;
    }
    name.replace("tts_eos_classifier.", "tts_eos.")
        .replace("acoustic_tokenizer.encoder.", "at_enc.")
        .replace("acoustic_tokenizer.decoder.", "at_dec.")
        .replace("semantic_tokenizer.encoder.", "st_enc.")
        .replace("semantic_tokenizer.decoder.", "st_dec.")
        .replace("upsample_layers.", "us.")
        .replace("convtr.convtr.", "convtr.")
        .replace("acoustic_connector.", "at_conn.")
        .replace("semantic_connector.", "se_conn.")
        .replace("language_model.", "lm.")
        .replace("prediction_head.", "pred.")
        .replace("downsample_layers.", "ds.")
        .replace("stages.", "s.")
        .replace("mixer.conv.conv.conv.", "dw_conv.")
        .replace("ffn.linear1.", "ffn.up.")
        .replace("ffn.linear2.", "ffn.down.")
        .replace("ffn_norm.", "ffn_ln.")
        .replace("input_layernorm.", "attn_ln.")
        .replace("post_attention_layernorm.", "ffn_ln.")
        .replace("self_attn.", "attn.")
        .replace("mlp.gate_proj.", "ffn.gate.")
        .replace("mlp.up_proj.", "ffn.up.")
        .replace("mlp.down_proj.", "ffn.down.")
        .replace("embed_tokens.", "tok_emb.")
        .replace("head.conv.conv.", "head.")
        .replace("conv.conv.", "conv.")
}

const QK_K: usize = 256;
const QK8_0: usize = 32;
const QK4_0: usize = 32;

#[derive(Clone, Copy, Debug)]
enum GgmlType {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q8_0,
    Q4K,
    Q6K,
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self, String> {
        Ok(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            14 => Self::Q6K,
            30 => Self::Bf16,
            other => return Err(format!("unsupported GGUF ggml type {other}")),
        })
    }

    fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q4_0 | Self::Q8_0 => 32,
            Self::Q4K | Self::Q6K => QK_K,
        }
    }

    fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Q4_0 => 18,
            Self::Q8_0 => 34,
            Self::Q4K => 144,
            Self::Q6K => 210,
        }
    }
}

struct TensorInfo {
    dtype: GgmlType,
    shape: Vec<usize>,
    offset: u64,
}

pub struct GgufFile {
    data: Vec<u8>,
    tensors: HashMap<String, TensorInfo>,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self, String> {
        let data = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        if data.len() < 24 || &data[0..4] != b"GGUF" {
            return Err(format!("{} is not a GGUF file", path.display()));
        }
        let mut p = 4usize;
        let version = read_u32(&data, &mut p)?;
        if version < 2 || version > 3 {
            return Err(format!("unsupported GGUF version {version}"));
        }
        let n_tensors = read_u64(&data, &mut p)? as usize;
        let n_kv = read_u64(&data, &mut p)? as usize;
        let mut alignment = 32u64;
        for _ in 0..n_kv {
            let key = read_string(&data, &mut p)?;
            let val = read_value(&data, &mut p)?;
            if key == "general.alignment" {
                if let GgufVal::U32(v) | GgufVal::U64(v) = val {
                    alignment = v;
                }
            }
        }
        let mut infos = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_string(&data, &mut p)?;
            let n_dims = read_u32(&data, &mut p)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(&data, &mut p)? as usize);
            }
            // GGUF stores ggml dim order (innermost first); HF/numpy is the reverse.
            dims.reverse();
            let dtype = GgmlType::from_u32(read_u32(&data, &mut p)?)?;
            let offset = read_u64(&data, &mut p)?;
            infos.push((name, TensorInfo { dtype, shape: dims, offset }));
        }
        let data_base = align_up(p as u64, alignment);
        let mut tensors = HashMap::new();
        for (name, mut info) in infos {
            info.offset += data_base;
            tensors.insert(name, info);
        }
        Ok(Self { data, tensors })
    }

    pub fn has_name(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn tensor_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("GGUF tensor not found: {name}"))?;
        let n: usize = info.shape.iter().product();
        let bs = info.dtype.block_size();
        let nbytes = if n == 0 {
            0
        } else {
            (n / bs) * info.dtype.block_bytes()
        };
        let start = info.offset as usize;
        let end = start.checked_add(nbytes).ok_or("GGUF offset overflow")?;
        if end > self.data.len() {
            return Err(format!("GGUF tensor {name} out of range"));
        }
        let raw = &self.data[start..end];
        let floats = dequant(info.dtype, raw, n)?;
        Ok((floats, info.shape.clone()))
    }
}

enum GgufVal {
    U32(u64),
    U64(u64),
    Other,
}

fn align_up(n: u64, align: u64) -> u64 {
    let a = align.max(1);
    (n + a - 1) / a * a
}

fn read_u32(data: &[u8], p: &mut usize) -> Result<u32, String> {
    let s = *p;
    *p = s.checked_add(4).ok_or("GGUF truncated")?;
    data.get(s..*p)
        .and_then(|b| b.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| "GGUF truncated".into())
}

fn read_u64(data: &[u8], p: &mut usize) -> Result<u64, String> {
    let s = *p;
    *p = s.checked_add(8).ok_or("GGUF truncated")?;
    data.get(s..*p)
        .and_then(|b| b.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| "GGUF truncated".into())
}

fn read_string(data: &[u8], p: &mut usize) -> Result<String, String> {
    let n = read_u64(data, p)? as usize;
    let s = *p;
    *p = s.checked_add(n).ok_or("GGUF truncated string")?;
    let bytes = data.get(s..*p).ok_or("GGUF truncated string")?;
    Ok(String::from_utf8_lossy(bytes).into_owned())
}

fn skip_value(data: &[u8], p: &mut usize, ty: u32) -> Result<(), String> {
    match ty {
        0 | 1 | 7 => {
            *p += 1;
        }
        2 | 3 => {
            *p += 2;
        }
        4 | 5 | 6 => {
            *p += 4;
        }
        10 | 11 | 12 => {
            *p += 8;
        }
        8 => {
            let _ = read_string(data, p)?;
        }
        9 => {
            let elem = read_u32(data, p)?;
            let n = read_u64(data, p)? as usize;
            for _ in 0..n {
                skip_value(data, p, elem)?;
            }
        }
        other => return Err(format!("unknown GGUF value type {other}")),
    }
    Ok(())
}

fn read_value(data: &[u8], p: &mut usize) -> Result<GgufVal, String> {
    let ty = read_u32(data, p)?;
    match ty {
        4 => Ok(GgufVal::U32(read_u32(data, p)? as u64)),
        10 => Ok(GgufVal::U64(read_u64(data, p)?)),
        _ => {
            skip_value(data, p, ty)?;
            Ok(GgufVal::Other)
        }
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

fn dequant(dtype: GgmlType, raw: &[u8], n: usize) -> Result<Vec<f32>, String> {
    match dtype {
        GgmlType::F32 => {
            if raw.len() < n * 4 {
                return Err("F32 blob short".into());
            }
            Ok(raw
                .chunks_exact(4)
                .take(n)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        GgmlType::F16 => {
            if raw.len() < n * 2 {
                return Err("F16 blob short".into());
            }
            Ok(raw
                .chunks_exact(2)
                .take(n)
                .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect())
        }
        GgmlType::Bf16 => {
            if raw.len() < n * 2 {
                return Err("BF16 blob short".into());
            }
            Ok(raw
                .chunks_exact(2)
                .take(n)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32())
                .collect())
        }
        GgmlType::Q8_0 => dequant_q8_0(raw, n),
        GgmlType::Q4_0 => dequant_q4_0(raw, n),
        GgmlType::Q4K => dequant_q4_k(raw, n),
        GgmlType::Q6K => dequant_q6_k(raw, n),
    }
}

fn dequant_q8_0(raw: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let nb = n / QK8_0;
    if raw.len() < nb * 34 {
        return Err("Q8_0 blob short".into());
    }
    let mut y = vec![0f32; n];
    for i in 0..nb {
        let b = &raw[i * 34..];
        let d = f16_to_f32(u16::from_le_bytes(b[0..2].try_into().unwrap()));
        for j in 0..QK8_0 {
            y[i * QK8_0 + j] = d * (b[2 + j] as i8 as f32);
        }
    }
    Ok(y)
}

fn dequant_q4_0(raw: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let nb = n / QK4_0;
    if raw.len() < nb * 18 {
        return Err("Q4_0 blob short".into());
    }
    let mut y = vec![0f32; n];
    for i in 0..nb {
        let b = &raw[i * 18..];
        let d = f16_to_f32(u16::from_le_bytes(b[0..2].try_into().unwrap()));
        for j in 0..16 {
            let q = b[2 + j];
            y[i * 32 + j] = d * ((q & 0x0F) as i8 as f32 - 8.0);
            y[i * 32 + j + 16] = d * ((q >> 4) as i8 as f32 - 8.0);
        }
    }
    Ok(y)
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn dequant_q4_k(raw: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let nb = n / QK_K;
    if raw.len() < nb * 144 {
        return Err("Q4_K blob short".into());
    }
    let mut y = vec![0f32; n];
    for i in 0..nb {
        let b = &raw[i * 144..];
        let d = f16_to_f32(u16::from_le_bytes(b[0..2].try_into().unwrap()));
        let min = f16_to_f32(u16::from_le_bytes(b[2..4].try_into().unwrap()));
        let scales = &b[4..16];
        let mut q = &b[16..144];
        let mut yo = i * QK_K;
        let mut is = 0usize;
        for _ in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1v = min * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2v = min * m2 as f32;
            for l in 0..32 {
                y[yo + l] = d1 * (q[l] & 0xF) as f32 - m1v;
                y[yo + 32 + l] = d2 * (q[l] >> 4) as f32 - m2v;
            }
            q = &q[32..];
            is += 2;
            yo += 64;
        }
    }
    Ok(y)
}

fn dequant_q6_k(raw: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let nb = n / QK_K;
    if raw.len() < nb * 210 {
        return Err("Q6_K blob short".into());
    }
    let mut y = vec![0f32; n];
    for i in 0..nb {
        let b = &raw[i * 210..];
        let ql = &b[0..128];
        let qh = &b[128..192];
        let sc = &b[192..208];
        let d = f16_to_f32(u16::from_le_bytes(b[208..210].try_into().unwrap()));
        let mut yo = i * QK_K;
        for nblk in 0..2 {
            let ql = &ql[nblk * 64..];
            let qh = &qh[nblk * 32..];
            let sc = &sc[nblk * 8..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) as i8 | (((qh[l] >> 0) & 3) << 4) as i8) - 32;
                let q2 = ((ql[l + 32] & 0xF) as i8 | (((qh[l] >> 2) & 3) << 4) as i8) - 32;
                let q3 = ((ql[l] >> 4) as i8 | (((qh[l] >> 4) & 3) << 4) as i8) - 32;
                let q4 = ((ql[l + 32] >> 4) as i8 | (((qh[l] >> 6) & 3) << 4) as i8) - 32;
                y[yo + l] = d * (sc[is] as i8 as f32) * q1 as f32;
                y[yo + 32 + l] = d * (sc[is + 2] as i8 as f32) * q2 as f32;
                y[yo + 64 + l] = d * (sc[is + 4] as i8 as f32) * q3 as f32;
                y[yo + 96 + l] = d * (sc[is + 6] as i8 as f32) * q4 as f32;
            }
            yo += 128;
        }
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::{get_scale_min_k4, GgufFile};
    use std::path::Path;

    #[test]
    fn k4_scale_unpack_low_j() {
        let q = [10u8, 11, 12, 13, 20, 21, 22, 23, 0, 0, 0, 0];
        let (d, m) = get_scale_min_k4(0, &q);
        assert_eq!(d, 10);
        assert_eq!(m, 20);
    }

    #[test]
    fn opens_local_q4_k_if_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let p = root.join("models/vibevoice-asr/vibevoice-asr-q4_k.gguf");
        if !p.is_file() || p.metadata().map(|m| m.len() < 4_000_000_000).unwrap_or(true) {
            eprintln!("skip: GGUF Q4 not downloaded yet ({})", p.display());
            return;
        }
        let f = GgufFile::open(&p).expect("parse GGUF Q4");
        assert!(
            f.has_name("lm.tok_emb.weight")
                || f.has_name("lm.layers.0.attn.q_proj.weight")
                || f.has_name("lm_head.weight"),
            "expected CrispASR VibeVoice LM tensor names"
        );
        let (w, shape) = f
            .tensor_f32("lm.layers.0.attn.q_proj.weight")
            .or_else(|_| f.tensor_f32("lm.layers.0.attn.q.weight"))
            .expect("dequant q_proj");
        assert!(shape.len() == 2, "q_proj shape {shape:?}");
        assert!(!w.is_empty());
    }
}
