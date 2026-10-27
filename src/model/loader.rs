//! Model weights -> Burn tensors: memory-mapped HF safetensors (single file or sharded), or a
//! GGUF file (llama.cpp naming, mapped back to HF names). GGUF `Q8_0` linear weights stay
//! packed as Burn `Q8F` block-32 quantized tensors; other quant types are dequantized at load.
use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::{DType, Device, Tensor, TensorCreationOptions, TensorData};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct Weights {
    maps: Vec<Mmap>,
    index: HashMap<String, usize>,
    gguf: Option<Arc<burn_gguf::GgufReader>>,
    device: Device,
    dtype: DType,
    /// Keep GGUF Q8_0 linear weights quantized (PYRE_GGUF_DEQUANT=1 turns this off).
    pub keep_quant: bool,
}

/// The `.gguf` file to use for `dir`: `dir` itself, or the only/first `*.gguf` inside it when
/// it has no safetensors.
pub fn gguf_path(dir: &Path) -> Option<PathBuf> {
    if dir.extension().and_then(|e| e.to_str()) == Some("gguf") { return Some(dir.to_path_buf()); }
    if dir.join("model.safetensors").exists() || dir.join("model.safetensors.index.json").exists() { return None; }
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf")).collect();
    v.sort();
    v.into_iter().next()
}

/// HF parameter name -> llama.cpp GGUF tensor name.
fn hf_to_gguf(name: &str) -> String {
    let fixed = [("model.embed_tokens.weight", "token_embd.weight"), ("model.norm.weight", "output_norm.weight"), ("lm_head.weight", "output.weight")];
    for (a, b) in fixed { if name == a { return b.into(); } }
    if let Some(rest) = name.strip_prefix("model.layers.") {
        let (i, tail) = rest.split_once('.').unwrap_or((rest, ""));
        let map = [
            ("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm"),
            ("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_output"), ("self_attn.q_norm", "attn_q_norm"), ("self_attn.k_norm", "attn_k_norm"),
            ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"), ("mlp.down_proj", "ffn_down"),
        ];
        for (a, b) in map {
            if let Some(suf) = tail.strip_prefix(a) { return format!("blk.{i}.{b}{suf}"); }
        }
    }
    name.to_string()
}

impl Weights {
    pub fn open(dir: &Path, device: Device, dtype: DType) -> Result<Self> {
        let keep_quant = std::env::var("PYRE_GGUF_DEQUANT").map(|v| v != "1").unwrap_or(true);
        if let Some(g) = gguf_path(dir) {
            let r = burn_gguf::GgufReader::open(&g).map_err(|e| anyhow!("gguf {}: {e}", g.display()))?;
            tracing::info!("GGUF {} (arch {:?}, {} tensors)", g.display(), r.architecture(), r.tensors().len());
            return Ok(Self { maps: vec![], index: HashMap::new(), gguf: Some(Arc::new(r)), device, dtype, keep_quant });
        }
        let mut files = vec![];
        let idx = dir.join("model.safetensors.index.json");
        if idx.exists() {
            let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&idx)?)?;
            let mut set: Vec<String> = j["weight_map"].as_object().context("weight_map")?.values()
                .filter_map(|v| v.as_str().map(String::from)).collect();
            set.sort(); set.dedup();
            files.extend(set.into_iter().map(|f| dir.join(f)));
        } else {
            files.push(dir.join("model.safetensors"));
        }
        let mut maps = vec![];
        let mut index = HashMap::new();
        for (i, f) in files.iter().enumerate() {
            let file = std::fs::File::open(f).with_context(|| format!("open {}", f.display()))?;
            let m = unsafe { Mmap::map(&file)? };
            for name in SafeTensors::deserialize(&m)?.names() {
                index.insert(name.to_string(), i);
            }
            maps.push(m);
        }
        Ok(Self { maps, index, gguf: None, device, dtype, keep_quant })
    }

    pub fn is_gguf(&self) -> bool { self.gguf.is_some() }
    pub fn gguf(&self) -> Option<&burn_gguf::GgufReader> { self.gguf.as_deref() }

    pub fn has(&self, name: &str) -> bool {
        match &self.gguf { Some(r) => r.tensor(&hf_to_gguf(name)).is_some(), None => self.index.contains_key(name) }
    }

    /// A tensor's shape (HF order), without loading it.
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        if let Some(r) = &self.gguf {
            let g = hf_to_gguf(name);
            let info = r.tensor(&g).with_context(|| format!("missing tensor {name} (gguf {g})"))?;
            return Ok(info.dims.iter().rev().copied().collect());
        }
        let i = *self.index.get(name).with_context(|| format!("missing tensor {name}"))?;
        Ok(SafeTensors::deserialize(&self.maps[i])?.tensor(name)?.shape().to_vec())
    }

    /// A float tensor in the model dtype (quantized GGUF data is dequantized).
    pub fn get<const D: usize>(&self, name: &str) -> Result<Tensor<D>> {
        let opts = TensorCreationOptions::new(self.device.clone()).with_dtype(self.dtype);
        let (data, _) = self.get_data(name, D)?;
        Ok(Tensor::from_data(data, opts))
    }

    /// Several float tensors concatenated along dim 0, joined on the host and uploaded once (so
    /// fusing q/k/v or gate/up costs no extra device memory, unlike a device-side `cat`, whose
    /// freed inputs stay in the allocator pool).
    pub fn get_cat0<const D: usize>(&self, names: &[String]) -> Result<Tensor<D>> {
        let opts = TensorCreationOptions::new(self.device.clone()).with_dtype(self.dtype);
        let mut bytes = Vec::new();
        let mut shape: Option<Vec<usize>> = None;
        let mut dt = None;
        for n in names {
            let (d, sh) = self.get_data(n, D)?;
            if dt.is_some_and(|t| t != d.dtype()) { bail!("{n}: dtype differs from the other fused tensors"); }
            dt = Some(d.dtype());
            match &mut shape {
                None => shape = Some(sh),
                Some(s) => { if s[1..] != sh[1..] { bail!("{n}: trailing dims {sh:?} differ from {s:?}"); } s[0] += sh[0]; }
            }
            bytes.extend_from_slice(d.bytes());
        }
        let data = TensorData::from_bytes_vec(bytes, shape.context("no tensors")?, dt.unwrap());
        Ok(Tensor::from_data(data, opts))
    }

    fn get_data(&self, name: &str, rank: usize) -> Result<(TensorData, Vec<usize>)> {
        if let Some(r) = &self.gguf {
            let g = hf_to_gguf(name);
            let info = r.tensor(&g).with_context(|| format!("missing tensor {name} (gguf {g})"))?;
            let shape: Vec<usize> = info.dims.iter().rev().copied().collect();
            if shape.len() != rank { bail!("{name}: expected rank {rank}, got {shape:?}"); }
            let bytes = r.tensor_bytes(info).map_err(|e| anyhow!("{g}: {e}"))?;
            let vals = burn_gguf::dequantize_checked(info.tensor_type, &bytes, info.num_elements().map_err(|e| anyhow!("{e}"))?)
                .map_err(|e| anyhow!("{g}: {e}"))?;
            return Ok((TensorData::new(vals, shape.clone()), shape));
        }
        let i = *self.index.get(name).with_context(|| format!("missing tensor {name}"))?;
        let st = SafeTensors::deserialize(&self.maps[i])?;
        let v = st.tensor(name)?;
        let dt = match v.dtype() {
            Dtype::BF16 => DType::BF16,
            Dtype::F16 => DType::F16,
            Dtype::F32 => DType::F32,
            d => bail!("unsupported safetensors dtype {d:?} for {name}"),
        };
        let shape = v.shape().to_vec();
        if shape.len() != rank { bail!("{name}: expected rank {rank}, got {:?}", shape); }
        Ok((TensorData::from_bytes_vec(v.data().to_vec(), shape.clone(), dt), shape))
    }

    /// A `[out, in]` linear weight kept packed when the GGUF type has an exact Burn scheme
    /// (Q8_0 -> Q8F block 32, f16 scales). `None` means: use [`Self::get`].
    pub fn get_quantized(&self, name: &str) -> Result<Option<(Tensor<2>, Option<Tensor<2>>)>> {
        self.get_quantized_cat(&[name.to_string()])
    }

    /// Like [`Self::get_quantized`] for several `[out_i, in]` weights of the same GGUF type, stacked
    /// along `out` on the host before repacking (GGUF blocks run along `in`, rows are contiguous),
    /// so q/k/v or gate/up become one packed weight (and one Q4_K offset matrix). `None` when the
    /// types or `in` differ, or the type isn't kept packed.
    pub fn get_quantized_cat(&self, names: &[String]) -> Result<Option<(Tensor<2>, Option<Tensor<2>>)>> {
        use burn_gguf::GgmlType as G;
        let Some(r) = &self.gguf else { return Ok(None) };
        if !self.keep_quant { return Ok(None); }
        let mut infos = Vec::with_capacity(names.len());
        for name in names {
            let g = hf_to_gguf(name);
            infos.push((g.clone(), r.tensor(&g).with_context(|| format!("missing tensor {name} (gguf {g})"))?));
        }
        let (g, info) = &infos[0];
        let g = g.clone();
        let mut shape: Vec<usize> = info.dims.iter().rev().copied().collect();
        if shape.len() != 2 { return Ok(None); }
        let mut n = 0;
        for (_, i) in &infos {
            let sh: Vec<usize> = i.dims.iter().rev().copied().collect();
            if i.tensor_type != info.tensor_type || sh.len() != 2 || sh[1] != shape[1] { return Ok(None); }
            n += i.num_elements().map_err(|e| anyhow!("{e}"))?;
        }
        shape[0] = n / shape[1];
        let raw = || -> Result<Vec<u8>> {
            let mut b = Vec::new();
            for (gn, i) in &infos { b.extend_from_slice(&r.tensor_bytes(i).map_err(|e| anyhow!("{gn}: {e}"))?); }
            Ok(b)
        };
        let dev = || TensorCreationOptions::new(self.device.clone());
        match info.tensor_type {
            G::Q8_0 => {
                let scheme = info.tensor_type.to_quant_scheme().unwrap();
                let bytes = raw()?;
                let packed = burn_gguf::repack_to_burn(info.tensor_type, &bytes, scheme, n).map_err(|e| anyhow!("{g}: {e}"))?;
                Ok(Some((Tensor::from_data(TensorData::from_bytes_vec(packed, shape, DType::QFloat(scheme)), dev()), None)))
            }
            G::Q4K if self.keep_kquant() => {
                let bytes = raw()?;
                let (packed, offs) = super::kquant::repack_q4k(&bytes, n);
                let s = super::kquant::q4k_scheme();
                let w = Tensor::from_data(TensorData::from_bytes_vec(packed, shape.clone(), DType::QFloat(s)), dev());
                let c = Tensor::from_data(TensorData::new(offs, [shape[0], shape[1] / 32]), dev().with_dtype(self.dtype));
                Ok(Some((w, Some(c))))
            }
            G::Q5_0 if self.keep_kquant() => {
                let bytes = raw()?;
                let s = super::kquant::q5_0_scheme();
                let packed = super::kquant::repack_q5_0(&bytes, n);
                Ok(Some((Tensor::from_data(TensorData::from_bytes_vec(packed, shape, DType::QFloat(s)), dev()), None)))
            }
            G::Q6K if self.keep_kquant() => {
                let bytes = raw()?;
                let packed = super::kquant::repack_q6k(&bytes, n);
                let s = super::kquant::q6k_scheme();
                Ok(Some((Tensor::from_data(TensorData::from_bytes_vec(packed, shape, DType::QFloat(s)), dev()), None)))
            }
            _ => Ok(None),
        }
    }

    /// K-quants stay packed unless PYRE_GGUF_KQUANT=0.
    fn keep_kquant(&self) -> bool { std::env::var("PYRE_GGUF_KQUANT").map(|v| v != "0").unwrap_or(true) }
}
