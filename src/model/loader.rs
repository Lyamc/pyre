//! Memory-mapped HF safetensors (single file or sharded) -> Burn tensors.
use anyhow::{bail, Context, Result};
use burn::tensor::{DType, Device, Tensor, TensorCreationOptions, TensorData};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

pub struct Weights {
    maps: Vec<Mmap>,
    index: HashMap<String, usize>,
    device: Device,
    dtype: DType,
}

impl Weights {
    pub fn open(dir: &Path, device: Device, dtype: DType) -> Result<Self> {
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
        Ok(Self { maps, index, device, dtype })
    }

    pub fn has(&self, name: &str) -> bool { self.index.contains_key(name) }

    pub fn get<const D: usize>(&self, name: &str) -> Result<Tensor<D>> {
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
        if shape.len() != D { bail!("{name}: expected rank {D}, got {:?}", shape); }
        let data = TensorData::from_bytes_vec(v.data().to_vec(), shape, dt);
        Ok(Tensor::from_data(data, TensorCreationOptions::new(self.device.clone()).with_dtype(self.dtype)))
    }
}
