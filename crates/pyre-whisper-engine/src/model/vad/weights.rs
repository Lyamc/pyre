use std::collections::HashMap;

use safetensors::SafeTensors;

use crate::model::vibevoice::weights::dtype::bytes_to_f32;

pub type TensorMap = HashMap<String, (Vec<f32>, Vec<usize>)>;

pub fn load_f32(bytes: &[u8]) -> Result<TensorMap, String> {
    let tensors =
        SafeTensors::deserialize(bytes).map_err(|e| format!("safetensors: {e}"))?;
    let mut map = HashMap::new();
    for name in tensors.names() {
        let t = tensors
            .tensor(name)
            .map_err(|e| format!("tensor {name}: {e}"))?;
        let shape: Vec<usize> = t.shape().to_vec();
        let data = bytes_to_f32(t.data(), t.dtype())?;
        map.insert(name.to_string(), (data, shape));
    }
    Ok(map)
}

pub fn load_named(map: &TensorMap, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    if let Some(v) = map.get(name) {
        return Ok(v.clone());
    }
    let short = name.rsplit('.').next().unwrap_or(name);
    for (k, v) in map {
        if k == name || k.ends_with(name) || k.ends_with(short) {
            return Ok(v.clone());
        }
    }
    Err(format!(
        "tensor {name} missing (have {})",
        map.keys().cloned().collect::<Vec<_>>().join(", ")
    ))
}

pub fn flatten_last(data: Vec<f32>, shape: &[usize]) -> Vec<f32> {
    let _ = shape;
    data
}
