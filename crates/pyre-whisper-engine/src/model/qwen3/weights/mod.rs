mod load;

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::Path;

use safetensors::SafeTensors;

use crate::model::vibevoice::weights::dtype::bytes_to_f32;

pub use load::{load_qwen3_lm_weights, load_qwen3_weights};

/// Map `Qwen3-ASR-*-hf` (`model.*`) keys onto the original `thinker.*` layout.
pub fn canonicalize_qwen3_key(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("model.multi_modal_projector.linear_1") {
        return format!("thinker.audio_tower.proj1{rest}");
    }
    if let Some(rest) = key.strip_prefix("model.multi_modal_projector.linear_2") {
        return format!("thinker.audio_tower.proj2{rest}");
    }
    if let Some(rest) = key.strip_prefix("model.language_model.") {
        return format!("thinker.model.{rest}");
    }
    if let Some(rest) = key.strip_prefix("model.audio_tower.") {
        return format!("thinker.audio_tower.{rest}");
    }
    // Text-only Qwen3-0.6B: `model.layers.*` / `model.embed_tokens.*` / `model.norm.*`
    if let Some(rest) = key.strip_prefix("model.layers.") {
        return format!("thinker.model.layers.{rest}");
    }
    if let Some(rest) = key.strip_prefix("model.embed_tokens.") {
        return format!("thinker.model.embed_tokens.{rest}");
    }
    if let Some(rest) = key.strip_prefix("model.norm.") {
        return format!("thinker.model.norm.{rest}");
    }
    if let Some(rest) = key.strip_prefix("lm_head.") {
        return format!("thinker.lm_head.{rest}");
    }
    key.to_string()
}

pub struct Qwen3WeightStore {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl Qwen3WeightStore {
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        let path = model_dir.join("model.safetensors");
        let bytes = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let st = SafeTensors::deserialize(&bytes)
            .map_err(|e| format!("parse {}: {e}", path.display()))?;
        let mut tensors = HashMap::new();
        for key in st.names() {
            let tensor = st.tensor(key).map_err(|e| format!("{key}: {e}"))?;
            let shape: Vec<usize> = tensor.shape().iter().copied().collect();
            let floats = bytes_to_f32(tensor.data(), tensor.dtype())?;
            tensors.insert(canonicalize_qwen3_key(key), (floats, shape));
        }
        if !tensors.contains_key("thinker.lm_head.weight") {
            if let Some(embed) = tensors.get("thinker.model.embed_tokens.weight").cloned() {
                tensors.insert("thinker.lm_head.weight".into(), embed);
            }
        }
        Ok(Self { tensors })
    }

    pub fn has_key(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    pub fn tensor_f32(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        self.tensors
            .get(key)
            .cloned()
            .ok_or_else(|| format!("tensor key not found: {key}"))
    }
}

fn safetensors_header_keys(path: &Path) -> Result<Vec<String>, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|e| format!("read header length {}: {e}", path.display()))?;
    let n = u64::from_le_bytes(len_buf) as usize;
    if n == 0 || n > 50_000_000 {
        return Err(format!("implausible safetensors header size {n}"));
    }
    let mut header = vec![0u8; n];
    file.read_exact(&mut header)
        .map_err(|e| format!("read header {}: {e}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_slice(&header).map_err(|e| format!("parse header {}: {e}", path.display()))?;
    let obj = value
        .as_object()
        .ok_or_else(|| "safetensors header is not an object".to_string())?;
    Ok(obj
        .keys()
        .filter(|k| k.as_str() != "__metadata__")
        .cloned()
        .collect())
}

pub fn weights_present(model_dir: &Path) -> bool {
    let path = model_dir.join("model.safetensors");
    safetensors_header_keys(&path)
        .map(|keys| {
            keys.iter().any(|k| {
                let canon = canonicalize_qwen3_key(k);
                canon == "thinker.audio_tower.conv2d1.weight"
                    || canon == "thinker.model.layers.0.self_attn.q_proj.weight"
            })
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::canonicalize_qwen3_key;

    #[test]
    fn remaps_hf_layout_to_thinker_keys() {
        assert_eq!(
            canonicalize_qwen3_key("model.audio_tower.conv2d1.weight"),
            "thinker.audio_tower.conv2d1.weight"
        );
        assert_eq!(
            canonicalize_qwen3_key("model.language_model.layers.0.self_attn.q_proj.weight"),
            "thinker.model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            canonicalize_qwen3_key("model.multi_modal_projector.linear_1.weight"),
            "thinker.audio_tower.proj1.weight"
        );
        assert_eq!(
            canonicalize_qwen3_key("thinker.audio_tower.conv2d1.weight"),
            "thinker.audio_tower.conv2d1.weight"
        );
        assert_eq!(
            canonicalize_qwen3_key("model.layers.0.self_attn.q_proj.weight"),
            "thinker.model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            canonicalize_qwen3_key("lm_head.weight"),
            "thinker.lm_head.weight"
        );
    }
}