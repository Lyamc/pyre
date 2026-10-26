use std::collections::HashMap;
use std::fs;
use std::path::Path;

use safetensors::SafeTensors;

use super::dtype::bytes_to_f32;
use super::gguf::{gguf_shorten, GgufFile};
use super::index::VibeVoiceWeightIndex;

enum StoreInner {
    Safetensors {
        index: VibeVoiceWeightIndex,
        shard_cache: HashMap<String, Vec<u8>>,
    },
    Gguf(GgufFile),
}

pub struct VibeVoiceWeightStore {
    inner: StoreInner,
}

impl VibeVoiceWeightStore {
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        if let Some(gguf) = find_gguf(model_dir) {
            return Ok(Self {
                inner: StoreInner::Gguf(GgufFile::open(&gguf)?),
            });
        }
        Ok(Self {
            inner: StoreInner::Safetensors {
                index: VibeVoiceWeightIndex::open(model_dir)?,
                shard_cache: HashMap::new(),
            },
        })
    }

    pub fn has_key(&self, key: &str) -> bool {
        self.resolve(key).is_some()
    }

    pub fn resolve(&self, key: &str) -> Option<String> {
        for cand in key_aliases(key) {
            match &self.inner {
                StoreInner::Safetensors { index, .. } => {
                    if index.has_key(&cand) {
                        return Some(cand);
                    }
                }
                StoreInner::Gguf(gguf) => {
                    let short = gguf_shorten(&cand);
                    if gguf.has_name(&short) {
                        return Some(short);
                    }
                    if gguf.has_name(&cand) {
                        return Some(cand);
                    }
                }
            }
        }
        None
    }

    pub fn tensor_f32(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let resolved = self
            .resolve(key)
            .ok_or_else(|| format!("tensor key not in index: {key}"))?;
        match &mut self.inner {
            StoreInner::Gguf(gguf) => gguf.tensor_f32(&resolved),
            StoreInner::Safetensors { index, shard_cache } => {
                let shard_path = index.shard_path(&resolved)?;
                let shard_name = shard_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .ok_or_else(|| "invalid shard path".to_string())?
                    .to_string();

                if !shard_cache.contains_key(&shard_name) {
                    shard_cache.clear();
                    let bytes = fs::read(&shard_path)
                        .map_err(|e| format!("read shard {}: {e}", shard_path.display()))?;
                    shard_cache.insert(shard_name.clone(), bytes);
                }

                let bytes = shard_cache.get(&shard_name).unwrap();
                let tensors = SafeTensors::deserialize(bytes)
                    .map_err(|e| format!("deserialize {}: {e}", shard_path.display()))?;
                let tensor = tensors
                    .tensor(&resolved)
                    .map_err(|e| format!("tensor {resolved} in {}: {e}", shard_path.display()))?;

                let shape: Vec<usize> = tensor.shape().iter().copied().collect();
                let floats = bytes_to_f32(tensor.data(), tensor.dtype())?;
                Ok((floats, shape))
            }
        }
    }
}

fn find_gguf(dir: &Path) -> Option<std::path::PathBuf> {
    let preferred = dir.join("vibevoice-asr-q4_k.gguf");
    if preferred.is_file() {
        return Some(preferred);
    }
    fs::read_dir(dir).ok()?.flatten().find_map(|e| {
        let p = e.path();
        (p.extension().and_then(|s| s.to_str()) == Some("gguf")).then_some(p)
    })
}

/// HF Transformers (`VibeVoice-ASR-HF`) vs original (`VibeVoice-ASR` / BitNet) key spellings.
fn key_aliases(key: &str) -> Vec<String> {
    let mut keys = Vec::new();
    let push = |keys: &mut Vec<String>, k: String| {
        if !keys.iter().any(|e| e == &k) {
            keys.push(k);
        }
    };

    push(&mut keys, key.to_string());
    let stripped = key.strip_prefix("model.").unwrap_or(key);
    push(&mut keys, stripped.to_string());
    if !key.starts_with("model.") {
        push(&mut keys, format!("model.{key}"));
    }

    let lm_rest = stripped
        .strip_prefix("language_model.model.")
        .or_else(|| stripped.strip_prefix("language_model."));
    if let Some(rest) = lm_rest {
        push(&mut keys, format!("language_model.model.{rest}"));
        push(&mut keys, format!("language_model.{rest}"));
        push(&mut keys, format!("model.language_model.{rest}"));
        push(&mut keys, format!("model.language_model.model.{rest}"));
    }

    if stripped == "language_model.lm_head.weight" || stripped.ends_with("lm_head.weight") {
        push(&mut keys, "lm_head.weight".into());
        push(&mut keys, "model.lm_head.weight".into());
        push(&mut keys, "language_model.lm_head.weight".into());
    }

    keys
}

#[cfg(test)]
mod tests {
    use super::key_aliases;

    #[test]
    fn aliases_bitnet_lm_keys() {
        let aliases = key_aliases("language_model.model.embed_tokens.weight");
        assert!(aliases.iter().any(|k| k == "model.language_model.embed_tokens.weight"));
        let head = key_aliases("language_model.lm_head.weight");
        assert!(head.iter().any(|k| k == "lm_head.weight"));
    }

    #[test]
    fn gguf_shorten_matches_crispasr() {
        use super::super::gguf::gguf_shorten;
        assert_eq!(
            gguf_shorten("model.language_model.layers.0.self_attn.q_proj.weight"),
            "lm.layers.0.attn.q_proj.weight"
        );
        assert_eq!(
            gguf_shorten("language_model.embed_tokens.weight"),
            "lm.tok_emb.weight"
        );
        assert_eq!(
            gguf_shorten("model.acoustic_connector.fc1.weight"),
            "at_conn.fc1.weight"
        );
        assert_eq!(
            gguf_shorten("acoustic_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight"),
            "at_enc.ds.0.0.conv.weight"
        );
    }
}