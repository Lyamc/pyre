use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

pub struct VibeVoiceWeightIndex {
    model_dir: PathBuf,
    weight_map: HashMap<String, String>,
}

impl VibeVoiceWeightIndex {
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let content = fs::read_to_string(&index_path)
            .map_err(|e| format!("read {}: {e}", index_path.display()))?;
        let index: SafetensorsIndex = serde_json::from_str(&content)
            .map_err(|e| format!("parse {}: {e}", index_path.display()))?;

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            weight_map: index.weight_map,
        })
    }

    pub fn shard_path(&self, key: &str) -> Result<PathBuf, String> {
        let shard = self
            .weight_map
            .get(key)
            .ok_or_else(|| format!("tensor key not in index: {key}"))?;
        Ok(self.model_dir.join(shard))
    }

    pub fn has_key(&self, key: &str) -> bool {
        self.weight_map.contains_key(key)
    }
}