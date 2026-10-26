use std::path::Path;

pub use pyre_whisper_core::{model_dir, models_dir, resolve_model_dir};

pub fn is_model_ready(name: &str) -> bool {
    has_burn_bundle(&resolve_model_dir(name), name)
}

pub fn needs_reconversion(name: &str, dir: &Path) -> bool {
    crate::convert::needs_reconversion_for(name, dir)
}

fn has_vibevoice_gguf(dir: &Path) -> bool {
    let has_gguf = dir.join("vibevoice-asr-q4_k.gguf").is_file()
        || std::fs::read_dir(dir)
            .ok()
            .map(|it| {
                it.flatten()
                    .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("gguf"))
            })
            .unwrap_or(false);
    has_gguf
        && dir.join("tokenizer.json").exists()
        && dir.join("config.json").exists()
        && dir.join("vibevoice_runtime.json").exists()
}

fn has_vibevoice_bundle(dir: &Path) -> bool {
    if has_vibevoice_gguf(dir) {
        return true;
    }
    let index_path = dir.join("model.safetensors.index.json");
    if !index_path.exists() || !dir.join("tokenizer.json").exists() || !dir.join("config.json").exists() {
        return false;
    }
    if !dir.join("vibevoice_runtime.json").exists() {
        return false;
    }
    let Ok(raw) = std::fs::read_to_string(&index_path) else {
        return false;
    };
    let Ok(index) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    let Some(map) = index.get("weight_map").and_then(|v| v.as_object()) else {
        return false;
    };
    let shards: std::collections::HashSet<&str> = map
        .values()
        .filter_map(|v| v.as_str())
        .collect();
    !shards.is_empty() && shards.iter().all(|shard| dir.join(shard).exists())
}

fn has_moonshine_bundle(dir: &Path) -> bool {
    dir.join("model.safetensors").exists()
        && dir.join("config.json").exists()
        && dir.join("tokenizer.json").exists()
        && dir.join("moonshine_runtime.json").exists()
}

fn has_tone_bundle(dir: &Path) -> bool {
    dir.join("model.safetensors").exists()
        && dir.join("config.json").exists()
        && dir.join("tone_runtime.json").exists()
}

fn has_qwen3_bundle(dir: &Path) -> bool {
    let has_tokenizer = dir.join("tokenizer.json").exists()
        || (dir.join("vocab.json").exists() && dir.join("merges.txt").exists());
    dir.join("model.safetensors").exists()
        && dir.join("config.json").exists()
        && dir.join("qwen3_runtime.json").exists()
        && has_tokenizer
}

pub fn has_burn_bundle(dir: &Path, name: &str) -> bool {
    if crate::convert::is_qwen3_model(name) {
        return has_qwen3_bundle(dir);
    }
    if crate::convert::is_moonshine_model(name) {
        return has_moonshine_bundle(dir);
    }
    if crate::convert::is_tone_model(name) {
        return has_tone_bundle(dir);
    }
    if crate::convert::is_vibevoice_model(name) {
        return has_vibevoice_bundle(dir);
    }
    if crate::convert::is_vad_model(name) {
        let ok = dir.join("model.safetensors").exists() && dir.join("vad_runtime.json").exists();
        if name == "ten-vad" {
            return ok && dir.join("ten_features.json").exists();
        }
        return ok;
    }
    if crate::convert::is_diarize_model(name) {
        return (dir.join("embedding.safetensors").exists() || dir.join("model.safetensors").exists())
            && dir.join("diarize_runtime.json").exists();
    }

    let mpk_candidates = [
        dir.join("model.bpk"),
        dir.join(format!("{name}.bpk")),
        dir.join("model.bpk"),
        dir.join(format!("{name}.bpk")),
    ];
    let cfg_candidates = [
        dir.join("config.cfg"),
        dir.join(format!("{name}.cfg")),
    ];
    let tokenizer = dir.join("tokenizer.json");

    mpk_candidates.iter().any(|p| p.exists())
        && cfg_candidates.iter().any(|p| p.exists())
        && tokenizer.exists()
}