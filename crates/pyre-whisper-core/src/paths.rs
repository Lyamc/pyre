use std::path::{Path, PathBuf};

/// Directory where new model downloads are written.
///
/// Override with `WHISBURN_MODELS_DIR`. Defaults to `./models` (cwd).
pub fn models_dir() -> PathBuf {
    std::env::var("WHISBURN_MODELS_DIR")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"))
}

/// Preferred on-disk location for a named model (used when downloading).
pub fn model_dir(name: &str) -> PathBuf {
    models_dir().join(name)
}

/// Directory that actually contains `name`, searching fallbacks used in development
/// and next to the binary. Falls back to [`model_dir`] if nothing is present yet.
pub fn resolve_model_dir(name: &str) -> PathBuf {
    let primary = model_dir(name);
    if model_dir_has_assets(&primary) {
        return primary;
    }
    for candidate in extra_model_dirs(name) {
        if model_dir_has_assets(&candidate) {
            return candidate;
        }
    }
    primary
}

pub fn resolve_model_file(name: &str, file: &str) -> PathBuf {
    resolve_model_dir(name).join(file)
}

fn extra_model_dirs(name: &str) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.join("models").join(name));
        }
    }
    dirs
}

fn model_dir_has_assets(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let has_tokenizer = dir.join("tokenizer.json").is_file()
        || (dir.join("vocab.json").is_file() && dir.join("merges.txt").is_file());
    let has_weights = dir.join("model.bpk").is_file()
        || dir.join("model.bpk").is_file()
        || dir.join("model.safetensors").is_file()
        || dir.join("model.safetensors.index.json").is_file();
    let has_config = dir.join("config.cfg").is_file()
        || dir.join("config.json").is_file()
        || dir.join("qwen3_runtime.json").is_file()
        || dir.join("parakeet_decode.json").is_file()
        || dir.join("vibevoice_runtime.json").is_file()
        || dir.join("moonshine_runtime.json").is_file()
        || dir.join("tone_runtime.json").is_file();
    has_tokenizer && (has_weights || has_config)
}
