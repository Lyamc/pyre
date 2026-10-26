mod layout;
mod mapping;

use std::fs;
use std::path::Path;

use anyhow::Context;
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::convert::dtype::bytes_to_f32;
use crate::convert::npy::NpyDump;
use crate::download::DownloadOptions;

use layout::{burn_linear_layout, write_attn_heads};
use mapping::map_hf_tensor;

const BURN_VERSION: &str = "0.21.0";
/// Bump when the HF → npy → mpk layout changes so existing bundles reconvert.
const WHISPER_LAYOUT_REV: &str = "linear-T2";

#[derive(Debug, Deserialize)]
struct WhisperConfig {
    num_mel_bins: usize,
    d_model: usize,
    encoder_layers: usize,
    decoder_layers: usize,
    encoder_attention_heads: usize,
    decoder_attention_heads: usize,
}

pub fn convert_whisper_from_hf(
    model_dir: &Path,
    model_name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let safetensors_path = model_dir.join("model.safetensors");
    let config_path = model_dir.join("config.json");

    if !safetensors_path.exists() {
        anyhow::bail!(
            "missing model.safetensors in {} — download from HuggingFace first",
            model_dir.display()
        );
    }

    options.report(crate::download::PrepProgress::convert(
        format!("Converting {model_name} weights"),
        0.05,
    ));
    if options.verbose {
        tracing::info!("converting whisper weights from HF safetensors for '{model_name}'");
    }

    let config: WhisperConfig =
        serde_json::from_str(&fs::read_to_string(&config_path).context("read config.json")?)?;

    let bytes = fs::read(&safetensors_path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;

    let mut dump = NpyDump::new(model_dir);

    dump.write_scalar("encoder/n_mels.npy", config.num_mel_bins as f32)?;
    dump.write_scalar("encoder/n_audio_state.npy", config.d_model as f32)?;
    dump.write_scalar("encoder/n_layer.npy", config.encoder_layers as f32)?;
    dump.write_scalar("decoder/n_layer.npy", config.decoder_layers as f32)?;

    let names: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();
    let total = names.len().max(1);
    let mut written = 0usize;
    for (i, key) in names.iter().enumerate() {
        let tensor = tensors.tensor(key)?;
        let shape: Vec<usize> = tensor.shape().iter().copied().collect();
        let floats = bytes_to_f32(tensor.data(), tensor.dtype())?;

        if let Some(rel) = map_hf_tensor(key) {
            let (data, shape) = burn_linear_layout(&rel, &floats, &shape);
            dump.write_f32(&rel, &data, &shape)?;
            written += 1;
        }
        if i % 10 == 0 || i + 1 == total {
            options.report(crate::download::PrepProgress::convert(
                format!("Mapping tensors {}/{total}", i + 1),
                (i + 1) as f64 / total as f64 * 0.7,
            ));
        }
    }

    write_attn_heads(
        &mut dump,
        "encoder",
        config.encoder_layers,
        config.encoder_attention_heads,
    )?;
    write_attn_heads(
        &mut dump,
        "decoder",
        config.decoder_layers,
        config.decoder_attention_heads,
    )?;

    dump.flush_shapes()?;

    options.report(crate::download::PrepProgress::convert(
        format!("Mapped {written} tensors, writing Burn bundle"),
        0.75,
    ));
    if options.verbose {
        tracing::info!("mapped {written} tensors, running burn mpk conversion");
    }

    super::burn_mpk::save_model_from_npy(model_name, options)?;

    fs::write(model_dir.join(".burn_version"), BURN_VERSION)?;
    fs::write(model_dir.join(".whisper_layout"), WHISPER_LAYOUT_REV)?;

    if options.verbose {
        tracing::info!("whisper model '{model_name}' converted for burn {BURN_VERSION}");
    }

    Ok(())
}

pub fn needs_reconversion(model_dir: &Path) -> bool {
    let burn_ok = fs::read_to_string(model_dir.join(".burn_version"))
        .map(|v| v.trim() == BURN_VERSION)
        .unwrap_or(false);
    let layout_ok = fs::read_to_string(model_dir.join(".whisper_layout"))
        .map(|v| v.trim() == WHISPER_LAYOUT_REV)
        .unwrap_or(false);
    !burn_ok || !layout_ok
}

pub fn is_whisper_model(name: &str) -> bool {
    matches!(
        name,
        "tiny"
            | "tiny_en"
            | "base"
            | "base_en"
            | "small"
            | "small_en"
            | "medium"
            | "medium_en"
            | "large-v1"
            | "large-v2"
            | "large-v3"
            | "large-v3-turbo"
            | "distil-medium-en"
            | "distil-large-v3"
    )
}