mod cleanup;
mod config;
mod layout;
mod mapping;

use std::fs;
use std::path::Path;

use anyhow::Context;
use safetensors::SafeTensors;

use crate::convert::dtype::bytes_to_f32;
use crate::convert::npy::NpyDump;
use crate::download::DownloadOptions;

pub use config::ParakeetDecodeConfig;

use cleanup::{cleanup_parakeet_npy_artifacts, write_attn_heads};
use config::ParakeetConfig;
use layout::layout_tensor;
use mapping::{map_hf_tensor, should_skip_key};

const BURN_VERSION: &str = "0.21.0";

pub fn convert_parakeet_from_hf(
    model_dir: &Path,
    model_name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let safetensors_path = model_dir.join("model.safetensors");
    let config_path = model_dir.join("config.json");

    if !safetensors_path.exists() && model_dir.join("shapes.json").exists() {
        if options.verbose {
            tracing::info!("resuming Burn mpk conversion from existing npy dump");
        }
        super::burn_mpk::save_model_from_npy(model_name, options)?;
        cleanup_parakeet_npy_artifacts(model_dir, options.verbose)?;
        fs::write(model_dir.join(".burn_version"), BURN_VERSION)?;
        return Ok(());
    }

    if !safetensors_path.exists() {
        anyhow::bail!(
            "missing model.safetensors in {} — download from HuggingFace first",
            model_dir.display()
        );
    }

    if options.verbose {
        tracing::info!("converting parakeet weights from HF safetensors for '{model_name}'");
    }

    let config: ParakeetConfig =
        serde_json::from_str(&fs::read_to_string(&config_path).context("read config.json")?)?;

    let arch = config
        .architectures
        .first()
        .map(String::as_str)
        .unwrap_or("ParakeetForTDT");
    let is_tdt = arch.contains("TDT") || arch.contains("RNNT");
    let nemotron = arch.contains("Nemotron");

    let bytes = fs::read(&safetensors_path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;

    let mut dump = NpyDump::new(model_dir);
    let enc = &config.encoder_config;

    dump.write_scalar("encoder/n_layers.npy", enc.num_hidden_layers as f32)?;
    dump.write_scalar("encoder/d_model.npy", enc.hidden_size as f32)?;
    dump.write_scalar("encoder/n_head.npy", enc.num_attention_heads as f32)?;
    dump.write_scalar("encoder/n_mels.npy", enc.num_mel_bins as f32)?;
    dump.write_scalar("encoder/kind.npy", if nemotron { 2.0 } else { 1.0 })?;
    if nemotron {
        dump.write_scalar("encoder/conv_kernel.npy", enc.conv_kernel_size.unwrap_or(9) as f32)?;
        dump.write_scalar("encoder/sub_channels.npy", enc.subsampling_conv_channels.unwrap_or(256) as f32)?;
        dump.write_scalar("encoder/sub_kernel.npy", enc.subsampling_conv_kernel_size.unwrap_or(3) as f32)?;
        dump.write_scalar("encoder/ff_inner.npy", enc.intermediate_size.unwrap_or(enc.hidden_size * 4) as f32)?;
        dump.write_scalar("encoder/lookahead.npy", enc.default_num_lookahead_tokens.unwrap_or(13) as f32)?;
        dump.write_scalar("encoder/sliding_window.npy", enc.sliding_window.unwrap_or(71) as f32)?;
        dump.write_scalar("encoder/decoder_hidden.npy", config.decoder_hidden_size.unwrap_or(640) as f32)?;
        dump.write_scalar("encoder/decoder_layers.npy", config.num_decoder_layers.unwrap_or(2) as f32)?;
        dump.write_scalar("encoder/max_symbols.npy", config.max_symbols_per_step.unwrap_or(10) as f32)?;
        dump.write_scalar("encoder/blank_id.npy", config.blank_id() as f32)?;
    }

    let num_durations = config.durations.as_ref().map(|d| d.len()).unwrap_or(0);
    let joint_out = config.vocab_size + num_durations;
    dump.write_scalar("actual_n_vocab.npy", joint_out as f32)?;
    let decode_cfg = ParakeetDecodeConfig {
        blank_token_id: config.blank_id(),
        vocab_size: config.vocab_size,
        duration_start: config.vocab_size,
        num_durations,
        pad_token_id: config.pad_token_id,
        max_symbols_per_step: config.max_symbols_per_step.unwrap_or(10),
        decoder_start_token_id: config.decoder_start_token_id.or(Some(config.blank_id())),
        durations: config.durations.clone().unwrap_or_else(|| (0..num_durations).collect()),
    };
    fs::write(
        model_dir.join("parakeet_decode.json"),
        serde_json::to_string_pretty(&decode_cfg)?,
    )?;

    let mut written = 0usize;
    let mut unmapped = Vec::new();
    for key in tensors.names() {
        let tensor = tensors.tensor(key)?;
        if !matches!(
            tensor.dtype(),
            safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16
        ) {
            continue;
        }
        let shape: Vec<usize> = tensor.shape().iter().copied().collect();
        let floats = bytes_to_f32(tensor.data(), tensor.dtype())?;

        if let Some(rel) = map_hf_tensor(key, is_tdt, nemotron) {
            let (data, shape) = layout_tensor(&rel, &floats, &shape);
            dump.write_f32(&rel, &data, &shape)?;
            written += 1;
        } else if !should_skip_key(key) {
            unmapped.push(key.to_string());
        }
    }

    write_attn_heads(&mut dump, enc.num_hidden_layers, enc.num_attention_heads)?;
    dump.flush_shapes()?;

    if options.verbose {
        tracing::info!("mapped {written} tensors ({} unmapped)", unmapped.len());
        tracing::info!("running burn mpk conversion");
    }

    drop(tensors);
    drop(bytes);
    if safetensors_path.exists() {
        fs::remove_file(&safetensors_path)?;
        if options.verbose {
            tracing::info!("removed {} to reclaim disk", safetensors_path.display());
        }
    }

    super::burn_mpk::save_model_from_npy(model_name, options)?;
    cleanup_parakeet_npy_artifacts(model_dir, options.verbose)?;

    fs::write(model_dir.join(".burn_version"), BURN_VERSION)?;

    if options.verbose {
        tracing::info!("parakeet model '{model_name}' converted for burn {BURN_VERSION}");
    }

    Ok(())
}

pub fn is_parakeet_model(name: &str) -> bool {
    matches!(
        name,
        "parakeet-tdt-0.6b-v3"
            | "parakeet-tdt-0.6b-v2"
            | "parakeet-ctc-0.6b"
            | "parakeet-ctc-1.1b"
    )
}

pub fn is_nemotron_asr_model(name: &str) -> bool {
    name == "nemotron-speech-streaming-en-0.6b"
}

pub fn is_parakeet_burn_ready(name: &str) -> bool {
    matches!(
        name,
        "parakeet-tdt-0.6b-v3" | "parakeet-tdt-0.6b-v2" | "parakeet-ctc-0.6b" | "parakeet-ctc-1.1b"
    )
}