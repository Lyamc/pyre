use burn::{
    module::Module,
    tensor::{Tensor},
    config::Config,
};
use crate::model::*;
use crate::model::conformer::*;
use crate::model::load::helpers::*;
use crate::model::load::whisper::*;
use crate::model::load::parakeet::*;
use crate::model::load::nemotron::load_nemotron;
use crate::model::load::qwen3::load_qwen3_model;
use crate::model::load::vibevoice::load_vibevoice_model;
use crate::token::Gpt2Tokenizer;
use std::error::Error;
use std::path::Path;
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

pub fn load_model_from_npy(
    path: &str,
    device: &burn::tensor::Device,
) -> Result<(Model, ModelConfig), Box<dyn Error>> {
    if Path::new(&format!("{}/tone.encoder", path)).exists() || path.contains("t-one") {
        return Err(
            "T-one uses safetensors, not npy. Load via `load_model(\"t-one\", ...)` after download."
                .into(),
        );
    } else if Path::new(&format!("{path}/encoder/kind.npy")).exists()
        && load_usize("kind", &format!("{path}/encoder"), device).unwrap_or(1) == 2
    {
        let (nemotron, config) = load_nemotron(path, device)?;
        Ok((Model::Nemotron(nemotron), ModelConfig::Nemotron(config)))
    } else if Path::new(&format!("{}/ctc_linear", path)).exists() {
        let parakeet = load_parakeet(path, device)?;
        
        // Detect actual params from the loaded model
        let n_layers = parakeet.encoder.layers.len();
        let d_model = parakeet.encoder.layers[0].final_ln.gamma.dims()[0];
        let n_head = parakeet.encoder.layers[0].attn.n_head;
        let kernel_size = parakeet.encoder.layers[0].conv.depth_conv.weight.dims()[2];
        
        let n_mels = if Path::new(&format!("{}/encoder/n_mels.npy", path)).exists() {
            load_usize("n_mels", &format!("{}/encoder", path), device)?
        } else {
            128 // Default for Parakeet-TDT v3
        };

        // Burn Linear weight is [d_in, d_out]; joint head output is d_out.
        let n_vocab = parakeet.ctc_linear.weight.dims()[1];

        let layer_bias = tensor_exists("bias", &format!("{path}/encoder/block_0/ff1/lin1"));
        let config = ModelConfig::Parakeet(ParakeetConfig {
            encoder: ConformerEncoderConfig {
                d_model,
                n_head,
                n_layers,
                kernel_size,
                layer_bias,
            },
            n_vocab,
            n_mels,
        });
        
        Ok((Model::Parakeet(parakeet), config))
    } else if Path::new(&format!("{}/generic_marker.npy", path)).exists() || path.contains("moonshine") {
        // Placeholder for new architectures like Moonshine, Voxtral, etc.
        if path.contains("moonshine") {
             return Err("Moonshine architecture detected but full Burn implementation is pending.".into());
        }
        return Err("Model loaded successfully but architecture is not yet implemented in Burn.".into());
    } else {
        let (whisper, config) = load_whisper(path, device)?;
        Ok((Model::Whisper(whisper), ModelConfig::Whisper(config)))
    }
}

pub fn load_model(
    model_name: &str,
    device: &burn::tensor::Device,
    verbose: bool,
) -> Result<(Gpt2Tokenizer, ModelConfig, Model), Box<dyn Error>> {
    tracing::info!(model = model_name, "initializing model load");
    if verbose {
        println!("Initializing model loading for: {}", model_name);
    }

    if model_name == "vibevoice-asr" || model_name == "bitnet-asr" {
        return load_vibevoice_model(model_name, device, verbose);
    }

    if model_name.starts_with("moonshine") {
        return crate::model::load::moonshine::load_moonshine_model(model_name, device, verbose);
    }

    if model_name == "t-one" || model_name.starts_with("t-one") {
        return crate::model::load::tone::load_tone_model(model_name, device, verbose);
    }

    if model_name.starts_with("qwen3-asr-") {
        return load_qwen3_model(model_name, device, verbose);
    }

    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner()
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        .template("{spinner:.green} {msg}").unwrap());
    pb.set_message(format!("Loading model '{}'...", model_name));
    pb.enable_steady_tick(Duration::from_millis(100));

    let bpe = match Gpt2Tokenizer::new(model_name) {
        Ok(bpe) => bpe,
        Err(e) => {
            pb.finish_and_clear();
            return Err(format!("Failed to load tokenizer for {}: {}", model_name, e).into());
        }
    };

    let dir = pyre_whisper_core::resolve_model_dir(model_name);
    let config_path1 = dir.join(format!("{}.cfg", model_name));
    let config_path2 = dir.join("config.cfg");
    let config_path = if config_path1.exists() {
        config_path1
    } else if config_path2.exists() {
        config_path2
    } else {
        // Search for ANY .cfg in the directory
        let mut found = None;
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if entry.path().extension().is_some_and(|ext| ext == "cfg") {
                    found = Some(entry.path());
                    break;
                }
            }
        }
        found.unwrap_or_default()
    };

    if config_path.as_os_str().is_empty() {
        pb.finish_and_clear();
        return Err(format!("Config file not found in {}", dir.display()).into());
    }

    if verbose {
        pb.println(format!("Using config file: {}", config_path.display()));
    }

    let model_config = match ModelConfig::load(&config_path) {
        Ok(config) => config,
        Err(_) => {
            match WhisperConfig::load(&config_path) {
                Ok(w) => ModelConfig::Whisper(w),
                Err(e) => {
                    pb.finish_and_clear();
                    return Err(format!("Failed to load model config at {}: {}", config_path.display(), e).into());
                }
            }
        }
    };

    let named_stem = dir.join(model_name);
    let mpk_path = named_stem.with_extension("mpk");
    let gz_path = dir.join(format!("{model_name}.bpk"));

    let model_mpk_path = dir.join("model.bpk");
    let model_gz_path = dir.join("model.bpk");

    let (final_path, final_gz_path) = if mpk_path.exists() || gz_path.exists() {
        (mpk_path, gz_path)
    } else {
        (model_mpk_path, model_gz_path)
    };

    if !final_path.exists() && final_gz_path.exists() {
        pb.set_message(format!("Decompressing {}...", final_gz_path.display()));
        let gz_file = std::fs::File::open(&final_gz_path)?;
        let mut decoder = GzDecoder::new(gz_file);
        let mut mpk_file = std::fs::File::create(&final_path)?;
        std::io::copy(&mut decoder, &mut mpk_file)?;
    }

    if !final_path.exists() {
        pb.finish_and_clear();
        return Err(format!("Model weights not found: {}", final_path.display()).into());
    }

    tracing::info!(
        model = model_name,
        path = %final_path.display(),
        "loading model weights onto device"
    );
    pb.set_message(format!("Loading model weights from {}...", final_path.display()));
    // Burnpack (`model.bpk`), written by pyre's converter with the workspace Burn. Burn 0.21
    // `.mpk` recorder bundles are not readable by this Burn; they are re-converted from HF.
    let model: Model = match model_config.init(device).try_load_file(&final_path) {
        Ok(m) => m,
        Err(e) => {
            pb.finish_and_clear();
            return Err(format!("Failed to load model weights {}: {e:?}", final_path.display()).into());
        }
    };
    pb.set_message("Moving model to device...");
    let model = model.to_device(device);

    pb.finish_with_message(format!("Model '{}' loaded successfully.", model_name));
    tracing::info!(model = model_name, "model loaded onto device");

    Ok((bpe, model_config, model))
}

pub fn get_model_stats(model_name: &str, device: &burn::tensor::Device) -> (Option<Tensor<2>>, Option<Tensor<2>>) {
    let dir = pyre_whisper_core::resolve_model_dir(model_name);
    let path = dir.to_string_lossy().into_owned();
    let mean = if dir.join("mean.npy").exists() {
        load_tensor::<2>("mean", &path, device).ok()
    } else {
        None
    };
    let std = if dir.join("std.npy").exists() {
        load_tensor::<2>("std", &path, device).ok()
    } else {
        None
    };
    (mean, std)
}
