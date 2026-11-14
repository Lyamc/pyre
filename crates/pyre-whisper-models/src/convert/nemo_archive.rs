//! Read a NeMo `.nemo` tar (Parakeet TDT v2) and write the Hugging Face tensor layout
//! `convert_parakeet_from_hf` already consumes. No Python, and no zero-filled biases:
//! the Burn loader treats a missing bias file as "this linear has no bias".

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::Context;
use flate2::read::GzDecoder;
use pytorch_reader::{DType, PytorchReader};
use safetensors::tensor::{serialize, Dtype, TensorView};

/// Rename one NeMo state-dict key onto the HF Parakeet layout. Same substitutions as the
/// old `scripts/extract_nemo_parakeet.py`.
pub fn rename_nemo_key(key: &str) -> String {
    let mut k = key.to_string();
    if let Some(rest) = k.strip_prefix("encoder.pre_encode.conv.") {
        if let Some((idx, tail)) = rest.split_once('.') {
            if !idx.is_empty() && idx.chars().all(|c| c.is_ascii_digit()) {
                k = format!("encoder.subsampling.layers.{idx}.{tail}");
            }
        }
    }
    if let Some(rest) = k.strip_prefix("encoder.pre_encode.out.") {
        k = format!("encoder.subsampling.linear.{rest}");
    }
    const MID: &[(&str, &str)] = &[
        (".self_attn.linear_q.", ".self_attn.q_proj."),
        (".self_attn.linear_k.", ".self_attn.k_proj."),
        (".self_attn.linear_v.", ".self_attn.v_proj."),
        (".self_attn.linear_out.", ".self_attn.o_proj."),
        (".self_attn.linear_pos.", ".self_attn.relative_k_proj."),
        (".conv.batch_norm.", ".conv.norm."),
    ];
    for (from, to) in MID {
        k = k.replace(from, to);
    }
    if let Some(prefix) = k.strip_suffix(".self_attn.pos_bias_u") {
        k = format!("{prefix}.self_attn.bias_u");
    }
    if let Some(prefix) = k.strip_suffix(".self_attn.pos_bias_v") {
        k = format!("{prefix}.self_attn.bias_v");
    }
    if let Some(rest) = k.strip_prefix("decoder.prediction.embed.") {
        k = format!("decoder.embedding.{rest}");
    }
    if let Some(rest) = k.strip_prefix("decoder.prediction.dec_rnn.lstm.") {
        k = format!("decoder.lstm.{rest}");
    }
    if let Some(rest) = k.strip_prefix("joint.enc.") {
        k = format!("encoder_projector.{rest}");
    }
    if let Some(rest) = k.strip_prefix("joint.pred.") {
        k = format!("decoder.decoder_projector.{rest}");
    }
    if let Some(rest) = k.strip_prefix("joint.joint_net.2.") {
        k = format!("joint.head.{rest}");
    }
    k
}

pub fn yaml_scalar<'a>(yaml: &'a str, key: &str) -> Option<&'a str> {
    yaml.lines().find_map(|line| {
        let line = line.trim();
        let rest = line.strip_prefix(key)?;
        let rest = rest.trim_start();
        let rest = rest.strip_prefix(':')?.trim();
        let rest = rest.split('#').next().unwrap_or(rest).trim();
        let rest = rest.trim_matches(|c| c == '"' || c == '\'');
        if rest.is_empty() || rest == "|" || rest == ">" {
            None
        } else {
            Some(rest)
        }
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct FluidSpeechSpec {
    pub insert_dim: usize,
    pub insert_kernel: usize,
    pub rank: usize,
    pub alpha: f64,
}

impl FluidSpeechSpec {
    pub fn scale(&self) -> f64 {
        if self.rank == 0 {
            1.0
        } else {
            self.alpha / self.rank as f64
        }
    }
}

/// The `fluid_speech:` block in a NeMo `model_config.yaml`. Other `rank` keys in the file are ignored.
pub fn fluid_speech_spec(yaml: &str) -> Option<FluidSpeechSpec> {
    let mut in_block = false;
    let mut in_clean = false;
    let mut spec = FluidSpeechSpec { insert_dim: 1024, insert_kernel: 15, rank: 32, alpha: 64.0 };
    let mut found = false;
    for line in yaml.lines() {
        if !in_block {
            if line.trim() == "fluid_speech:" || line.trim().starts_with("fluid_speech:") {
                in_block = true;
                found = true;
            }
            continue;
        }
        if !line.starts_with(' ') && !line.starts_with('\t') && !line.trim().is_empty() {
            break;
        }
        let trim = line.trim();
        if trim.starts_with("clean_mode:") {
            in_clean = true;
            continue;
        }
        if let Some(v) = trim.strip_prefix("insert_dim:") {
            spec.insert_dim = v.trim().parse().unwrap_or(spec.insert_dim);
        } else if let Some(v) = trim.strip_prefix("insert_kernel:") {
            spec.insert_kernel = v.trim().parse().unwrap_or(spec.insert_kernel);
        } else if in_clean {
            if let Some(v) = trim.strip_prefix("rank:") {
                spec.rank = v.trim().parse().unwrap_or(spec.rank);
            } else if let Some(v) = trim.strip_prefix("alpha:") {
                spec.alpha = v.trim().parse().unwrap_or(spec.alpha);
            }
        }
    }
    found.then_some(spec)
}

pub fn yaml_durations(yaml: &str) -> Vec<usize> {
    if let Some(rest) = yaml_scalar(yaml, "durations") {
        if rest.starts_with('[') {
            return rest
                .trim_matches(['[', ']'])
                .split(|c: char| !c.is_ascii_digit())
                .filter(|s| !s.is_empty())
                .filter_map(|s| s.parse().ok())
                .collect();
        }
    }
    let mut out = Vec::new();
    let mut in_block = false;
    for line in yaml.lines() {
        let trim = line.trim();
        if trim.starts_with("durations:") {
            in_block = true;
            continue;
        }
        if in_block {
            if let Some(n) = trim.strip_prefix("- ") {
                if let Ok(v) = n.trim().parse() {
                    out.push(v);
                    continue;
                }
            }
            if !trim.is_empty() && !line.starts_with(' ') && !line.starts_with('\t') {
                break;
            }
        }
    }
    if out.is_empty() {
        vec![0, 1, 2, 3, 4]
    } else {
        out
    }
}

/// Extract `model_weights.ckpt` + `model_config.yaml` + `tokenizer.vocab` into
/// `model.safetensors`, `config.json`, and `tokenizer.json`.
pub fn extract_nemo_to_hf(nemo: &Path, out_dir: &Path) -> anyhow::Result<()> {
    let mut archive = open_archive(nemo)?;
    let mut yaml = None;
    let mut ckpt: Option<Vec<u8>> = None;
    let mut vocab = None;
    for entry in archive.entries().context("reading .nemo tar")? {
        let mut entry = entry.context("tar entry")?;
        let path = entry.path().context("tar path")?.into_owned();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name == "model_config.yaml" {
            let mut s = String::new();
            entry.read_to_string(&mut s)?;
            yaml = Some(s);
        } else if name == "model_weights.ckpt" {
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf)?;
            ckpt = Some(buf);
        } else if name.ends_with("tokenizer.vocab") {
            let mut s = String::new();
            entry.read_to_string(&mut s)?;
            vocab = Some(s);
        }
    }
    let yaml = yaml.context("model_config.yaml missing from .nemo")?;
    let ckpt = ckpt.context("model_weights.ckpt missing from .nemo")?;
    let vocab_txt = vocab.context("tokenizer.vocab missing from .nemo")?;

    let ckpt_path = out_dir.join(".nemo_weights.ckpt");
    fs::create_dir_all(out_dir)?;
    fs::write(&ckpt_path, &ckpt)?;
    let tensors = read_state_dict(&ckpt_path)?;
    let _ = fs::remove_file(&ckpt_path);

    let mut names = Vec::new();
    let mut views_owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (key, tensor) in tensors {
        if key.starts_with("preprocessor.") || key.ends_with("num_batches_tracked") {
            continue;
        }
        let renamed = rename_nemo_key(&key);
        let (shape, bytes) = tensor_f32(&tensor).with_context(|| format!("tensor {key}"))?;
        names.push(renamed);
        views_owned.push((String::new(), shape, bytes));
    }
    // names and views_owned are parallel; attach names onto the owned records.
    for (rec, name) in views_owned.iter_mut().zip(names.iter()) {
        rec.0 = name.clone();
    }

    let head = views_owned
        .iter()
        .find(|(n, _, _)| n == "joint.head.weight")
        .context("joint.head.weight missing after rename")?;
    let vocab_size = yaml_scalar(&yaml, "vocab_size")
        .context("vocab_size missing")?
        .parse::<usize>()
        .context("vocab_size")?
        + 1;
    let durations = yaml_durations(&yaml);
    let head_out = head.1.first().copied().unwrap_or(0);
    if head_out != vocab_size + durations.len() {
        anyhow::bail!(
            "joint head rows {head_out} != vocab {vocab_size} + {} durations",
            durations.len()
        );
    }

    let views: Vec<(&str, TensorView<'_>)> = views_owned
        .iter()
        .map(|(name, shape, bytes)| {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                .map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
            Ok((name.as_str(), view))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let blob = serialize(views, None).map_err(|e| anyhow::anyhow!("safetensors: {e}"))?;
    fs::write(out_dir.join("model.safetensors"), blob)?;

    let d_model = yaml_scalar(&yaml, "d_model").context("d_model")?.parse::<usize>()?;
    let n_heads = yaml_scalar(&yaml, "n_heads")
        .or_else(|| yaml_scalar(&yaml, "num_attention_heads"))
        .context("n_heads")?
        .parse::<usize>()?;
    let n_layers = yaml_scalar(&yaml, "n_layers").context("n_layers")?.parse::<usize>()?;
    let features = yaml_scalar(&yaml, "features")
        .or_else(|| yaml_scalar(&yaml, "n_mels"))
        .context("features")?
        .parse::<usize>()?;
    let scale = yaml_scalar(&yaml, "xscaling").unwrap_or("false") == "true";
    let cfg = serde_json::json!({
        "architectures": ["ParakeetForTDT"],
        "model_type": "parakeet_tdt",
        "vocab_size": vocab_size,
        "blank_token_id": vocab_size - 1,
        "pad_token_id": vocab_size - 1,
        "decoder_start_token_id": vocab_size - 1,
        "durations": durations,
        "encoder_config": {
            "hidden_size": d_model,
            "num_attention_heads": n_heads,
            "num_hidden_layers": n_layers,
            "num_mel_bins": features,
            "scale_input": scale,
        }
    });
    if !out_dir.join("config.json").exists() {
        fs::write(out_dir.join("config.json"), serde_json::to_vec_pretty(&cfg)?)?;
    }
    if let Some(spec) = fluid_speech_spec(&yaml) {
        let meta = serde_json::json!({
            "insert_dim": spec.insert_dim,
            "insert_kernel": spec.insert_kernel,
            "rank": spec.rank,
            "alpha": spec.alpha,
            "scale": spec.scale(),
        });
        fs::write(out_dir.join("fluid_speech.json"), serde_json::to_vec_pretty(&meta)?)?;
    }

    let pieces: Vec<&str> = vocab_txt
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.split('\t').next().unwrap_or(l))
        .collect();
    let mut map = serde_json::Map::new();
    for (i, p) in pieces.iter().enumerate() {
        map.insert((*p).to_string(), serde_json::json!(i));
    }
    let unk = pieces.first().copied().unwrap_or("<unk>");
    let tok = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": { "type": "WordLevel", "vocab": map, "unk_token": unk }
    });
    if !out_dir.join("tokenizer.json").exists() {
        let mut tok_file = File::create(out_dir.join("tokenizer.json"))?;
        serde_json::to_writer(&mut tok_file, &tok)?;
        tok_file.write_all(b"\n")?;
    }
    fs::write(out_dir.join("vocab.txt"), pieces.join("\n") + "\n")?;
    tracing::info!("extracted {} tensors from {}", views_owned.len(), nemo.display());
    Ok(())
}

fn read_state_dict(path: &Path) -> anyhow::Result<std::collections::HashMap<String, pytorch_reader::Tensor>> {
    if let Ok(reader) = PytorchReader::with_top_level_key(path, "state_dict") {
        let map = reader.into_tensors();
        if !map.is_empty() {
            return Ok(map);
        }
    }
    let reader = PytorchReader::new(path).context("reading NeMo checkpoint")?;
    Ok(reader.into_tensors())
}

fn tensor_f32(tensor: &pytorch_reader::Tensor) -> anyhow::Result<(Vec<usize>, Vec<u8>)> {
    let shape = tensor.shape().to_vec();
    let bytes = tensor.read().context("read tensor")?;
    let floats: Vec<f32> = match tensor.dtype() {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => anyhow::bail!("unsupported dtype {other:?}"),
    };
    let mut out = Vec::with_capacity(floats.len() * 4);
    for v in floats {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok((shape, out))
}

fn open_archive(path: &Path) -> anyhow::Result<tar::Archive<Box<dyn Read>>> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut magic = [0u8; 2];
    let n = file.read(&mut magic)?;
    file.seek(SeekFrom::Start(0))?;
    if n == 2 && magic == [0x1f, 0x8b] {
        Ok(tar::Archive::new(Box::new(GzDecoder::new(file))))
    } else {
        Ok(tar::Archive::new(Box::new(file)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renames_match_the_old_python_extractor() {
        assert_eq!(
            rename_nemo_key("encoder.pre_encode.conv.0.weight"),
            "encoder.subsampling.layers.0.weight"
        );
        assert_eq!(
            rename_nemo_key("encoder.pre_encode.out.weight"),
            "encoder.subsampling.linear.weight"
        );
        assert_eq!(
            rename_nemo_key("encoder.layers.3.self_attn.linear_q.weight"),
            "encoder.layers.3.self_attn.q_proj.weight"
        );
        assert_eq!(
            rename_nemo_key("encoder.layers.1.self_attn.pos_bias_u"),
            "encoder.layers.1.self_attn.bias_u"
        );
        assert_eq!(
            rename_nemo_key("encoder.layers.0.conv.batch_norm.weight"),
            "encoder.layers.0.conv.norm.weight"
        );
        assert_eq!(
            rename_nemo_key("decoder.prediction.embed.weight"),
            "decoder.embedding.weight"
        );
        assert_eq!(
            rename_nemo_key("decoder.prediction.dec_rnn.lstm.weight_ih_l0"),
            "decoder.lstm.weight_ih_l0"
        );
        assert_eq!(rename_nemo_key("joint.enc.weight"), "encoder_projector.weight");
        assert_eq!(
            rename_nemo_key("joint.pred.bias"),
            "decoder.decoder_projector.bias"
        );
        assert_eq!(rename_nemo_key("joint.joint_net.2.weight"), "joint.head.weight");
    }

    #[test]
    fn yaml_scalars_and_duration_list() {
        let yaml = "d_model: 1024\nvocab_size: 1024\ndurations:\n  - 0\n  - 1\n  - 2\n  - 3\n  - 4\nxscaling: false\n";
        assert_eq!(yaml_scalar(yaml, "d_model"), Some("1024"));
        assert_eq!(yaml_durations(yaml), vec![0, 1, 2, 3, 4]);
        assert_eq!(yaml_durations("durations: [0, 1, 2]\n"), vec![0, 1, 2]);
        let yaml = "model:\n  rank: 8\nfluid_speech:\n  insert_dim: 1024\n  insert_kernel: 15\n  clean_mode:\n    rank: 32\n    alpha: 64.0\nencoder:\n  d_model: 1024\n";
        let spec = fluid_speech_spec(yaml).unwrap();
        assert_eq!(spec.rank, 32);
        assert_eq!(spec.scale(), 2.0);
        assert!(fluid_speech_spec("d_model: 1024\n").is_none());
    }
}
