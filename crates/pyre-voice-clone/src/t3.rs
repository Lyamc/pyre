//! T3-Nano layout. This does not run the model and does not emit audio.
//!
//! Nano is built in `ChatterboxTurboTTS.from_local(nano=True)` as a GPT-2 small
//! backbone with separate text and speech embeddings. `input_pos_emb` is `None`,
//! the perceiver is off, and emotion conditioning is off. `inference_turbo` does
//! not wrap the text with `start_text_token` / `stop_text_token` (those defaults,
//! 255 and 0, belong to the older 704-id English tokenizer and are unused here).
//!
//! The checkpoint still stores `tfmr.wte`. The loader deletes it and uses
//! `text_emb` instead. `tfmr.wpe` stays: GPT-2 adds it to `inputs_embeds`.
//! Attention and MLP weights are Hugging Face Conv1D layout, `[in, out]`, not
//! `[out, in]`. `c_attn` is the fused QKV projection, `[768, 2304]`.
//!
//! `gelu_new` is the GPT-2 tanh approximation. A forward port has to use that,
//! not the exact erf GELU.
//!
//! Speech ids at or above 6561 are stripped before the vocoder (`speech_tokens < 6561`).
//! The embedding table is 6563 wide so the start id 6561 and stop id 6562 fit.

use anyhow::{anyhow, bail, Context, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub const N_LAYER: usize = 12;
pub const N_HEAD: usize = 12;
pub const N_EMBD: usize = 768;
pub const N_POSITIONS: usize = 8196;
pub const TEXT_TOKENS: usize = 50_276;
pub const SPEECH_TOKENS: usize = 6563;
pub const START_SPEECH: u32 = 6561;
pub const STOP_SPEECH: u32 = 6562;
pub const SPEAKER_EMBED: usize = 256;
pub const SPEECH_COND_PROMPT_LEN: usize = 375;
pub const LAYER_NORM_EPS: f64 = 1e-5;

/// Decode settings `inference_turbo` uses when the caller does not override them.
pub struct TurboDecode {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub max_gen_len: usize,
}

impl Default for TurboDecode {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 1000,
            top_p: 0.95,
            repetition_penalty: 1.2,
            max_gen_len: 1000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: &'static str,
    pub shape: &'static [usize],
    /// Present in the file. `wte` is unused after load.
    pub used_at_inference: bool,
}

pub fn nano_tensors() -> Vec<TensorSpec> {
    vec![
        TensorSpec {
            name: "cond_enc.spkr_enc.weight",
            shape: &[N_EMBD, SPEAKER_EMBED],
            used_at_inference: true,
        },
        TensorSpec {
            name: "cond_enc.spkr_enc.bias",
            shape: &[N_EMBD],
            used_at_inference: true,
        },
        TensorSpec {
            name: "text_emb.weight",
            shape: &[TEXT_TOKENS, N_EMBD],
            used_at_inference: true,
        },
        TensorSpec {
            name: "text_head.weight",
            shape: &[TEXT_TOKENS, N_EMBD],
            used_at_inference: true,
        },
        TensorSpec {
            name: "speech_emb.weight",
            shape: &[SPEECH_TOKENS, N_EMBD],
            used_at_inference: true,
        },
        TensorSpec {
            name: "speech_head.weight",
            shape: &[SPEECH_TOKENS, N_EMBD],
            used_at_inference: true,
        },
        TensorSpec {
            name: "speech_head.bias",
            shape: &[SPEECH_TOKENS],
            used_at_inference: true,
        },
        TensorSpec {
            name: "tfmr.wpe.weight",
            shape: &[N_POSITIONS, N_EMBD],
            used_at_inference: true,
        },
        TensorSpec {
            name: "tfmr.wte.weight",
            shape: &[TEXT_TOKENS, N_EMBD],
            used_at_inference: false,
        },
        TensorSpec {
            name: "tfmr.ln_f.weight",
            shape: &[N_EMBD],
            used_at_inference: true,
        },
        TensorSpec {
            name: "tfmr.ln_f.bias",
            shape: &[N_EMBD],
            used_at_inference: true,
        },
    ]
}

pub fn layer_suffixes() -> &'static [&'static str] {
    &[
        "attn.c_attn.weight",
        "attn.c_attn.bias",
        "attn.c_proj.weight",
        "attn.c_proj.bias",
        "ln_1.weight",
        "ln_1.bias",
        "ln_2.weight",
        "ln_2.bias",
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    ]
}

pub fn layer_shape(suffix: &str) -> &'static [usize] {
    match suffix {
        "attn.c_attn.weight" => &[N_EMBD, 3 * N_EMBD],
        "attn.c_attn.bias" => &[3 * N_EMBD],
        "attn.c_proj.weight" => &[N_EMBD, N_EMBD],
        "attn.c_proj.bias" => &[N_EMBD],
        "mlp.c_fc.weight" => &[N_EMBD, 4 * N_EMBD],
        "mlp.c_fc.bias" => &[4 * N_EMBD],
        "mlp.c_proj.weight" => &[4 * N_EMBD, N_EMBD],
        "mlp.c_proj.bias" => &[N_EMBD],
        "ln_1.weight" | "ln_1.bias" | "ln_2.weight" | "ln_2.bias" => &[N_EMBD],
        _ => &[],
    }
}

/// Read only the safetensors JSON header. Does not map the tensor bytes.
pub fn read_header(path: &Path) -> Result<BTreeMap<String, Vec<usize>>> {
    let mut f = File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)
        .context("safetensors header length")?;
    let n = u64::from_le_bytes(len_buf) as usize;
    if n == 0 || n > 32 * 1024 * 1024 {
        bail!("safetensors header length {n} is not a T3 checkpoint");
    }
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf).context("safetensors header")?;
    let v: serde_json::Value = serde_json::from_slice(&buf).context("safetensors header json")?;
    let obj = v.as_object().context("header is not an object")?;
    let mut out = BTreeMap::new();
    for (k, val) in obj {
        if k == "__metadata__" {
            continue;
        }
        let shape = val
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| anyhow!("{k}: missing shape"))?;
        let dims = shape
            .iter()
            .map(|d| {
                d.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| anyhow!("{k}: bad dim"))
            })
            .collect::<Result<Vec<_>>>()?;
        let dtype = val.get("dtype").and_then(|d| d.as_str()).unwrap_or("");
        if dtype != "F32" {
            bail!("{k}: dtype {dtype}, Nano T3 ships f32");
        }
        out.insert(k.clone(), dims);
    }
    Ok(out)
}

/// Compare a checkpoint header to the Nano contract. Does not run the network.
pub fn check_checkpoint(path: &Path) -> Result<usize> {
    let header = read_header(path)?;
    for spec in nano_tensors() {
        match header.get(spec.name) {
            Some(shape) if shape.as_slice() == spec.shape => {}
            Some(shape) => bail!("{}: shape {shape:?} != {:?}", spec.name, spec.shape),
            None => bail!("missing {}", spec.name),
        }
    }
    for layer in 0..N_LAYER {
        for suffix in layer_suffixes() {
            let name = format!("tfmr.h.{layer}.{suffix}");
            let expect = layer_shape(suffix);
            match header.get(&name) {
                Some(shape) if shape.as_slice() == expect => {}
                Some(shape) => bail!("{name}: shape {shape:?} != {expect:?}"),
                None => bail!("missing {name}"),
            }
        }
    }
    let expect_n = nano_tensors().len() + N_LAYER * layer_suffixes().len();
    if header.len() != expect_n {
        let extra: Vec<_> = header
            .keys()
            .filter(|k| !expected_name(k))
            .cloned()
            .collect();
        bail!(
            "header has {} tensors, contract is {expect_n}, extra {extra:?}",
            header.len()
        );
    }
    Ok(header.len())
}

fn expected_name(name: &str) -> bool {
    if nano_tensors().iter().any(|s| s.name == name) {
        return true;
    }
    let mut parts = name.split('.');
    if parts.next() == Some("tfmr") && parts.next() == Some("h") {
        let layer: usize = parts
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(usize::MAX);
        if layer < N_LAYER {
            let suffix = name.splitn(4, '.').nth(3).unwrap_or("");
            return layer_suffixes().contains(&suffix);
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_covers_twelve_gpt2_blocks_and_drops_wte() {
        assert_eq!(
            nano_tensors()
                .iter()
                .filter(|s| s.name == "tfmr.wte.weight")
                .count(),
            1
        );
        assert!(
            !nano_tensors()
                .iter()
                .find(|s| s.name == "tfmr.wte.weight")
                .unwrap()
                .used_at_inference
        );
        assert!(
            nano_tensors()
                .iter()
                .find(|s| s.name == "tfmr.wpe.weight")
                .unwrap()
                .used_at_inference
        );
        assert_eq!(layer_shape("attn.c_attn.weight"), &[768, 2304]);
        assert_eq!(layer_shape("mlp.c_fc.weight"), &[768, 3072]);
        assert_eq!(
            N_LAYER * layer_suffixes().len() + nano_tensors().len(),
            12 * 12 + 11
        );
        assert_eq!(TurboDecode::default().top_k, 1000);
        assert_eq!(START_SPEECH, 6561);
    }

    #[test]
    fn header_reader_rejects_a_truncated_length() {
        let tmp = std::env::temp_dir().join(format!("pyre-t3-hdr-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("bad.safetensors");
        std::fs::write(&path, [8u8, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        assert!(read_header(&path).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    #[ignore = "set PYRE_CHATTERBOX_T3 to t3_nano_v1.safetensors (header only is enough) and pass --ignored"]
    fn real_t3_header_matches_contract() {
        let Some(path) = std::env::var_os("PYRE_CHATTERBOX_T3") else {
            eprintln!("PYRE_CHATTERBOX_T3 unset");
            return;
        };
        let n = check_checkpoint(Path::new(&path)).unwrap();
        assert_eq!(n, 155);
    }
}
