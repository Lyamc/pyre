use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(alias = "type")]
    pub rope_type: Option<String>,
    pub factor: Option<f64>,
    pub low_freq_factor: Option<f64>,
    pub high_freq_factor: Option<f64>,
    pub original_max_position_embeddings: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub vocab_size: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_theta")]
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub attention_bias: Option<bool>,
    pub torch_dtype: Option<String>,
    pub bos_token_id: Option<serde_json::Value>,
    pub eos_token_id: Option<serde_json::Value>,
}
fn default_eps() -> f64 { 1e-6 }
fn default_theta() -> f64 { 10000.0 }
fn default_max_pos() -> usize { 4096 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch { Llama, Qwen2, Qwen3 }

impl LlamaConfig {
    pub fn arch(&self) -> anyhow::Result<Arch> {
        let a = self.architectures.first().map(String::as_str).unwrap_or("");
        Ok(match (self.model_type.as_str(), a) {
            ("qwen2", _) | (_, "Qwen2ForCausalLM") => Arch::Qwen2,
            ("qwen3", _) | (_, "Qwen3ForCausalLM") => Arch::Qwen3,
            ("llama", _) | ("mistral", _) | (_, "LlamaForCausalLM") | (_, "MistralForCausalLM") => Arch::Llama,
            (m, a) => anyhow::bail!("unsupported architecture model_type={m} arch={a} (supported: llama, mistral, qwen2, qwen3 dense)"),
        })
    }
    pub fn n_kv_heads(&self) -> usize { self.num_key_value_heads.unwrap_or(self.num_attention_heads) }
    pub fn head_dim(&self) -> usize { self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads) }
    pub fn qkv_bias(&self) -> bool {
        self.attention_bias.unwrap_or(false) || matches!(self.arch(), Ok(Arch::Qwen2))
    }
    pub fn qk_norm(&self) -> bool { matches!(self.arch(), Ok(Arch::Qwen3)) }
    pub fn eos_ids(&self) -> Vec<u32> { ids(&self.eos_token_id) }
}

pub fn ids(v: &Option<serde_json::Value>) -> Vec<u32> {
    match v {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Some(serde_json::Value::Array(a)) => a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect(),
        _ => vec![],
    }
}

/// Inverse RoPE frequencies, with Llama 3 / linear scaling applied.
pub fn inv_freq(cfg: &LlamaConfig) -> Vec<f64> {
    let hd = cfg.head_dim();
    let mut f: Vec<f64> = (0..hd / 2).map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f64 / hd as f64)).collect();
    if let Some(s) = &cfg.rope_scaling {
        let ty = s.rope_type.as_deref().unwrap_or("default");
        let factor = s.factor.unwrap_or(1.0);
        match ty {
            "llama3" => {
                let lo = s.low_freq_factor.unwrap_or(1.0);
                let hi = s.high_freq_factor.unwrap_or(4.0);
                let old = s.original_max_position_embeddings.unwrap_or(8192) as f64;
                let lo_wl = old / lo;
                let hi_wl = old / hi;
                for x in f.iter_mut() {
                    let wl = 2.0 * std::f64::consts::PI / *x;
                    if wl < hi_wl {
                    } else if wl > lo_wl {
                        *x /= factor;
                    } else {
                        let smooth = (old / wl - lo) / (hi - lo);
                        *x = (1.0 - smooth) * *x / factor + smooth * *x;
                    }
                }
            }
            "linear" => f.iter_mut().for_each(|x| *x /= factor),
            "default" => {}
            other => tracing::warn!("rope_scaling type {other} not implemented; using unscaled RoPE"),
        }
    }
    f
}
