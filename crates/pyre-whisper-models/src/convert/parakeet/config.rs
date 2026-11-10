use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ParakeetEncoderConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_mel_bins: usize,
    /// Present on HF encoder configs (CTC sets true; TDT sets false). Burn applies the
    /// matching scale from architecture, not this flag, but we keep it for round-trip JSON.
    #[serde(default)]
    #[allow(dead_code)]
    pub scale_input: bool,
    #[serde(default)]
    pub conv_kernel_size: Option<usize>,
    #[serde(default)]
    pub subsampling_conv_channels: Option<usize>,
    #[serde(default)]
    pub subsampling_conv_kernel_size: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub default_num_lookahead_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ParakeetConfig {
    pub vocab_size: usize,
    #[serde(default)]
    pub blank_token_id: Option<usize>,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: usize,
    pub durations: Option<Vec<usize>>,
    pub architectures: Vec<String>,
    pub encoder_config: ParakeetEncoderConfig,
    #[serde(default)]
    pub decoder_hidden_size: Option<usize>,
    #[serde(default)]
    pub num_decoder_layers: Option<usize>,
    #[serde(default)]
    pub max_symbols_per_step: Option<usize>,
    #[serde(default)]
    pub decoder_start_token_id: Option<usize>,
}

impl ParakeetConfig {
    pub fn blank_id(&self) -> usize {
        self.blank_token_id.unwrap_or(self.pad_token_id)
    }
}

fn default_pad_token_id() -> usize {
    2
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ParakeetDecodeConfig {
    pub blank_token_id: usize,
    pub vocab_size: usize,
    pub duration_start: usize,
    pub num_durations: usize,
    pub pad_token_id: usize,
    #[serde(default = "default_max_symbols")]
    pub max_symbols_per_step: usize,
    #[serde(default)]
    pub decoder_start_token_id: Option<usize>,
    #[serde(default)]
    pub durations: Vec<usize>,
}

fn default_max_symbols() -> usize {
    10
}