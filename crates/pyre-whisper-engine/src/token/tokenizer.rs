use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde::ser::StdError;
use std::collections::HashMap;
use std::result;
use tokenizers::models::bpe::{BpeBuilder, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::AddedToken;
use crate::token::special::SpecialToken;

#[derive(Debug, Deserialize)]
struct Qwen3AddedTokenEntry {
    content: String,
    special: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct Qwen3TokenizerConfig {
    added_tokens_decoder: Option<HashMap<String, Qwen3AddedTokenEntry>>,
}

fn read_qwen3_added_tokens(config_path: &std::path::Path) -> Option<Vec<AddedToken>> {
    let config_raw = fs::read_to_string(config_path).ok()?;
    let config = serde_json::from_str::<Qwen3TokenizerConfig>(&config_raw).ok()?;
    let added = config.added_tokens_decoder?;
    let mut tokens: Vec<AddedToken> = added
        .values()
        .map(|entry| AddedToken::from(entry.content.as_str(), entry.special.unwrap_or(true)))
        .collect();
    tokens.sort_by_key(|t| t.content.clone());
    Some(tokens)
}

fn load_qwen3_tokenizer(model_name: &str) -> Result<tokenizers::Tokenizer> {
    let model_dir = pyre_whisper_core::resolve_model_dir(model_name);
    let vocab_path = model_dir.join("vocab.json");
    let merges_path = model_dir.join("merges.txt");
    let config_path = model_dir.join("tokenizer_config.json");

    let vocab_s = vocab_path.to_string_lossy();
    let merges_s = merges_path.to_string_lossy();
    let (mut vocab, merges) = BPE::read_file(vocab_s.as_ref(), merges_s.as_ref())
        .map_err(|e| format!("failed to read Qwen3 vocab/merges: {e}"))?;

    if let Ok(config_raw) = fs::read_to_string(&config_path) {
        if let Ok(config) = serde_json::from_str::<Qwen3TokenizerConfig>(&config_raw) {
            if let Some(added) = config.added_tokens_decoder {
                for (id_str, entry) in added {
                    if let Ok(id) = id_str.parse::<u32>() {
                        vocab.insert(entry.content, id);
                    }
                }
            }
        }
    }

    let bpe = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| format!("failed to build Qwen3 BPE: {e}"))?;

    let mut tokenizer = tokenizers::Tokenizer::new(bpe);
    // Qwen2Tokenizer is GPT-2 byte-level BPE. Without this decoder, `decode()` emits raw
    // Ġ-prefixed pieces instead of spaces (JFK looked like "And Ġso , Ġmy Ġfellow...").
    let pretok = ByteLevel::default()
        .add_prefix_space(false)
        .trim_offsets(false);
    tokenizer.with_pre_tokenizer(Some(pretok));
    let decoder = ByteLevel::default()
        .add_prefix_space(true)
        .trim_offsets(false);
    tokenizer.with_decoder(Some(decoder));
    if let Some(specials) = read_qwen3_added_tokens(&config_path) {
        tokenizer
            .add_special_tokens(specials)
            .map_err(|e| format!("failed to add Qwen3 special tokens: {e}"))?;
    }
    Ok(tokenizer)
}

pub type Result<T> = result::Result<T, Box<dyn StdError + Send + Sync + 'static>>;

pub struct Gpt2Tokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Gpt2Tokenizer {
    pub fn new(model_name: &str) -> Result<Self> {
        let dir = pyre_whisper_core::resolve_model_dir(model_name);
        let json_path = dir.join("tokenizer.json");
        let vocab_path = dir.join("vocab.json");
        let merges_path = dir.join("merges.txt");

        let is_qwen_family = model_name.contains("qwen3")
            || model_name.contains("vibevoice")
            || model_name.contains("bitnet")
            || model_name.contains("moonshine")
            || model_name.contains("t-one");

        let mut tokenizer = if Path::new(&json_path).exists() {
            tokenizers::Tokenizer::from_file(&json_path)?
        } else if is_qwen_family
            && Path::new(&vocab_path).exists()
            && Path::new(&merges_path).exists()
        {
            load_qwen3_tokenizer(model_name)?
        } else if Path::new(&vocab_path).exists() && Path::new(&merges_path).exists() {
            let vocab_s = vocab_path.to_string_lossy();
            let merges_s = merges_path.to_string_lossy();
            let bpe = tokenizers::models::bpe::BPE::from_file(vocab_s.as_ref(), merges_s.as_ref())
                .build()
                .map_err(|e| format!("failed to load BPE from vocab/merges: {e}"))?;
            tokenizers::Tokenizer::new(bpe)
        } else {
            tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default())
        };

        if !is_qwen_family {
            tokenizer
                .add_special_tokens(crate::token::special::construct_special_tokens())
                .map_err(|e| format!("failed to add special tokens: {e}"))?;
        }

        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.encode_with_special_tokens(text, true)
    }

    pub fn encode_with_special_tokens(&self, text: &str, add_special_tokens: bool) -> Vec<usize> {
        let tokens = self
            .tokenizer
            .encode(text, add_special_tokens)
            .unwrap();
        tokens.get_ids().iter().map(|t| *t as usize).collect()
    }

    pub fn special_token_by_name(&self, name: &str) -> Option<usize> {
        self.tokenizer.token_to_id(name).map(|id| id as usize)
    }

    pub fn special_token(&self, token: SpecialToken) -> Option<usize> {
        self.tokenizer
            .token_to_id(&token.to_string())
            .map(|id| id as usize)
    }


    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.tokenizer
            .token_to_id(token)
            .map(|id| id as usize)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    pub fn decode(&self, tokens: &[usize], skip_special: bool) -> Result<String> {
        let text = self.tokenizer.decode(
            &tokens.iter().map(|t| *t as u32).collect::<Vec<u32>>(),
            skip_special,
        )?;
        Ok(normalize_bpe_decode(&text))
    }

    pub fn is_special(&self, token: usize) -> bool {
        self.tokenizer
            .decode(vec![token as u32].as_slice(), true)
            .ok()
            .map(|s| s.is_empty())
            .unwrap_or(false)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn is_timestamp(&self, token: usize) -> bool {
        if let Some(s) = self.tokenizer.id_to_token(token as u32) {
            // Pattern: <|dd.dd|>
            if s.starts_with("<|") && s.ends_with("|>") && s.len() >= 7 {
                let inner = &s[2..s.len()-2];
                return inner.parse::<f64>().is_ok();
            }
        }
        false
    }

    pub fn token_to_timestamp(&self, token: usize) -> Option<f64> {
        if let Some(s) = self.tokenizer.id_to_token(token as u32) {
            if s.starts_with("<|") && s.ends_with("|>") && s.len() >= 7 {
                let inner = &s[2..s.len()-2];
                return inner.parse::<f64>().ok();
            }
        }
        None
    }
}

/// GPT-2 / Qwen byte-level BPE uses U+0120 (Ġ) for spaces. Collapse leftovers if a
/// tokenizer was loaded without a ByteLevel decoder.
fn normalize_bpe_decode(text: &str) -> String {
    let replaced = text.replace('\u{0120}', " ").replace('\u{2581}', " ");
    let collapsed = replaced.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed
}

#[cfg(test)]
mod tests {
    use super::normalize_bpe_decode;

    #[test]
    fn strips_gpt2_space_markers() {
        let raw = "And \u{0120}so , \u{0120}my \u{0120}fellow \u{0120}Americans";
        assert_eq!(
            normalize_bpe_decode(raw),
            "And so , my fellow Americans"
        );
    }
}
