/// Map HF safetensor keys to Burn module paths (for future weight loading).
pub fn map_hf_tensor(key: &str) -> Option<String> {
    let key = key.strip_prefix("model.").unwrap_or(key);

    key.strip_prefix("multi_modal_projector.")
        .and_then(map_projector)
        .or_else(|| {
            key.strip_prefix("language_model.")
                .map(|rest| format!("language_model/{rest}.npy"))
        })
        .or_else(|| {
            key.strip_prefix("acoustic_tokenizer_encoder.")
                .map(|rest| format!("acoustic_encoder/{rest}.npy"))
        })
        .or_else(|| {
            key.strip_prefix("semantic_tokenizer_encoder.")
                .map(|rest| format!("semantic_encoder/{rest}.npy"))
        })
        // Legacy VibeVoice-ASR (non-HF) key layout
        .or_else(|| {
            key.strip_prefix("acoustic_connector.")
                .map(|rest| format!("acoustic_connector/{rest}.npy"))
        })
        .or_else(|| {
            key.strip_prefix("semantic_connector.")
                .map(|rest| format!("semantic_connector/{rest}.npy"))
        })
        .or_else(|| {
            key.strip_prefix("acoustic_tokenizer.encoder.")
                .map(|rest| format!("acoustic_encoder/{rest}.npy"))
        })
        .or_else(|| {
            key.strip_prefix("semantic_tokenizer.encoder.")
                .map(|rest| format!("semantic_encoder/{rest}.npy"))
        })
        .or_else(|| {
            (key == "lm_head.weight").then(|| "language_model/lm_head.weight.npy".into())
        })
}

fn map_projector(rest: &str) -> Option<String> {
    let (connector, field) = rest
        .strip_prefix("acoustic_")
        .map(|r| ("acoustic_connector", r))
        .or_else(|| rest.strip_prefix("semantic_").map(|r| ("semantic_connector", r)))?;

    let burn_field = match field {
        "linear_1.weight" => "fc1.weight",
        "linear_1.bias" => "fc1.bias",
        "norm.weight" => "norm_gamma",
        "linear_2.weight" => "fc2.weight",
        "linear_2.bias" => "fc2.bias",
        _ => return None,
    };
    Some(format!("{connector}/{burn_field}.npy"))
}

pub fn should_skip_key(key: &str) -> bool {
    let key = key.strip_prefix("model.").unwrap_or(key);
    key.contains("acoustic_tokenizer_decoder")
        || key.contains("semantic_tokenizer_decoder")
        || key.contains("acoustic_tokenizer.decoder")
        || key.contains("semantic_tokenizer.decoder")
        || key.contains("diffusion_head")
        || key.contains("inv_freq")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_hf_asr_projector_keys() {
        assert_eq!(
            map_hf_tensor("multi_modal_projector.acoustic_linear_1.weight"),
            Some("acoustic_connector/fc1.weight.npy".into())
        );
        assert_eq!(
            map_hf_tensor("multi_modal_projector.acoustic_norm.weight"),
            Some("acoustic_connector/norm_gamma.npy".into())
        );
        assert_eq!(
            map_hf_tensor("multi_modal_projector.semantic_linear_2.bias"),
            Some("semantic_connector/fc2.bias.npy".into())
        );
    }

    #[test]
    fn maps_encoders_and_language_model() {
        assert_eq!(
            map_hf_tensor("acoustic_tokenizer_encoder.head.conv.weight"),
            Some("acoustic_encoder/head.conv.weight.npy".into())
        );
        assert_eq!(
            map_hf_tensor("language_model.lm_head.weight"),
            Some("language_model/lm_head.weight.npy".into())
        );
        assert_eq!(
            map_hf_tensor("language_model.model.embed_tokens.weight"),
            Some("language_model/model.embed_tokens.weight.npy".into())
        );
        assert_eq!(
            map_hf_tensor("model.language_model.embed_tokens.weight"),
            Some("language_model/embed_tokens.weight.npy".into())
        );
        assert_eq!(
            map_hf_tensor("lm_head.weight"),
            Some("language_model/lm_head.weight.npy".into())
        );
        assert_eq!(
            map_hf_tensor("model.acoustic_tokenizer.encoder.head.conv.conv.weight"),
            Some("acoustic_encoder/head.conv.conv.weight.npy".into())
        );
    }

    #[test]
    fn skips_decoder_tensors() {
        assert!(should_skip_key(
            "acoustic_tokenizer_decoder.stages.0.block.0.ffn.linear1.weight"
        ));
    }
}