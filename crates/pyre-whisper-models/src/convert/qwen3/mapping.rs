/// Map HF safetensor keys (`thinker.*`) to Burn module paths.
pub fn map_hf_tensor(key: &str) -> Option<String> {
    let key = key.strip_prefix("thinker.").unwrap_or(key);

    key.strip_prefix("audio_tower.")
        .map(|rest| format!("audio_tower/{rest}.npy"))
        .or_else(|| {
            key.strip_prefix("model.")
                .map(|rest| format!("thinker/{rest}.npy"))
        })
        .or_else(|| {
            key.strip_prefix("lm_head.")
                .map(|rest| format!("lm_head/{rest}.npy"))
        })
}

pub fn should_skip_key(key: &str) -> bool {
    let key = key.strip_prefix("thinker.").unwrap_or(key);
    key.contains("positional_embedding") || key.contains("inv_freq")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_audio_tower_and_thinker() {
        assert_eq!(
            map_hf_tensor("thinker.audio_tower.conv2d1.weight"),
            Some("audio_tower/conv2d1.weight.npy".into())
        );
        assert_eq!(
            map_hf_tensor("thinker.model.layers.0.self_attn.q_proj.weight"),
            Some("thinker/layers.0.self_attn.q_proj.weight.npy".into())
        );
        assert_eq!(
            map_hf_tensor("thinker.lm_head.weight"),
            Some("lm_head/weight.npy".into())
        );
    }

    #[test]
    fn skips_positional_embedding() {
        assert!(should_skip_key(
            "thinker.audio_tower.positional_embedding.positional_embedding"
        ));
    }
}