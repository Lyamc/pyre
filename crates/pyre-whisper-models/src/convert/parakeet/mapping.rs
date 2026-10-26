use super::layout::split_layer_tail;

/// HF subsampling ModuleList indices for the five Conv2d layers (8× subsampling).
pub const SUBSAMPLING_CONV_INDICES: [(usize, &str); 5] = [
    (0, "encoder/conv1"),
    (2, "encoder/conv2"),
    (3, "encoder/conv3"),
    (5, "encoder/conv4"),
    (6, "encoder/conv5"),
];

pub fn should_skip_key(key: &str) -> bool {
    let key = key.strip_prefix("model.").unwrap_or(key);
    key.contains("inv_freq") || key.contains("num_batches_tracked")
}

pub fn map_hf_tensor(key: &str, is_tdt: bool) -> Option<String> {
    let key = key.strip_prefix("model.").unwrap_or(key);

    SUBSAMPLING_CONV_INDICES
        .iter()
        .find_map(|&(idx, dest)| {
            let prefix = format!("encoder.subsampling.layers.{idx}.");
            key.strip_prefix(&prefix)
                .map(|rest| format!("{dest}/{rest}.npy"))
        })
        .or_else(|| {
            key.strip_prefix("encoder.subsampling.linear.")
                .map(|rest| format!("encoder/pre_encode_out/{rest}.npy"))
        })
        .or_else(|| {
            key.strip_prefix("encoder.layers.")
                .and_then(map_encoder_layer)
        })
        .or_else(|| map_tdt_or_ctc_head(key, is_tdt))
}

fn map_tdt_or_ctc_head(key: &str, is_tdt: bool) -> Option<String> {
    if is_tdt {
        key.strip_prefix("encoder_projector.")
            .map(|rest| format!("final_proj/{rest}.npy"))
            .or_else(|| {
                key.strip_prefix("decoder.decoder_projector.")
                    .map(|rest| format!("decoder/projector/{rest}.npy"))
            })
            .or_else(|| {
                key.strip_prefix("decoder.embedding.")
                    .map(|rest| format!("decoder/embedding/{rest}.npy"))
            })
            .or_else(|| key.strip_prefix("decoder.lstm.").and_then(map_decoder_lstm))
            .or_else(|| {
                key.strip_prefix("joint.head.")
                    .map(|rest| format!("ctc_linear/{rest}.npy"))
            })
    } else {
        key.strip_prefix("ctc_head.")
            .map(|rest| format!("ctc_linear/{rest}.npy"))
    }
}

fn map_decoder_lstm(rest: &str) -> Option<String> {
    (0..8).find_map(|layer| {
        [
            ("weight_ih", "weight_ih"),
            ("weight_hh", "weight_hh"),
            ("bias_ih", "bias_ih"),
            ("bias_hh", "bias_hh"),
        ]
        .iter()
        .find_map(|&(hf_suffix, dest)| {
            let prefix = format!("{hf_suffix}_l{layer}");
            (rest == prefix).then(|| format!("decoder/lstm/l{layer}/{dest}.npy"))
        })
    })
}

fn map_encoder_layer(rest: &str) -> Option<String> {
    let (idx, tail) = split_layer_tail(rest)?;

    match tail {
        "feed_forward1.linear1.weight" | "feed_forward1.linear1.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("encoder/block_{idx}/ff1/lin1/{part}.npy"))
        }
        "feed_forward1.linear2.weight" | "feed_forward1.linear2.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("encoder/block_{idx}/ff1/lin2/{part}.npy"))
        }
        "norm_feed_forward1.weight" | "norm_feed_forward1.bias" => {
            let part = ln_component(tail);
            Some(format!("encoder/block_{idx}/ff1/ln/{part}.npy"))
        }
        "self_attn.q_proj.weight" | "self_attn.q_proj.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("encoder/block_{idx}/attn/query/{part}.npy"))
        }
        "self_attn.k_proj.weight" | "self_attn.k_proj.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("encoder/block_{idx}/attn/key/{part}.npy"))
        }
        "self_attn.v_proj.weight" | "self_attn.v_proj.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("encoder/block_{idx}/attn/value/{part}.npy"))
        }
        "self_attn.o_proj.weight" | "self_attn.o_proj.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("encoder/block_{idx}/attn/out/{part}.npy"))
        }
        "self_attn.relative_k_proj.weight" => {
            Some(format!("encoder/block_{idx}/attn/pos/weight.npy"))
        }
        "self_attn.bias_u" => Some(format!("encoder/block_{idx}/attn/pos_bias_u.npy")),
        "self_attn.bias_v" => Some(format!("encoder/block_{idx}/attn/pos_bias_v.npy")),
        "norm_self_att.weight" | "norm_self_att.bias" => {
            let part = ln_component(tail);
            Some(format!("encoder/block_{idx}/attn_ln/{part}.npy"))
        }
        "conv.pointwise_conv1.weight" | "conv.pointwise_conv1.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("encoder/block_{idx}/conv/point_conv1/{part}.npy"))
        }
        "conv.depthwise_conv.weight" | "conv.depthwise_conv.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("encoder/block_{idx}/conv/depth_conv/{part}.npy"))
        }
        "conv.pointwise_conv2.weight" | "conv.pointwise_conv2.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("encoder/block_{idx}/conv/point_conv2/{part}.npy"))
        }
        "conv.norm.weight" => Some(format!("encoder/block_{idx}/conv/bn/weight.npy")),
        "conv.norm.bias" => Some(format!("encoder/block_{idx}/conv/bn/bias.npy")),
        "conv.norm.running_mean" => Some(format!("encoder/block_{idx}/conv/bn/running_mean.npy")),
        "conv.norm.running_var" => Some(format!("encoder/block_{idx}/conv/bn/running_var.npy")),
        "norm_conv.weight" | "norm_conv.bias" => {
            let part = ln_component(tail);
            Some(format!("encoder/block_{idx}/conv/ln/{part}.npy"))
        }
        "feed_forward2.linear1.weight" | "feed_forward2.linear1.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("encoder/block_{idx}/ff2/lin1/{part}.npy"))
        }
        "feed_forward2.linear2.weight" | "feed_forward2.linear2.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("encoder/block_{idx}/ff2/lin2/{part}.npy"))
        }
        "norm_feed_forward2.weight" | "norm_feed_forward2.bias" => {
            let part = ln_component(tail);
            Some(format!("encoder/block_{idx}/ff2/ln/{part}.npy"))
        }
        "norm_out.weight" | "norm_out.bias" => {
            let part = ln_component(tail);
            Some(format!("encoder/block_{idx}/final_ln/{part}.npy"))
        }
        _ => None,
    }
}

pub fn ln_component(tail: &str) -> &'static str {
    if tail.ends_with("weight") {
        "gamma"
    } else {
        "beta"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_subsampling_and_joint_keys() {
        assert_eq!(
            map_hf_tensor("encoder.subsampling.layers.0.weight", true),
            Some("encoder/conv1/weight.npy".into())
        );
        assert_eq!(
            map_hf_tensor("encoder_projector.bias", true),
            Some("final_proj/bias.npy".into())
        );
        assert_eq!(
            map_hf_tensor("joint.head.weight", true),
            Some("ctc_linear/weight.npy".into())
        );
    }

    #[test]
    fn maps_decoder_lstm_keys() {
        assert_eq!(
            map_hf_tensor("decoder.lstm.weight_ih_l0", true),
            Some("decoder/lstm/l0/weight_ih.npy".into())
        );
        assert_eq!(
            map_hf_tensor("decoder.embedding.weight", true),
            Some("decoder/embedding/weight.npy".into())
        );
        assert_eq!(
            map_hf_tensor("decoder.decoder_projector.bias", true),
            Some("decoder/projector/bias.npy".into())
        );
    }

    #[test]
    fn maps_nemo_renames_used_by_v2_extract() {
        // After extract_nemo_parakeet.py remaps NeMo keys onto the HF layout.
        assert_eq!(
            map_hf_tensor("encoder.subsampling.layers.0.weight", true),
            Some("encoder/conv1/weight.npy".into())
        );
        assert_eq!(
            map_hf_tensor("encoder.subsampling.linear.weight", true),
            Some("encoder/pre_encode_out/weight.npy".into())
        );
        assert_eq!(
            map_hf_tensor("encoder_projector.weight", true),
            Some("final_proj/weight.npy".into())
        );
        assert_eq!(
            map_hf_tensor("joint.head.weight", true),
            Some("ctc_linear/weight.npy".into())
        );
    }
}