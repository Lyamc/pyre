pub fn split_layer_tail(rest: &str) -> Option<(usize, &str)> {
    let mut parts = rest.splitn(2, '.');
    let idx: usize = parts.next()?.parse().ok()?;
    let tail = parts.next()?;
    Some((idx, tail))
}

fn ln_part(tail: &str) -> &'static str {
    if tail.ends_with("weight") {
        "gamma"
    } else {
        "beta"
    }
}

pub fn map_hf_tensor(key: &str) -> Option<String> {
    let key = key.strip_prefix("model.")?;

    key.strip_prefix("encoder.conv1.")
        .map(|rest| format!("encoder/conv1/{rest}.npy"))
        .or_else(|| {
            key.strip_prefix("encoder.conv2.")
                .map(|rest| format!("encoder/conv2/{rest}.npy"))
        })
        .or_else(|| {
            (key == "encoder.embed_positions.weight")
                .then(|| "encoder/positional_embedding.npy".into())
        })
        .or_else(|| {
            key.strip_prefix("encoder.layer_norm.").map(|rest| {
                format!("encoder/ln_post/{}.npy", ln_part(rest))
            })
        })
        .or_else(|| {
            key.strip_prefix("encoder.layers.")
                .and_then(map_encoder_layer)
        })
        .or_else(|| {
            (key == "decoder.embed_tokens.weight")
                .then(|| "decoder/token_embedding/weight.npy".into())
        })
        .or_else(|| {
            (key == "decoder.embed_positions.weight")
                .then(|| "decoder/positional_embedding.npy".into())
        })
        .or_else(|| {
            key.strip_prefix("decoder.layer_norm.").map(|rest| {
                format!("decoder/ln/{}.npy", ln_part(rest))
            })
        })
        .or_else(|| {
            key.strip_prefix("decoder.layers.")
                .and_then(map_decoder_layer)
        })
}

fn map_encoder_layer(rest: &str) -> Option<String> {
    let (idx, tail) = split_layer_tail(rest)?;
    match tail {
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
        "self_attn.out_proj.weight" | "self_attn.out_proj.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("encoder/block_{idx}/attn/out/{part}.npy"))
        }
        "self_attn_layer_norm.weight" | "self_attn_layer_norm.bias" => {
            Some(format!("encoder/block_{idx}/attn_ln/{}.npy", ln_part(tail)))
        }
        "fc1.weight" | "fc1.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("encoder/block_{idx}/mlp/lin1/{part}.npy"))
        }
        "fc2.weight" | "fc2.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("encoder/block_{idx}/mlp/lin2/{part}.npy"))
        }
        "final_layer_norm.weight" | "final_layer_norm.bias" => {
            Some(format!("encoder/block_{idx}/mlp_ln/{}.npy", ln_part(tail)))
        }
        _ => None,
    }
}

fn map_decoder_layer(rest: &str) -> Option<String> {
    let (idx, tail) = split_layer_tail(rest)?;
    match tail {
        "self_attn.q_proj.weight" | "self_attn.q_proj.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("decoder/block_{idx}/attn/query/{part}.npy"))
        }
        "self_attn.k_proj.weight" | "self_attn.k_proj.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("decoder/block_{idx}/attn/key/{part}.npy"))
        }
        "self_attn.v_proj.weight" | "self_attn.v_proj.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("decoder/block_{idx}/attn/value/{part}.npy"))
        }
        "self_attn.out_proj.weight" | "self_attn.out_proj.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("decoder/block_{idx}/attn/out/{part}.npy"))
        }
        "self_attn_layer_norm.weight" | "self_attn_layer_norm.bias" => {
            Some(format!("decoder/block_{idx}/attn_ln/{}.npy", ln_part(tail)))
        }
        "encoder_attn.q_proj.weight" | "encoder_attn.q_proj.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("decoder/block_{idx}/cross_attn/query/{part}.npy"))
        }
        "encoder_attn.k_proj.weight" | "encoder_attn.k_proj.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("decoder/block_{idx}/cross_attn/key/{part}.npy"))
        }
        "encoder_attn.v_proj.weight" | "encoder_attn.v_proj.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("decoder/block_{idx}/cross_attn/value/{part}.npy"))
        }
        "encoder_attn.out_proj.weight" | "encoder_attn.out_proj.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("decoder/block_{idx}/cross_attn/out/{part}.npy"))
        }
        "encoder_attn_layer_norm.weight" | "encoder_attn_layer_norm.bias" => {
            Some(format!("decoder/block_{idx}/cross_attn_ln/{}.npy", ln_part(tail)))
        }
        "fc1.weight" | "fc1.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("decoder/block_{idx}/mlp/lin1/{part}.npy"))
        }
        "fc2.weight" | "fc2.bias" => {
            let part = tail.rsplit_once('.')?.1;
            Some(format!("decoder/block_{idx}/mlp/lin2/{part}.npy"))
        }
        "final_layer_norm.weight" | "final_layer_norm.bias" => {
            Some(format!("decoder/block_{idx}/mlp_ln/{}.npy", ln_part(tail)))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::map_hf_tensor;

    #[test]
    fn maps_double_digit_encoder_layers() {
        assert_eq!(
            map_hf_tensor("model.encoder.layers.10.self_attn.q_proj.weight").as_deref(),
            Some("encoder/block_10/attn/query/weight.npy")
        );
        assert_eq!(
            map_hf_tensor("model.encoder.layers.1.self_attn.q_proj.weight").as_deref(),
            Some("encoder/block_1/attn/query/weight.npy")
        );
    }
}