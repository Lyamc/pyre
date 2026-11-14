use burn::{
    module::Module,
    nn::{self, conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig}, BatchNorm, BatchNormConfig, Linear, LinearConfig, LayerNorm, LayerNormConfig},
    tensor::{Tensor},
    module::Param,
};
use crate::model::*;
use crate::model::conformer::*;
use crate::model::attention::*;
use crate::model::load::helpers::*;
use std::error::Error;

pub fn load_linear(
    path: &str,
    d_in: usize,
    d_out: usize,
    device: &burn::tensor::Device,
) -> Result<Linear, Box<dyn Error>> {
    let weight = load_tensor::<2>("weight", path, device)?;
    let bias = if tensor_exists("bias", path) {
        Some(load_tensor::<1>("bias", path, device)?)
    } else {
        None
    };

    let linear = LinearConfig::new(d_in, d_out)
        .with_bias(bias.is_some())
        .init(device);
    let mut record = linear.clone();
    record.weight = Param::from_tensor(weight);
    record.bias = bias.map(Param::from_tensor);

    Ok(record)
}

pub fn load_conv1d(
    path: &str,
    config: Conv1dConfig,
    device: &burn::tensor::Device,
) -> Result<Conv1d, Box<dyn Error>> {
    let weight = load_tensor::<3>("weight", path, device)?;
    let bias = if tensor_exists("bias", path) {
        Some(load_tensor::<1>("bias", path, device)?)
    } else {
        None
    };

    let conv = config.with_bias(bias.is_some()).init(device);
    let mut record = conv.clone();
    record.weight = Param::from_tensor(weight);
    record.bias = bias.map(Param::from_tensor);

    Ok(record)
}

pub fn load_conv2d(
    path: &str,
    config: Conv2dConfig,
    device: &burn::tensor::Device,
) -> Result<Conv2d, Box<dyn Error>> {
    let weight = load_tensor::<4>("weight", path, device)?;
    let bias = if tensor_exists("bias", path) {
        Some(load_tensor::<1>("bias", path, device)?)
    } else {
        None
    };

    let conv = config.with_bias(bias.is_some()).init(device);
    let mut record = conv.clone();
    record.weight = Param::from_tensor(weight);
    record.bias = bias.map(Param::from_tensor);

    Ok(record)
}

pub fn load_layer_norm(
    path: &str,
    n_state: usize,
    device: &burn::tensor::Device,
) -> Result<LayerNorm, Box<dyn Error>> {
    let weight = load_tensor::<1>("gamma", path, device)
        .or_else(|_| load_tensor::<1>("weight", path, device))?;
    let bias = load_tensor::<1>("beta", path, device)
        .or_else(|_| load_tensor::<1>("bias", path, device))?;
    let eps = load_tensor::<1>("eps", path, device)
        .map(|t| t.into_data().to_vec::<f32>().unwrap()[0] as f64)
        .unwrap_or(1e-5);

    let ln = LayerNormConfig::new(n_state).with_epsilon(eps).init(device);
    let mut record = ln.clone();
    record.gamma = Param::from_tensor(weight);
    record.beta = Some(Param::from_tensor(bias));

    Ok(record)
}

pub fn load_batch_norm(path: &str, n_state: usize, device: &burn::tensor::Device) -> Result<BatchNorm, Box<dyn Error>> {
    let weight = if tensor_exists("weight", path) { load_tensor::<1>("weight", path, device)? } else { Tensor::ones([n_state], device) };
    let bias = if tensor_exists("bias", path) { load_tensor::<1>("bias", path, device)? } else { Tensor::zeros([n_state], device) };
    let mean = if tensor_exists("running_mean", path) { load_tensor::<1>("running_mean", path, device)? } else { Tensor::zeros([n_state], device) };
    let var = if tensor_exists("running_var", path) { load_tensor::<1>("running_var", path, device)? } else { Tensor::ones([n_state], device) };

    let bn = BatchNormConfig::new(n_state).init(device);
    let mut record = bn.clone();
    record.gamma = Param::from_tensor(weight);
    record.beta = Param::from_tensor(bias);
    record.running_mean = burn::module::RunningState::new(mean);
    record.running_var = burn::module::RunningState::new(var);

    Ok(record)
}

pub fn load_multihead_attention(
    path: &str,
    n_state: usize,
    device: &burn::tensor::Device,
) -> Result<MultiHeadAttention, Box<dyn Error>> {
    let n_head: usize = load_usize("n_head", path, device)?;
    let query = load_linear(&format!("{}/{}", path, "query"), n_state, n_state, device)?;
    
    // Key bias is optional in many Whisper models
    let key_path = format!("{}/{}", path, "key");
    let key_weight = load_tensor::<2>("weight", &key_path, device)?;
    let key_bias = load_tensor::<1>("bias", &key_path, device).ok();
    let key = {
        let linear = nn::LinearConfig::new(n_state, n_state).init(device);
        let mut record = linear.clone();
        record.weight = Param::from_tensor(key_weight);
        record.bias = key_bias.map(Param::from_tensor);
        record
    };

    let value = load_linear(&format!("{}/{}", path, "value"), n_state, n_state, device)?;
    let out = load_linear(&format!("{}/{}", path, "out"), n_state, n_state, device)?;
    
    Ok(MultiHeadAttention { n_head, query, key, value, out })
}

pub fn load_rel_pos_multihead_attention(
    path: &str,
    n_state: usize,
    device: &burn::tensor::Device,
) -> Result<RelPosMultiHeadAttention, Box<dyn Error>> {
    let n_head: usize = load_usize("n_head", path, device)?;
    let query = load_linear(&format!("{}/{}", path, "query"), n_state, n_state, device)?;
    let key = load_linear(&format!("{}/{}", path, "key"), n_state, n_state, device)?;
    let value = load_linear(&format!("{}/{}", path, "value"), n_state, n_state, device)?;
    let out = load_linear(&format!("{}/{}", path, "out"), n_state, n_state, device)?;
    
    let pos = load_linear(&format!("{}/{}", path, "pos"), n_state, n_state, device)?;
    let pos_bias_u = load_tensor::<2>("pos_bias_u", path, device)?;
    let pos_bias_v = load_tensor::<2>("pos_bias_v", path, device)?;

    Ok(RelPosMultiHeadAttention {
        n_head,
        query,
        key,
        value,
        out,
        pos,
        pos_bias_u: Param::from_tensor(pos_bias_u),
        pos_bias_v: Param::from_tensor(pos_bias_v),
        lora_q: None,
        lora_k: None,
        lora_v: None,
        lora_o: None,
        lora_pos: None,
    })
}

pub fn load_convolution_module(path: &str, d_model: usize, _kernel_size: usize, device: &burn::tensor::Device) -> Result<ConvolutionModule, Box<dyn Error>> {
    let ln = load_layer_norm(&format!("{}/{}", path, "ln"), d_model, device)?;
    
    let pc1_weight = load_tensor::<3>("weight", &format!("{}/{}", path, "point_conv1"), device)?;
    let [pc1_out, pc1_in, pc1_k] = pc1_weight.dims();
    let point_conv1_cfg = nn::conv::Conv1dConfig::new(pc1_in, pc1_out, pc1_k);
    let point_conv1 = load_conv1d(&format!("{}/{}", path, "point_conv1"), point_conv1_cfg, device)?;
    
    let dc_weight = load_tensor::<3>("weight", &format!("{}/{}", path, "depth_conv"), device)?;
    let [dc_out, dc_in_reduced, dc_k] = dc_weight.dims();
    let dc_in = if dc_in_reduced == 1 { dc_out } else { dc_in_reduced };
    let mut depth_conv_cfg = nn::conv::Conv1dConfig::new(dc_in, dc_out, dc_k)
        .with_padding(nn::PaddingConfig1d::Explicit(dc_k / 2, dc_k / 2));
    if dc_in_reduced == 1 { depth_conv_cfg = depth_conv_cfg.with_groups(dc_in); }
    let depth_conv = load_conv1d(&format!("{}/{}", path, "depth_conv"), depth_conv_cfg, device)?;
    
    let bn = load_batch_norm(&format!("{}/{}", path, "bn"), d_model, device)?;

    let pc2_weight = load_tensor::<3>("weight", &format!("{}/{}", path, "point_conv2"), device)?;
    let [pc2_out, pc2_in, pc2_k] = pc2_weight.dims();
    let point_conv2_cfg = nn::conv::Conv1dConfig::new(pc2_in, pc2_out, pc2_k);
    let point_conv2 = load_conv1d(&format!("{}/{}", path, "point_conv2"), point_conv2_cfg, device)?;

    Ok(ConvolutionModule {
        ln,
        point_conv1,
        depth_conv,
        bn,
        point_conv2,
        lora_pw1: None,
        lora_pw2: None,
    })
}

pub fn load_feed_forward_module(path: &str, d_model: usize, device: &burn::tensor::Device) -> Result<FeedForwardModule, Box<dyn Error>> {
    let ln = load_layer_norm(&format!("{}/{}", path, "ln"), d_model, device)?;
    let lin1 = load_linear(&format!("{}/{}", path, "lin1"), d_model, d_model * 4, device)?;
    let lin2 = load_linear(&format!("{}/{}", path, "lin2"), d_model * 4, d_model, device)?;
    Ok(FeedForwardModule { ln, lin1, lin2, lora_lin1: None, lora_lin2: None })
}

pub fn load_conformer_block(path: &str, d_model: usize, _n_head: usize, kernel_size: usize, device: &burn::tensor::Device) -> Result<ConformerBlock, Box<dyn Error>> {
    let ff1 = load_feed_forward_module(&format!("{}/{}", path, "ff1"), d_model, device)?;
    // RelPos attention for Conformer
    let attn = load_rel_pos_multihead_attention(&format!("{}/{}", path, "attn"), d_model, device)?;
    let attn_ln = load_layer_norm(&format!("{}/{}", path, "attn_ln"), d_model, device)?;
    let conv = load_convolution_module(&format!("{}/{}", path, "conv"), d_model, kernel_size, device)?;
    let ff2 = load_feed_forward_module(&format!("{}/{}", path, "ff2"), d_model, device)?;
    let final_ln = load_layer_norm(&format!("{}/{}", path, "final_ln"), d_model, device)?;

    Ok(ConformerBlock { ff1, attn, attn_ln, conv, ff2, final_ln })
}

// Whisper specific loaders
pub fn load_residual_encoder_attention_block(path: &str, n_state: usize, device: &burn::tensor::Device) -> Result<ResidualEncoderAttentionBlock, Box<dyn Error>> {
    let attn = load_multihead_attention(&format!("{}/{}", path, "attn"), n_state, device)?;
    let attn_ln = load_layer_norm(&format!("{}/{}", path, "attn_ln"), n_state, device)?;
    let mlp = load_mlp(&format!("{}/{}", path, "mlp"), n_state, device)?;
    let mlp_ln = load_layer_norm(&format!("{}/{}", path, "mlp_ln"), n_state, device)?;
    Ok(ResidualEncoderAttentionBlock { attn, attn_ln, mlp, mlp_ln })
}

pub fn load_residual_decoder_attention_block(path: &str, n_state: usize, device: &burn::tensor::Device) -> Result<ResidualDecoderAttentionBlock, Box<dyn Error>> {
    let attn = load_multihead_attention(&format!("{}/{}", path, "attn"), n_state, device)?;
    let attn_ln = load_layer_norm(&format!("{}/{}", path, "attn_ln"), n_state, device)?;
    let cross_attn = load_multihead_cross_attention(&format!("{}/{}", path, "cross_attn"), n_state, device)?;
    let cross_attn_ln = load_layer_norm(&format!("{}/{}", path, "cross_attn_ln"), n_state, device)?;
    let mlp = load_mlp(&format!("{}/{}", path, "mlp"), n_state, device)?;
    let mlp_ln = load_layer_norm(&format!("{}/{}", path, "mlp_ln"), n_state, device)?;
    Ok(ResidualDecoderAttentionBlock { attn, attn_ln, cross_attn, cross_attn_ln, mlp, mlp_ln })
}

pub fn load_multihead_cross_attention(path: &str, n_state: usize, device: &burn::tensor::Device) -> Result<MultiHeadCrossAttention, Box<dyn Error>> {
    let n_head: usize = load_usize("n_head", path, device)?;
    let query = load_linear(&format!("{}/{}", path, "query"), n_state, n_state, device)?;
    let key = load_linear(&format!("{}/{}", path, "key"), n_state, n_state, device)?;
    let value = load_linear(&format!("{}/{}", path, "value"), n_state, n_state, device)?;
    let out = load_linear(&format!("{}/{}", path, "out"), n_state, n_state, device)?;
    Ok(MultiHeadCrossAttention { n_head, query, key, value, out })
}

pub fn load_mlp(path: &str, n_state: usize, device: &burn::tensor::Device) -> Result<MLP, Box<dyn Error>> {
    let lin1 = load_linear(&format!("{}/{}", path, "lin1"), n_state, n_state * 4, device)?;
    let lin2 = load_linear(&format!("{}/{}", path, "lin2"), n_state * 4, n_state, device)?;
    Ok(MLP { lin1, gelu: nn::Gelu::new(), lin2 })
}
