use burn::nn::conv::{Conv1dConfig, Conv2dConfig};
use burn::nn::{PaddingConfig1d, PaddingConfig2d};

use crate::model::conformer::FeedForwardModule;
use crate::model::load::components::{load_conv1d, load_conv2d, load_layer_norm, load_linear};
use crate::model::load::helpers::{load_tensor, load_usize};
use crate::model::load::parakeet::load_parakeet_decoder;
use crate::model::nemotron::{NemotronAsr, NemotronAttention, NemotronBlock, NemotronConfig, NemotronConv};
use std::error::Error;

pub fn load_nemotron(path: &str, device: &burn::tensor::Device) -> Result<(NemotronAsr, NemotronConfig), Box<dyn Error>> {
    let enc = format!("{path}/encoder");
    let d_model = load_usize("d_model", &enc, device)?;
    let n_head = load_usize("n_head", &enc, device)?;
    let n_layers = load_usize("n_layers", &enc, device)?;
    let n_mels = load_usize("n_mels", &enc, device)?;
    let conv_kernel = load_usize("conv_kernel", &enc, device).unwrap_or(9);
    let sub_channels = load_usize("sub_channels", &enc, device).unwrap_or(256);
    let sub_kernel = load_usize("sub_kernel", &enc, device).unwrap_or(3);
    let ff_inner = load_usize("ff_inner", &enc, device).unwrap_or(d_model * 4);
    let lookahead = load_usize("lookahead", &enc, device).unwrap_or(13);
    let sliding_window = load_usize("sliding_window", &enc, device).unwrap_or(71);
    let decoder_hidden = load_usize("decoder_hidden", &enc, device).unwrap_or(640);
    let decoder_layers = load_usize("decoder_layers", &enc, device).unwrap_or(2);
    let max_symbols = load_usize("max_symbols", &enc, device).unwrap_or(10);
    let blank_id = load_usize("blank_id", &enc, device).unwrap_or(1024);
    let joint_w = load_tensor::<2>("weight", &format!("{path}/ctc_linear"), device)?;
    let n_vocab = joint_w.dims()[1];

    let cfg = NemotronConfig {
        d_model,
        n_head,
        n_layers,
        n_mels,
        n_vocab,
        decoder_hidden,
        decoder_layers,
        conv_channels: sub_channels,
        sub_kernel,
        conv_kernel,
        ff_inner,
        lookahead,
        sliding_window,
        max_symbols,
        blank_id,
    };
    let mut model = cfg.clone().init(device);
    model.sub.conv_in = load_causal2d(&format!("{enc}/sub/conv_in"), 1, device)?;
    model.sub.dw0 = load_causal2d(&format!("{enc}/sub/dw0"), sub_channels, device)?;
    model.sub.pw0 = load_point2d(&format!("{enc}/sub/pw0"), device)?;
    model.sub.dw1 = load_causal2d(&format!("{enc}/sub/dw1"), sub_channels, device)?;
    model.sub.pw1 = load_point2d(&format!("{enc}/sub/pw1"), device)?;
    let lin_w = load_tensor::<2>("weight", &format!("{enc}/pre_encode_out"), device)?;
    let [lin_in, lin_out] = lin_w.dims();
    drop(lin_w);
    model.sub.linear = load_linear(&format!("{enc}/pre_encode_out"), lin_in, lin_out, device)?;

    for i in 0..n_layers {
        model.layers[i] = load_block(&format!("{enc}/block_{i}"), d_model, n_head, conv_kernel, ff_inner, device)?;
    }

    let fp = load_tensor::<2>("weight", &format!("{path}/final_proj"), device)?;
    let [a, b] = fp.dims();
    drop(fp);
    model.final_proj = load_linear(&format!("{path}/final_proj"), a, b, device)?;
    let [jin, jout] = joint_w.dims();
    drop(joint_w);
    model.joint = load_linear(&format!("{path}/ctc_linear"), jin, jout, device)?;
    model.decoder = load_parakeet_decoder(path, device)?.ok_or("nemotron checkpoint has no LSTM decoder")?;
    Ok((model, cfg))
}

fn load_block(
    path: &str,
    d_model: usize,
    n_head: usize,
    conv_kernel: usize,
    ff_inner: usize,
    device: &burn::tensor::Device,
) -> Result<NemotronBlock, Box<dyn Error>> {
    Ok(NemotronBlock {
        ff1: load_ff(&format!("{path}/ff1"), d_model, ff_inner, device)?,
        attn: load_attn(&format!("{path}/attn"), d_model, n_head, device)?,
        attn_ln: load_layer_norm(&format!("{path}/attn_ln"), d_model, device)?,
        conv_ln: load_layer_norm(&format!("{path}/conv_ln"), d_model, device)?,
        conv: load_conv(&format!("{path}/conv"), d_model, conv_kernel, device)?,
        ff2: load_ff(&format!("{path}/ff2"), d_model, ff_inner, device)?,
        final_ln: load_layer_norm(&format!("{path}/final_ln"), d_model, device)?,
    })
}

fn load_ff(path: &str, d_model: usize, inner: usize, device: &burn::tensor::Device) -> Result<FeedForwardModule, Box<dyn Error>> {
    Ok(FeedForwardModule { lora_lin1: None, lora_lin2: None,
        ln: load_layer_norm(&format!("{path}/ln"), d_model, device)?,
        lin1: load_linear(&format!("{path}/lin1"), d_model, inner, device)?,
        lin2: load_linear(&format!("{path}/lin2"), inner, d_model, device)?,
    })
}

fn load_attn(path: &str, d_model: usize, n_head: usize, device: &burn::tensor::Device) -> Result<NemotronAttention, Box<dyn Error>> {
    let pos_bias_u = load_tensor::<2>("pos_bias_u", path, device)?;
    let pos_bias_v = load_tensor::<2>("pos_bias_v", path, device)?;
    Ok(NemotronAttention {
        n_head,
        query: load_linear(&format!("{path}/query"), d_model, d_model, device)?,
        key: load_linear(&format!("{path}/key"), d_model, d_model, device)?,
        value: load_linear(&format!("{path}/value"), d_model, d_model, device)?,
        out: load_linear(&format!("{path}/out"), d_model, d_model, device)?,
        pos: load_linear(&format!("{path}/pos"), d_model, d_model, device)?,
        pos_bias_u: burn::module::Param::from_tensor(pos_bias_u),
        pos_bias_v: burn::module::Param::from_tensor(pos_bias_v),
    })
}

fn load_conv(path: &str, d_model: usize, kernel: usize, device: &burn::tensor::Device) -> Result<NemotronConv, Box<dyn Error>> {
    let dw = load_tensor::<3>("weight", &format!("{path}/depth_conv"), device)?;
    let k = dw.dims()[2];
    drop(dw);
    let point_conv1 = load_conv1d(
        &format!("{path}/point_conv1"),
        Conv1dConfig::new(d_model, d_model * 2, 1),
        device,
    )?;
    let depth_conv = load_conv1d(
        &format!("{path}/depth_conv"),
        Conv1dConfig::new(d_model, d_model, k)
            .with_groups(d_model)
            .with_padding(PaddingConfig1d::Explicit(k.saturating_sub(1), 0)),
        device,
    )?;
    let point_conv2 = load_conv1d(
        &format!("{path}/point_conv2"),
        Conv1dConfig::new(d_model, d_model, 1),
        device,
    )?;
    let _ = kernel;
    Ok(NemotronConv {
        point_conv1,
        depth_conv,
        post_ln: load_layer_norm(&format!("{path}/post_ln"), d_model, device)?,
        point_conv2,
    })
}

fn load_causal2d(path: &str, groups_in: usize, device: &burn::tensor::Device) -> Result<burn::nn::conv::Conv2d, Box<dyn Error>> {
    let w = load_tensor::<4>("weight", path, device)?;
    let [out_ch, in_per, k, _] = w.dims();
    drop(w);
    let groups = if groups_in == 1 { 1 } else { out_ch };
    let _ = in_per;
    let cfg = Conv2dConfig::new([if groups == 1 { 1 } else { out_ch }, out_ch], [k, k])
        .with_stride([2, 2])
        .with_groups(groups)
        .with_padding(PaddingConfig2d::Explicit(k - 1, k - 1, 1, 1));
    load_conv2d(path, cfg, device)
}

fn load_point2d(path: &str, device: &burn::tensor::Device) -> Result<burn::nn::conv::Conv2d, Box<dyn Error>> {
    let w = load_tensor::<4>("weight", path, device)?;
    let [out_ch, in_ch, k1, k2] = w.dims();
    drop(w);
    load_conv2d(path, Conv2dConfig::new([in_ch, out_ch], [k1, k2]), device)
}
