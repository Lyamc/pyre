use burn::nn;
use burn::tensor::{Tensor};
use burn::module::Param;
use crate::model::*;
use crate::model::conformer::*;
use crate::model::load::helpers::*;
use crate::model::parakeet_decoder::{ParakeetDecoder, ParakeetLstm, ParakeetLstmLayer};
use std::error::Error;
use std::path::Path;

fn load_parakeet_decoder(
    path: &str,
    device: &burn::tensor::Device,
) -> Result<Option<ParakeetDecoder>, Box<dyn Error>> {
    let embed_path = format!("{path}/decoder/embedding/weight.npy");
    if !Path::new(&embed_path).exists() {
        return Ok(None);
    }

    let embed_weight = load_tensor::<2>("weight", &format!("{path}/decoder/embedding"), device)?;
    let [_n_vocab, _hidden_size] = embed_weight.dims();
    let embedding = nn::Embedding {
        weight: Param::from_tensor(embed_weight),
    };

    let mut lstm_layers = Vec::new();
    for layer_idx in 0..8 {
        let layer_path = format!("{path}/decoder/lstm/l{layer_idx}");
        if !Path::new(&format!("{layer_path}/weight_ih.npy")).exists() {
            break;
        }
        let weight_ih = load_tensor::<2>("weight_ih", &layer_path, device)?;
        let weight_hh = load_tensor::<2>("weight_hh", &layer_path, device)?;
        let bias_ih = load_tensor::<1>("bias_ih", &layer_path, device)?;
        let bias_hh = load_tensor::<1>("bias_hh", &layer_path, device)?;
        lstm_layers.push(ParakeetLstmLayer {
            weight_ih: Param::from_tensor(weight_ih),
            weight_hh: Param::from_tensor(weight_hh),
            bias_ih: Param::from_tensor(bias_ih),
            bias_hh: Param::from_tensor(bias_hh),
        });
    }

    if lstm_layers.is_empty() {
        return Ok(None);
    }

    let proj_weight = load_tensor::<2>("weight", &format!("{path}/decoder/projector"), device)?;
    let [d_out, d_in] = proj_weight.dims();
    let projector = load_linear(&format!("{path}/decoder/projector"), d_in, d_out, device)?;

    Ok(Some(ParakeetDecoder {
        embedding,
        lstm: ParakeetLstm { layers: lstm_layers },
        projector,
    }))
}

pub fn load_parakeet(path: &str, device: &burn::tensor::Device) -> Result<Parakeet, Box<dyn Error>> {
    let n_layers = load_usize("n_layers", &format!("{}/encoder", path), device)?;
    let d_model = load_usize("d_model", &format!("{}/encoder", path), device)?;
    let n_head = load_usize("n_head", &format!("{}/encoder", path), device)?;
    
    let kernel_size = if Path::new(&format!("{}/encoder/block_0/conv/depth_conv/weight.npy", path)).exists() {
        let weight = load_tensor::<3>("weight", &format!("{}/encoder/block_0/conv/depth_conv", path), device)?;
        weight.dims()[2]
    } else {
        31 // Default fallback
    };

    // Load subsampling convolutions (5 layers for 8x subsampling)
    let conv1 = {
        let weight = load_tensor::<4>("weight", &format!("{}/encoder/conv1", path), device)?;
        let [out_ch, in_ch, k1, k2] = weight.dims();
        let cfg = nn::conv::Conv2dConfig::new([in_ch, out_ch], [k1, k2])
            .with_padding(nn::PaddingConfig2d::Explicit(k1 / 2, k2 / 2, k1 / 2, k2 / 2))
            .with_stride([2, 2]);
        load_conv2d(&format!("{}/encoder/conv1", path), cfg, device)?
    };

    let conv2 = {
        let weight = load_tensor::<4>("weight", &format!("{}/encoder/conv2", path), device)?;
        let [out_ch, in_ch_per_group, k1, k2] = weight.dims();
        let groups = out_ch; // For depthwise, groups == out_channels
        let in_ch = in_ch_per_group * groups; 
        let cfg = nn::conv::Conv2dConfig::new([in_ch, out_ch], [k1, k2])
            .with_padding(nn::PaddingConfig2d::Explicit(k1 / 2, k2 / 2, k1 / 2, k2 / 2))
            .with_stride([2, 2])
            .with_groups(groups);
        load_conv2d(&format!("{}/encoder/conv2", path), cfg, device)?
    };

    let conv3 = {
        let weight = load_tensor::<4>("weight", &format!("{}/encoder/conv3", path), device)?;
        let [out_ch, in_ch, k1, k2] = weight.dims();
        let cfg = nn::conv::Conv2dConfig::new([in_ch, out_ch], [k1, k2]);
        load_conv2d(&format!("{}/encoder/conv3", path), cfg, device)?
    };

    let conv4 = {
        let weight = load_tensor::<4>("weight", &format!("{}/encoder/conv4", path), device)?;
        let [out_ch, in_ch_per_group, k1, k2] = weight.dims();
        let groups = out_ch;
        let in_ch = in_ch_per_group * groups;
        let cfg = nn::conv::Conv2dConfig::new([in_ch, out_ch], [k1, k2])
            .with_padding(nn::PaddingConfig2d::Explicit(k1 / 2, k2 / 2, k1 / 2, k2 / 2))
            .with_stride([2, 2])
            .with_groups(groups);
        load_conv2d(&format!("{}/encoder/conv4", path), cfg, device)?
    };

    let conv5 = {
        let weight = load_tensor::<4>("weight", &format!("{}/encoder/conv5", path), device)?;
        let [out_ch, in_ch, k1, k2] = weight.dims();
        let cfg = nn::conv::Conv2dConfig::new([in_ch, out_ch], [k1, k2]);
        load_conv2d(&format!("{}/encoder/conv5", path), cfg, device)?
    };

    let pe_out_weight = load_tensor::<2>("weight", &format!("{}/encoder/pre_encode_out", path), device)?;
    let [pe_out, pe_in] = pe_out_weight.dims(); 
    let pre_encode_out = load_linear(&format!("{}/encoder/pre_encode_out", path), pe_in, pe_out, device)?;

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        layers.push(load_conformer_block(&format!("{}/encoder/block_{}", path, i), d_model, n_head, kernel_size, device)?);
    }
    
    let encoder = ConformerEncoder { layers };
    
    // Final projection to joint space
    let final_proj = if Path::new(&format!("{}/final_proj/weight.npy", path)).exists() {
        let weight = load_tensor::<2>("weight", &format!("{}/final_proj", path), device)?;
        let [d_out, d_in] = weight.dims();
        load_linear(&format!("{}/final_proj", path), d_in, d_out, device)?
    } else {
        nn::LinearConfig::new(d_model, 640).init(device)
    };

    let weight = load_tensor::<2>("weight", &format!("{}/ctc_linear", path), device)?;
    let [n_v, n_in] = weight.dims();
    let ctc_linear = load_linear(&format!("{}/ctc_linear", path), n_in, n_v, device)?;

    // Load decoder projection bias (size 640)
    let joint_pred_bias = if Path::new(&format!("{}/final_proj/bias.npy", path)).exists() {
        // We actually want joint.pred.bias
        if tensor_exists("bias", &format!("{}/joint_pred", path)) {
             load_tensor::<1>("bias", &format!("{}/joint_pred", path), device)?
        } else {
             Tensor::zeros([640], device)
        }
    } else {
        Tensor::zeros([640], device)
    };

    let n_mels = if Path::new(&format!("{}/encoder/n_mels.npy", path)).exists() {
        load_usize("n_mels", &format!("{}/encoder", path), device)?
    } else {
        128 // Default for Parakeet-TDT v3
    };
// Load global mel stats if they exist
let mean = if Path::new(&format!("{}/mean.npy", path)).exists() {
    load_tensor::<2>("mean", &path, device).ok().map(Param::from_tensor)
} else {
    None
};
let std = if Path::new(&format!("{}/std.npy", path)).exists() {
    load_tensor::<2>("std", &path, device).ok().map(Param::from_tensor)
} else {
    None
};

    let decoder = load_parakeet_decoder(path, device)?;

    Ok(Parakeet {
        conv1,
        conv2,
        conv3,
        conv4,
        conv5,
        pre_encode_out,
        encoder,
        final_proj,
        ctc_linear,
        joint_pred_bias: Param::from_tensor(joint_pred_bias),
        decoder,
        mean,
        std,
        n_mels,
    })
}

