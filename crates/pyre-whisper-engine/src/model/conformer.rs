use burn::{
    config::Config,
    module::{Module, Param},
    nn::{conv::{Conv1d, Conv1dConfig}, Linear, LinearConfig, LayerNorm, LayerNormConfig, PaddingConfig1d, BatchNorm, BatchNormConfig},
    tensor::{Tensor},
};
use crate::model::attention::{RelPosMultiHeadAttention, RelPosMultiHeadAttentionConfig};

#[derive(Module, Debug)]
pub struct RMSNorm {
    pub gamma: Param<Tensor<1>>,
    #[module(skip)]
    pub epsilon: f64,
}

impl RMSNorm {
    pub fn new(d_model: usize, device: &burn::tensor::Device) -> Self {
        Self::with_eps(d_model, 1e-5, device)
    }

    pub fn with_eps(d_model: usize, epsilon: f64, device: &burn::tensor::Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::ones([d_model], device)),
            epsilon,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let rms = (x.clone().powf_scalar(2.0).mean_dim(D - 1) + self.epsilon).sqrt();
        (x / rms) * self.gamma.val().unsqueeze()
    }
}

#[derive(Config, Debug)]
pub struct ConformerEncoderConfig {
    pub d_model: usize,
    pub n_head: usize,
    pub n_layers: usize,
    pub kernel_size: usize,
    /// Feed-forward, attention, and conformer-conv projection bias.
    /// Published Parakeet v2/v3 set this false. Missing from an older cfg means false.
    #[config(default = false)]
    pub layer_bias: bool,
}

impl ConformerEncoderConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> ConformerEncoder {
        let layers = (0..self.n_layers)
            .map(|_| {
                ConformerBlockConfig::new(self.d_model, self.n_head, self.kernel_size)
                    .with_layer_bias(self.layer_bias)
                    .init(device)
            })
            .collect();
        ConformerEncoder { layers }
    }
}

#[derive(Module, Debug)]
pub struct ConformerEncoder {
    pub layers: Vec<ConformerBlock>,
}

impl ConformerEncoder {
    pub fn forward(&self, mut x: Tensor<3>) -> Tensor<3> {
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }
}

#[derive(Config, Debug)]
pub struct ConformerBlockConfig {
    pub d_model: usize,
    pub n_head: usize,
    pub kernel_size: usize,
    #[config(default = false)]
    pub layer_bias: bool,
}

impl ConformerBlockConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> ConformerBlock {
        let ff1 = FeedForwardModuleConfig::new(self.d_model)
            .with_bias(self.layer_bias)
            .init(device);
        let attn = RelPosMultiHeadAttentionConfig::new(self.d_model, self.n_head)
            .with_bias(self.layer_bias)
            .init(device);
        let attn_ln = LayerNormConfig::new(self.d_model).init(device);
        let conv = ConvolutionModuleConfig::new(self.d_model, self.kernel_size)
            .with_bias(self.layer_bias)
            .init(device);
        let ff2 = FeedForwardModuleConfig::new(self.d_model)
            .with_bias(self.layer_bias)
            .init(device);
        let final_ln = LayerNormConfig::new(self.d_model).init(device);

        ConformerBlock {
            ff1,
            attn,
            attn_ln,
            conv,
            ff2,
            final_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ConformerBlock {
    pub ff1: FeedForwardModule,
    pub attn: RelPosMultiHeadAttention,
    pub attn_ln: LayerNorm,
    pub conv: ConvolutionModule,
    pub ff2: FeedForwardModule,
    pub final_ln: LayerNorm,
}

impl ConformerBlock {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = x.clone() + self.ff1.forward(x.clone()).mul_scalar(0.5);
        let x = x.clone() + self.attn.forward(self.attn_ln.forward(x.clone()), None);
        let x = x.clone() + self.conv.forward(x.clone());
        let x = x.clone() + self.ff2.forward(x).mul_scalar(0.5);
        self.final_ln.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct FeedForwardModuleConfig {
    pub d_model: usize,
    #[config(default = false)]
    pub bias: bool,
}

impl FeedForwardModuleConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> FeedForwardModule {
        let ln = LayerNormConfig::new(self.d_model).init(device);
        let lin1 = LinearConfig::new(self.d_model, self.d_model * 4)
            .with_bias(self.bias)
            .init(device);
        let lin2 = LinearConfig::new(self.d_model * 4, self.d_model)
            .with_bias(self.bias)
            .init(device);
        FeedForwardModule { ln, lin1, lin2 }
    }
}

#[derive(Module, Debug)]
pub struct FeedForwardModule {
    pub ln: LayerNorm,
    pub lin1: Linear,
    pub lin2: Linear,
}

impl FeedForwardModule {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = self.ln.forward(x);
        
        // Burn Linear weights are [in, out] (the converter transposes nn.Linear [out, in]).
        // Manual lin1
        let x = x.matmul(self.lin1.weight.val().unsqueeze::<3>());
        let x = if let Some(bias) = &self.lin1.bias {
            x + bias.val().unsqueeze::<3>()
        } else {
            x
        };
        
        let x = burn::tensor::activation::silu(x);
        
        // Manual lin2
        let x = x.matmul(self.lin2.weight.val().unsqueeze::<3>());
        if let Some(bias) = &self.lin2.bias {
            x + bias.val().unsqueeze::<3>()
        } else {
            x
        }
    }
}

#[derive(Config, Debug)]
pub struct ConvolutionModuleConfig {
    pub d_model: usize,
    pub kernel_size: usize,
    #[config(default = false)]
    pub bias: bool,
}

impl ConvolutionModuleConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> ConvolutionModule {
        let ln = LayerNormConfig::new(self.d_model).init(device);
        // Batch-norm keeps its own affine parameters. These projections follow
        // `convolution_bias` from the checkpoint (`bias` on this config).
        let point_conv1 = Conv1dConfig::new(self.d_model, self.d_model * 2, 1)
            .with_bias(self.bias)
            .init(device);
        let depth_conv = Conv1dConfig::new(self.d_model, self.d_model, self.kernel_size)
            .with_groups(self.d_model)
            .with_padding(PaddingConfig1d::Explicit(self.kernel_size / 2, self.kernel_size / 2))
            .with_bias(self.bias)
            .init(device);
        let bn = BatchNormConfig::new(self.d_model).init(device);
        let point_conv2 = Conv1dConfig::new(self.d_model, self.d_model, 1)
            .with_bias(self.bias)
            .init(device);
        
        ConvolutionModule { ln, point_conv1, depth_conv, bn, point_conv2 }
    }
}

#[derive(Module, Debug)]
pub struct ConvolutionModule {
    pub ln: LayerNorm,
    pub point_conv1: Conv1d,
    pub depth_conv: Conv1d,
    pub bn: BatchNorm,
    pub point_conv2: Conv1d,
}

impl ConvolutionModule {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = self.ln.forward(x);
        let x = x.swap_dims(1, 2);
        
        let x = self.point_conv1.forward(x);
        let [b, d2, s] = x.dims();
        let d = d2 / 2;
        let x1 = x.clone().slice([0..b, 0..d, 0..s]);
        let x2 = x.slice([0..b, d..d2, 0..s]);
        let x = x1 * burn::tensor::activation::sigmoid(x2);
        
        let x = self.depth_conv.forward(x);
        let x = self.bn.forward(x);
        let x = burn::tensor::activation::silu(x);
        let x = self.point_conv2.forward(x);
        
        x.swap_dims(1, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::{ConvolutionModuleConfig, FeedForwardModuleConfig};

    #[test]
    fn published_checkpoints_omit_ff_and_conv_bias() {
        let device = burn::tensor::Device::flex();
        let ff = FeedForwardModuleConfig::new(8).init(&device);
        assert!(ff.lin1.bias.is_none());
        assert!(ff.lin2.bias.is_none());
        let conv = ConvolutionModuleConfig::new(8, 9).init(&device);
        assert!(conv.point_conv1.bias.is_none());
        assert!(conv.depth_conv.bias.is_none());
        assert!(conv.point_conv2.bias.is_none());
    }
}
