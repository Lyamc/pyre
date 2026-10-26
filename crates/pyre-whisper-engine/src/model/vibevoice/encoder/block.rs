use burn::module::{Module, Param};
use burn::nn::{Gelu, Linear, LinearConfig};
use burn::tensor::{Tensor};

use super::norm::ConvRmsNorm;
use super::sconv::SConv1d;

#[derive(Module, Debug)]
pub struct FfnBlock {
    pub linear1: Linear,
    pub gelu: Gelu,
    pub linear2: Linear,
}

#[derive(Module, Debug)]
pub struct Block1D {
    pub norm: ConvRmsNorm,
    pub mixer: SConv1d,
    pub gamma: Param<Tensor<1>>,
    pub ffn_norm: ConvRmsNorm,
    pub ffn: FfnBlock,
    pub ffn_gamma: Param<Tensor<1>>,
}

impl Block1D {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let residual = x.clone();
        let x = self.norm.forward(x);
        let x = self.mixer.forward(x);
        let x = x * self.gamma.val().unsqueeze_dims(&[0, 2]);
        let x = residual + x;

        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = x.swap_dims(1, 2);
        let x = self.ffn.linear2.forward(self.ffn.gelu.forward(self.ffn.linear1.forward(x)));
        let x = x.swap_dims(1, 2);
        let x = x * self.ffn_gamma.val().unsqueeze_dims(&[0, 2]);
        residual + x
    }
}

pub fn init_ffn(dim: usize, ffn_dim: usize, device: &burn::tensor::Device) -> FfnBlock {
    FfnBlock {
        linear1: LinearConfig::new(dim, ffn_dim).init(device),
        gelu: Gelu::new(),
        linear2: LinearConfig::new(ffn_dim, dim).init(device),
    }
}