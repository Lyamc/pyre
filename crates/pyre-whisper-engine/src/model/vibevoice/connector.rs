use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Tensor};

#[derive(Config, Debug)]
pub struct SpeechConnectorConfig {
    pub input_dim: usize,
    pub hidden_size: usize,
}

impl SpeechConnectorConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> SpeechConnector {
        SpeechConnector {
            fc1: LinearConfig::new(self.input_dim, self.hidden_size).init(device),
            norm_gamma: Param::from_tensor(Tensor::ones([self.hidden_size], device)),
            fc2: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            eps: 1e-6,
        }
    }
}

#[derive(Module, Debug)]
pub struct SpeechConnector {
    pub fc1: Linear,
    pub norm_gamma: Param<Tensor<1>>,
    pub fc2: Linear,
    #[module(skip)]
    pub eps: f64,
}

impl SpeechConnector {
    pub fn forward(&self, features: Tensor<3>) -> Tensor<3> {
        let x = self.fc1.forward(features);
        let x = rms_norm(x, self.norm_gamma.val(), self.eps);
        self.fc2.forward(x)
    }
}

fn rms_norm(x: Tensor<3>, gamma: Tensor<1>, eps: f64) -> Tensor<3> {
    let variance = x.clone().powf_scalar(2.0).mean_dim(2);
    let normed = x * variance.add_scalar(eps).powf_scalar(-0.5);
    normed * gamma.unsqueeze_dims(&[0, 1])
}