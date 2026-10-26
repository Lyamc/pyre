use burn::{
    nn,
    tensor::Tensor,
    config::Config,
    module::Module,
};

#[derive(Config, Debug)]
pub struct MLPConfig {
    pub n_state: usize,
}

impl MLPConfig {
    pub fn init(&self, tensor_device_ref: &burn::tensor::Device) -> MLP {
        let lin1 = nn::LinearConfig::new(self.n_state, 4 * self.n_state).init(tensor_device_ref);
        let gelu = nn::Gelu::new();
        let lin2 = nn::LinearConfig::new(4 * self.n_state, self.n_state).init(tensor_device_ref);

        MLP { lin1, gelu, lin2 }
    }
}

#[derive(Module, Debug)]
pub struct MLP {
    pub lin1: nn::Linear,
    pub gelu: nn::Gelu,
    pub lin2: nn::Linear,
}

impl MLP {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = self.lin1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.lin2.forward(x);

        return x;
    }
}
