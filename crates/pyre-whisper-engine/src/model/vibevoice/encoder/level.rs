use burn::module::Module;
use burn::tensor::{Tensor};

use super::block::Block1D;
use super::sconv::SConv1d;

#[derive(Module, Debug)]
pub struct EncoderLevel {
    pub conv: SConv1d,
    pub blocks: Vec<Block1D>,
}

impl EncoderLevel {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = self.conv.forward(x);
        self.blocks.iter().fold(x, |acc, block| block.forward(acc))
    }
}