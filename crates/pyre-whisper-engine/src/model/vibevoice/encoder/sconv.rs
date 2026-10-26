use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::PaddingConfig1d;
use burn::tensor::{Tensor};

use super::pad::{extra_padding_for_conv1d, padding_total, zero_pad1d};

#[derive(Module, Debug)]
pub struct SConv1d {
    pub conv: Conv1d,
    #[module(skip)]
    pub kernel_size: usize,
    #[module(skip)]
    pub stride: usize,
    #[module(skip)]
    pub padding_total: usize,
    #[module(skip)]
    pub groups: usize,
}

impl SConv1d {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        device: &burn::tensor::Device,
    ) -> Self {
        let groups = groups.max(1);
        let pad = padding_total(kernel_size, stride, 1);
        let conv = Conv1dConfig::new(in_ch, out_ch, kernel_size)
            .with_stride(stride)
            .with_groups(groups)
            .with_padding(PaddingConfig1d::Explicit(0, 0))
            .init(device);
        Self {
            conv,
            kernel_size,
            stride,
            padding_total: pad,
            groups,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let length = x.dims()[2];
        let extra = extra_padding_for_conv1d(length, self.kernel_size, self.stride, self.padding_total);
        let x = zero_pad1d(x, self.padding_total, extra);
        self.conv.forward(x)
    }
}

pub fn load_sconv(
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    stride: usize,
    groups: usize,
    device: &burn::tensor::Device,
) -> SConv1d {
    let layer = SConv1d::new(in_ch, out_ch, kernel_size, stride, groups, device);
    let mut record = layer.clone();
    record.conv.weight = Param::from_tensor(weight);
    record.conv.bias = bias.map(Param::from_tensor);
    record
}