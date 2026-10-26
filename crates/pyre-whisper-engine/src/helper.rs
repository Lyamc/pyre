use burn::tensor::{
    activation::relu, kind::Numeric, Tensor,
};
use burn::tensor::cast::ToElement;

pub fn tensor_max_scalar<const D: usize>(x: Tensor<D>, max: f64) -> Tensor<D> {
    relu(x - max) + max
}

pub fn tensor_min_scalar<const D: usize>(x: Tensor<D>, min: f64) -> Tensor<D> {
    -tensor_max_scalar(-x, -min)
}

pub fn tensor_max<const D: usize>(x: Tensor<D>, max: Tensor<D>) -> Tensor<D> {
    relu(x - max.clone()) + max
}

pub fn tensor_min<const D: usize>(x: Tensor<D>, min: Tensor<D>) -> Tensor<D> {
    -tensor_max(-x, -min)
}

pub fn tensor_log10<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let ln10 = (10.0f64).ln();
    x.log() / ln10
}

pub fn all_zeros<const D: usize>(x: Tensor<D>) -> bool {
    x.abs().max().into_scalar::<f64>() == 0.0
}

pub fn _10pow<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let log10 = (10.0f64).ln();
    (x * log10).exp()
}

pub fn reverse<const D: usize, K: Numeric>(
    x: Tensor<D, K>,
    dim: usize,
) -> Tensor<D, K>
where
{
    let len = x.dims()[dim];
    let indices = Tensor::arange(0..len as i64, &x.device())
        .mul_scalar(-1)
        .add_scalar((len - 1) as i64);
    x.select(dim, indices)
}
