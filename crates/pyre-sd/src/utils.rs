use alloc::vec;
use burn::tensor::{kind::Numeric, Tensor};

pub(crate) fn pad_with_zeros<const D: usize, K>(
    tensor: Tensor<D, K>,
    dim: usize,
    left: usize,
    right: usize,
) -> Tensor<D, K>
where
    K: Numeric,
{
    if left == 0 && right == 0 {
        return tensor.clone();
    }
    assert!(
        dim < D,
        "dim must be less than the number of dimensions of the tensor"
    );
    // The padding has to be in the tensor's own dtype: `Tensor::zeros` follows
    // the device's default, and a VAE running in f32 on an f16 device would
    // otherwise ask `cat` to join f32 and f16. That is the encoder's downsample
    // path, so it is what made an f32 VAE unusable. Both sides are built before
    // `tensor` is moved into the concatenation.
    let pad = |n: usize| {
        let mut dims = tensor.dims().to_vec();
        dims[dim] = n;
        Tensor::<D, K>::zeros(dims, (&tensor.device(), tensor.dtype()))
    };
    let left_pad = (left > 0).then(|| pad(left));
    let right_pad = (right > 0).then(|| pad(right));
    match (left_pad, right_pad) {
        (None, None) => tensor,
        (None, Some(right)) => Tensor::cat(vec![tensor, right], dim),
        (Some(left), None) => Tensor::cat(vec![left, tensor], dim),
        (Some(left), Some(right)) => Tensor::cat(vec![left, tensor, right], dim),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::{Shape, TensorData, Tolerance};

    #[test]
    fn test_pad_with_zeros() {
        let device = Default::default();
        let tensor: Tensor<TestBackend, 3> = Tensor::from_data(
            TensorData::from([[[1.6585, 0.4320], [-0.8701, -0.4649]]]),
            &device,
        );

        let padded = pad_with_zeros(tensor, 0, 1, 2);

        assert_eq!(padded.shape(), Shape::from([4, 2, 2]));
        padded.into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [[0.0000, 0.0000], [0.0000, 0.0000]],
                [[1.6585, 0.4320], [-0.8701, -0.4649]],
                [[0.0000, 0.0000], [0.0000, 0.0000]],
                [[0.0000, 0.0000], [0.0000, 0.0000]],
            ]),
            Tolerance::rel_abs(1e-3, 1e-3),
        )
    }
}
