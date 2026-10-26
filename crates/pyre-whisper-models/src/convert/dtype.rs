pub fn bytes_to_f32(data: &[u8], dtype: safetensors::Dtype) -> anyhow::Result<Vec<f32>> {
    Ok(match dtype {
        safetensors::Dtype::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        safetensors::Dtype::F16 => data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => anyhow::bail!("unsupported tensor dtype: {other:?}"),
    })
}

/// Transpose HF linear weights `[out, in]` → Burn 0.21 `[in, out]`.
///
/// Data must be permuted, not only the shape metadata. A shape-only swap
/// leaves square attention matrices unchanged but scrambles MLP `fc1`/`fc2`
/// (and any other rectangular linear), which is enough to make medium+
/// Whisper decode into comma noise.
pub fn transpose_linear_weight(data: &[f32], shape: &[usize]) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(shape.len(), 2, "linear weights are rank-2");
    let out = shape[0];
    let inp = shape[1];
    assert_eq!(data.len(), out * inp, "linear weight length != out*in");
    let mut transposed = vec![0.0; inp * out];
    for o in 0..out {
        for i in 0..inp {
            transposed[i * out + o] = data[o * inp + i];
        }
    }
    (transposed, vec![inp, out])
}

#[cfg(test)]
mod tests {
    use super::transpose_linear_weight;

    #[test]
    fn transposes_rectangular_hf_linear() {
        // HF [out=2, in=3] row-major:
        // [[0, 1, 2],
        //  [3, 4, 5]]
        let hf = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let (data, shape) = transpose_linear_weight(&hf, &[2, 3]);
        assert_eq!(shape, vec![3, 2]);
        // Burn [in=3, out=2]:
        // [[0, 3],
        //  [1, 4],
        //  [2, 5]]
        assert_eq!(data, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn shape_only_swap_is_not_a_transpose() {
        let hf = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let (data, _) = transpose_linear_weight(&hf, &[2, 3]);
        assert_ne!(
            data, hf,
            "must permute storage; copying with a swapped shape scrambles MLP layers"
        );
    }
}