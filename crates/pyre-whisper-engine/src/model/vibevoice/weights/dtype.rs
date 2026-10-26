use safetensors::Dtype;

pub fn bytes_to_f32(data: &[u8], dtype: Dtype) -> Result<Vec<f32>, String> {
    Ok(match dtype {
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => data
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => return Err(format!("unsupported tensor dtype: {other:?}")),
    })
}

/// Transpose HF linear weights `[out, in]` → Burn 0.21 `[in, out]`.
pub fn transpose_linear_weight(data: &[f32], shape: &[usize]) -> (Vec<f32>, [usize; 2]) {
    let [rows, cols] = [shape[0], shape[1]];
    let transposed = (0..cols)
        .flat_map(|c| (0..rows).map(move |r| data[r * cols + c]))
        .collect();
    (transposed, [cols, rows])
}