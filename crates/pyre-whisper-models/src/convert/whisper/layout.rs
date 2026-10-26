use crate::convert::dtype::transpose_linear_weight;
use crate::convert::npy::NpyDump;

/// Burn 0.21 `Linear` stores weights as `[d_input, d_output]`; HF uses `[out, in]`.
pub fn burn_linear_layout(rel: &str, data: &[f32], shape: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let is_linear_weight = rel.ends_with("/weight.npy")
        && !rel.contains("token_embedding")
        && !rel.contains("positional_embedding");

    if shape.len() == 2 && is_linear_weight {
        transpose_linear_weight(data, shape)
    } else {
        (data.to_vec(), shape.to_vec())
    }
}

pub fn write_attn_heads(
    dump: &mut NpyDump,
    side: &str,
    layers: usize,
    n_head: usize,
) -> anyhow::Result<()> {
    let head_val = n_head as f32;
    (0..layers).try_for_each(|i| {
        dump.write_scalar(&format!("{side}/block_{i}/attn/n_head.npy"), head_val)?;
        dump.write_scalar(&format!("{side}/block_{i}/cross_attn/n_head.npy"), head_val)
    })
}