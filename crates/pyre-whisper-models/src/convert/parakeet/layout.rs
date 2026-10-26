use crate::convert::dtype::transpose_linear_weight;

pub fn layout_tensor(rel: &str, data: &[f32], shape: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let is_conformer_ff = rel.contains("/ff1/") || rel.contains("/ff2/");
    let is_linear_weight = rel.ends_with("/weight.npy")
        && !is_conformer_ff
        && !rel.contains("/decoder/embedding/")
        && !rel.contains("/decoder/lstm/")
        && (rel.contains("/attn/")
            || rel.contains("pre_encode_out")
            || rel.contains("final_proj")
            || rel.contains("decoder/projector")
            || rel.contains("ctc_linear"));

    // HF CTC head is Conv1d(k=1): [vocab, hidden, 1] → Burn Linear [hidden, vocab].
    if rel.contains("ctc_linear") && shape.len() == 3 && shape[2] == 1 {
        let squeezed = [shape[0], shape[1]];
        return transpose_linear_weight(data, &squeezed);
    }
    if shape.len() == 2 && is_linear_weight {
        transpose_linear_weight(data, shape)
    } else {
        (data.to_vec(), shape.to_vec())
    }
}

pub fn split_layer_tail(rest: &str) -> Option<(usize, &str)> {
    let mut parts = rest.splitn(2, '.');
    let idx: usize = parts.next()?.parse().ok()?;
    let tail = parts.next()?;
    Some((idx, tail))
}