//! Tiny host Conv1d / LSTM used by Silero and TEN VAD (chunked, CPU).

pub fn conv1d(
    x: &[f32],
    c_in: usize,
    t: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    c_out: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
) -> (Vec<f32>, usize) {
    debug_assert_eq!(x.len(), c_in * t);
    debug_assert_eq!(weight.len(), c_out * c_in * kernel);
    let t_pad = t + 2 * pad;
    let t_out = if t_pad >= kernel {
        (t_pad - kernel) / stride.max(1) + 1
    } else {
        0
    };
    let mut y = vec![0f32; c_out * t_out];
    for oc in 0..c_out {
        let b = bias.map(|bb| bb[oc]).unwrap_or(0.0);
        for ot in 0..t_out {
            let start = ot * stride;
            let mut acc = b;
            for ic in 0..c_in {
                for k in 0..kernel {
                    let ti = start + k;
                    let src = if ti < pad {
                        None
                    } else {
                        let raw = ti - pad;
                        if raw < t {
                            Some(x[ic * t + raw])
                        } else {
                            None
                        }
                    };
                    if let Some(v) = src {
                        acc += v * weight[oc * c_in * kernel + ic * kernel + k];
                    }
                }
            }
            y[oc * t_out + ot] = acc;
        }
    }
    (y, t_out)
}

pub fn relu(x: &mut [f32]) {
    for v in x {
        *v = v.max(0.0);
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn reflect_pad_right(x: &[f32], pad: usize) -> Vec<f32> {
    if x.is_empty() || pad == 0 {
        return x.to_vec();
    }
    let mut out = Vec::with_capacity(x.len() + pad);
    out.extend_from_slice(x);
    for i in 0..pad {
        let src = x.len().saturating_sub(2).saturating_sub(i);
        out.push(x[src]);
    }
    out
}

/// PyTorch LSTMCell: gates = x @ W_ih^T + h @ W_hh^T + b_ih + b_hh (IFGO).
pub fn lstm_step(
    x: &[f32],
    h: &mut [f32],
    c: &mut [f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
) {
    let hidden = h.len();
    let input = x.len();
    debug_assert_eq!(w_ih.len(), 4 * hidden * input);
    debug_assert_eq!(w_hh.len(), 4 * hidden * hidden);
    let mut gates = vec![0f32; 4 * hidden];
    for o in 0..4 * hidden {
        let mut acc = b_ih[o] + b_hh[o];
        let row_ih = o * input;
        for i in 0..input {
            acc += x[i] * w_ih[row_ih + i];
        }
        let row_hh = o * hidden;
        for i in 0..hidden {
            acc += h[i] * w_hh[row_hh + i];
        }
        gates[o] = acc;
    }
    for j in 0..hidden {
        let i_g = sigmoid(gates[j]);
        let f_g = sigmoid(gates[hidden + j]);
        let g_g = gates[2 * hidden + j].tanh();
        let o_g = sigmoid(gates[3 * hidden + j]);
        c[j] = f_g * c[j] + i_g * g_g;
        h[j] = o_g * c[j].tanh();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv1d_identity_kernel() {
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![1.0];
        let (y, t) = conv1d(&x, 1, 3, &w, None, 1, 1, 1, 0);
        assert_eq!(t, 3);
        assert_eq!(y, x);
    }

    #[test]
    fn sigmoid_midpoint() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }
}
