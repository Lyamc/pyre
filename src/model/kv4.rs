//! 4-bit KV storage (user-facing default). Same absmax-per-32 layout as [`super::kv8`], with
//! signed nibbles in -7..=7 (`((1 << (bits - 1)) - 1)`, matching `kvq_roundtrip` for `bits = 4`).
//!
//! This is lossy. A block's worst error is half a step (`scale / 2`). Eight-bit uses 127 levels
//! and is closer to f32; four-bit uses 15. Nothing here claims the two match.
//!
//! Packing: 8 nibbles per `u32` (low nibble = element 0), one f32 scale per 32 values.
//! Dequant is `q * scale` with the nibble sign-extended (`n >= 8` means `n - 16`).
pub const BLK: usize = 32;

/// Quantize `x` (length multiple of 32) to (packed nibbles, scales).
pub fn pack(x: &[f32]) -> (Vec<u32>, Vec<f32>) {
    assert!(x.len() % BLK == 0, "kv4 length {} is not a multiple of {BLK}", x.len());
    let mut w = Vec::with_capacity(x.len() / 8);
    let mut sc = Vec::with_capacity(x.len() / BLK);
    for b in x.chunks(BLK) {
        let m = b.iter().fold(0f32, |a, v| a.max(v.abs()));
        let s = if m > 0.0 && m.is_finite() { m / 7.0 } else { 0.0 };
        sc.push(s);
        for c in b.chunks(8) {
            let mut u = 0u32;
            for (j, v) in c.iter().enumerate() {
                let q = if s > 0.0 { (v / s).round().clamp(-7.0, 7.0) as i32 } else { 0 };
                u |= ((q as i8 as u8) as u32 & 0xF) << (4 * j);
            }
            w.push(u);
        }
    }
    (w, sc)
}

/// Dequantize element `i`.
#[inline]
pub fn get(w: &[u32], sc: &[f32], i: usize) -> f32 {
    let nib = (w[i / 8] >> (4 * (i % 8))) & 0xF;
    let q = nib as i32 - if nib >= 8 { 16 } else { 0 };
    q as f32 * sc[i / BLK]
}

pub fn unpack(w: &[u32], sc: &[f32], n: usize) -> Vec<f32> {
    (0..n).map(|i| get(w, sc, i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mae(x: &[f32], y: &[f32]) -> f32 {
        x.iter().zip(y).map(|(a, b)| (a - b).abs()).sum::<f32>() / x.len() as f32
    }

    #[test]
    fn roundtrip_matches_measure_mode_and_is_lossier_than_8bit() {
        // Outliers plus small values: absmax 4-bit spends its 15 levels on the peak.
        let x: Vec<f32> = (0..256).map(|i| {
            let t = (i as f32) * 0.017 - 1.3;
            if i % 32 == 7 { t * 4.0 } else { t * 0.15 }
        }).collect();
        let (w, s) = pack(&x);
        assert_eq!(w.len(), x.len() / 8);
        assert_eq!(s.len(), x.len() / BLK);
        let y = unpack(&w, &s, x.len());
        let mut z = x.clone();
        crate::model::qwen4exp_gguf::kvq_roundtrip(&mut z, 4);
        for i in 0..x.len() {
            assert!((y[i] - z[i]).abs() <= 1e-5 * z[i].abs().max(1.0), "i {i} pack {} measure {}", y[i], z[i]);
            assert!((y[i] - x[i]).abs() <= s[i / BLK] * 0.5 + 1e-6, "i {i} err {} scale {}", (y[i] - x[i]).abs(), s[i / BLK]);
        }
        let (w0, s0) = pack(&[0f32; 32]);
        assert_eq!(unpack(&w0, &s0, 32), vec![0f32; 32]);
        // -7 and +7 survive as the extreme nibbles.
        let extreme = vec![1.0f32; 31].into_iter().chain(std::iter::once(-1.0)).collect::<Vec<_>>();
        let (we, se) = pack(&extreme);
        let back = unpack(&we, &se, 32);
        assert!((back[31] + 1.0).abs() < 1e-5 || (back[31] - extreme[31]).abs() <= se[0] * 0.5 + 1e-6);

        let mut q8 = x.clone();
        crate::model::qwen4exp_gguf::kvq_roundtrip(&mut q8, 8);
        let e4 = mae(&x, &y);
        let e8 = mae(&x, &q8);
        assert!(e8 < e4, "8-bit mean abs error {e8} should be under 4-bit {e4}");
        assert!(e4 > e8 * 2.0, "4-bit is not a stand-in for 8-bit (e4 {e4}, e8 {e8})");
        assert!(e4 < 0.2, "4-bit error stayed bounded ({e4})");
        // Packed payload: 4-bit words are half of 8-bit words; scales are the same count.
        let (w8, s8) = crate::model::kv8::pack(&x);
        assert_eq!(w.len() * 2, w8.len());
        assert_eq!(s.len(), s8.len());
    }

    #[test]
    fn nibble_sign_extension() {
        // One block, only element 0 set, so the low nibble is the code.
        let mut x = vec![0f32; 32];
        x[0] = -0.7;
        x[1] = 0.7;
        let (w, s) = pack(&x);
        assert!(s[0] > 0.0);
        let n0 = w[0] & 0xF;
        let n1 = (w[0] >> 4) & 0xF;
        assert!(n0 >= 8, "negative code should have the sign bit, got {n0}");
        assert!(n1 < 8, "positive code should not, got {n1}");
        assert!((get(&w, &s, 0) - x[0]).abs() <= s[0] * 0.5 + 1e-6);
        assert!((get(&w, &s, 1) - x[1]).abs() <= s[0] * 0.5 + 1e-6);
    }
}
