//! ctx-kvquant option 1 (int50, opt-in PYRE_QN_KV8, off by default): 8-bit storage for older K/V positions.
//! Layout per attention layer: positions [0, q8n) as int8 packed 4 per u32 with one f32 absmax scale per 32 values
//! (`pack`), positions [q8n, len) stay f32 in the existing buffers at offset (pos - q8n). Host reference code here;
//! GPU kernels must match `unpack` bit-for-bit on the dequantized values (x = q * scale).
pub const BLK: usize = 32;

/// Quantize `x` (len multiple of 32) to (packed i8x4 words, scales).
pub fn pack(x: &[f32]) -> (Vec<u32>, Vec<f32>) {
    assert!(x.len() % BLK == 0);
    let mut w = Vec::with_capacity(x.len() / 4); let mut sc = Vec::with_capacity(x.len() / BLK);
    for b in x.chunks(BLK) {
        let m = b.iter().fold(0f32, |a, v| a.max(v.abs())); let s = if m > 0.0 && m.is_finite() { m / 127.0 } else { 0.0 }; sc.push(s);
        for c in b.chunks(4) { let mut u = 0u32; for (j, v) in c.iter().enumerate() { let q = if s > 0.0 { (v / s).round().clamp(-127.0, 127.0) as i32 } else { 0 }; u |= ((q as i8 as u8) as u32) << (8 * j); } w.push(u); }
    }
    (w, sc)
}
/// Dequantize element i.
#[inline] pub fn get(w: &[u32], sc: &[f32], i: usize) -> f32 { let b = ((w[i / 4] >> (8 * (i % 4))) & 0xFF) as u8 as i8; b as f32 * sc[i / BLK] }
pub fn unpack(w: &[u32], sc: &[f32], n: usize) -> Vec<f32> { (0..n).map(|i| get(w, sc, i)).collect() }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_matches_measure_mode() {
        let x: Vec<f32> = (0..256).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013).collect();
        let (w, s) = pack(&x); let y = unpack(&w, &s, x.len());
        let mut z = x.clone(); crate::model::qwen4exp_gguf::kvq_roundtrip(&mut z, 8);
        for i in 0..x.len() { assert!((y[i] - z[i]).abs() <= 1e-6 * z[i].abs().max(1.0), "i {i} {} {}", y[i], z[i]); assert!((y[i] - x[i]).abs() <= s[i / BLK] * 0.5 + 1e-7); }
        let (w0, s0) = pack(&[0f32; 32]); assert_eq!(unpack(&w0, &s0, 32), vec![0f32; 32]);
    }
}
