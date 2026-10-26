//! Backend/device selection. Burn 0.22 dispatches on `Device`, so one binary carries every
//! backend compiled in via cargo features and `--backend` picks at runtime.
#![allow(deprecated)]
use anyhow::{bail, Result};
use burn::tensor::{DType, Device, FloatDType};

pub fn select(name: &str, dtype: &str, model_dtype: Option<&str>) -> Result<(Device, FloatDType, &'static str)> {
    let (mut dev, label): (Device, &'static str) = match name {
        #[cfg(feature = "cuda")]
        "cuda" => (Device::cuda(0), "cuda"),
        #[cfg(feature = "wgpu")]
        "wgpu" => (Device::wgpu(Default::default()), "wgpu"),
        #[cfg(feature = "vulkan")]
        "vulkan" => (Device::vulkan(Default::default()), "vulkan"),
        #[cfg(feature = "flex")]
        "cpu" | "flex" => (Device::flex(), "cpu(flex)"),
        #[cfg(feature = "ndarray")]
        "ndarray" => (Device::ndarray(), "ndarray"),
        #[cfg(all(feature = "ndarray", not(feature = "flex")))]
        "cpu" => (Device::ndarray(), "ndarray"),
        other => bail!("backend '{other}' not available in this build (enabled: {})", enabled().join(", ")),
    };
    let want = match dtype {
        "f32" => FloatDType::F32,
        "f16" => FloatDType::F16,
        "bf16" => FloatDType::BF16,
        _ => {
            let gpu = matches!(label, "cuda" | "wgpu" | "vulkan");
            if !gpu { FloatDType::F32 }
            else if model_dtype == Some("bfloat16") && dev.supports_dtype(DType::BF16) { FloatDType::BF16 }
            else if dev.supports_dtype(DType::F16) { FloatDType::F16 }
            else { FloatDType::F32 }
        }
    };
    if !dev.supports_dtype(DType::from(want)) { bail!("{label} does not support {want:?}"); }
    dev.configure(want)?;
    Ok((dev, want, label))
}

fn enabled() -> Vec<&'static str> {
    let mut v = vec![];
    if cfg!(feature = "cuda") { v.push("cuda"); }
    if cfg!(feature = "wgpu") { v.push("wgpu"); }
    if cfg!(feature = "vulkan") { v.push("vulkan"); }
    if cfg!(feature = "flex") { v.push("cpu"); }
    if cfg!(feature = "ndarray") { v.push("ndarray"); }
    v
}

/// `--prompt`: chat-templated greedy generation + timing printout, then exit.
pub fn oneshot(m: &crate::model::llama::Llama, tok: &tokenizers::Tokenizer, t: &crate::template::ChatTemplate, eos: &[u32], p: &str) -> Result<()> {
    let text = t.render(&[serde_json::json!({"role": "user", "content": p})], true, None)?;
    let ids = tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!(e))?.get_ids().to_vec();
    let mut cache = m.new_cache();
    // warm-up (kernel compilation / autotune)
    let _ = m.logits_last(m.forward(&ids, 0, &mut cache));
    let _ = m.logits_last(m.forward(&[ids[0]], ids.len(), &mut cache));
    let t0 = std::time::Instant::now();
    let mut logits = m.logits_last(m.forward(&ids, 0, &mut cache));
    let pre = t0.elapsed().as_secs_f64();
    let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("top5 first-token logits: {:?}", top[..5].iter().map(|(i, l)| (tok.id_to_token(*i as u32).unwrap_or_default(), *i, *l)).collect::<Vec<_>>());
    let mut out = vec![];
    let t1 = std::time::Instant::now();
    for _ in 0..128 {
        let n = crate::sampler::argmax(&logits);
        if eos.contains(&n) { break; }
        out.push(n);
        logits = m.logits_last(m.forward(&[n], ids.len() + out.len() - 1, &mut cache));
    }
    let dec = t1.elapsed().as_secs_f64();
    println!("--- output ---\n{}\n--------------", tok.decode(&out, true).unwrap_or_default());
    println!("prefill: {} tok in {:.3}s = {:.1} tok/s | decode: {} tok in {:.3}s = {:.1} tok/s",
        ids.len(), pre, ids.len() as f64 / pre, out.len(), dec, out.len() as f64 / dec);
    println!("greedy ids: {:?}", &out[..out.len().min(20)]);
    Ok(())
}
