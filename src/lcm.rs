//! LCM-LoRA for SD 1.5: merge the kohya-format LoRA (latent-consistency/lcm-lora-sdv1-5) into a UNet once,
//! natively, and cache the merged UNet next to the LoRA (`merged-<unet folder>/unet/...fp16.safetensors`).
//! W' = W + (alpha / rank) * up @ down for every UNet weight the LoRA covers (linear, 1x1 and 3x3 convs);
//! tensors whose shape does not match (e.g. conv_in of the 9-channel inpainting UNet) are kept as they are.
use anyhow::{anyhow, bail, Context, Result};
use rayon::prelude::*;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub fn lora_file(store_root: &Path) -> PathBuf { std::env::var("PYRE_SD_LCM_LORA").map(PathBuf::from).unwrap_or_else(|_| store_root.join("cabin").join("lcm").join("lcm-lora-sdv1-5.safetensors")) }

fn unet_file(model_dir: &Path) -> Option<PathBuf> {
    ["diffusion_pytorch_model.fp16.safetensors", "diffusion_pytorch_model.safetensors"].iter().map(|f| model_dir.join("unet").join(f)).find(|p| p.is_file())
}
fn to_f32(t: &TensorView) -> Result<Vec<f32>> {
    Ok(match t.dtype() {
        Dtype::F32 => t.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        Dtype::F16 => t.data().chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        Dtype::BF16 => t.data().chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        d => bail!("unsupported dtype {d:?}"),
    })
}
fn from_f32(v: &[f32], d: Dtype) -> Vec<u8> {
    match d { Dtype::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(), Dtype::BF16 => v.iter().flat_map(|x| half::bf16::from_f32(*x).to_le_bytes()).collect(), _ => v.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect() }
}

/// Directory of an SD model folder whose UNet has the LCM-LoRA merged in (built on first use, then cached).
/// `model_dir` holds `unet/`; the result holds `unet/` with the same config.
pub fn merged(lora: &Path, model_dir: &Path) -> Result<PathBuf> {
    let name = model_dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "model".into());
    let out = lora.parent().unwrap_or(Path::new(".")).join(format!("merged-{name}"));
    let dst = out.join("unet").join("diffusion_pytorch_model.fp16.safetensors");
    if dst.is_file() { return Ok(out); }
    let src = unet_file(model_dir).ok_or_else(|| anyhow!("no UNet weights in {}", model_dir.display()))?;
    let t0 = std::time::Instant::now();
    let lb = std::fs::read(lora).with_context(|| format!("reading {}", lora.display()))?;
    let ls = SafeTensors::deserialize(&lb)?;
    let f = std::fs::File::open(&src)?;
    let mm = unsafe { memmap2::Mmap::map(&f)? };
    let bs = SafeTensors::deserialize(&mm)?;
    let names: Vec<String> = bs.names().into_iter().map(|s| s.to_string()).collect();
    // kohya name -> base key
    let mut hits = 0usize; let mut skipped = Vec::new();
    let merged: Vec<(String, Option<(Vec<usize>, Vec<u8>)>)> = names.par_iter().map(|k| -> Result<(String, Option<(Vec<usize>, Vec<u8>)>)> {
        let Some(base) = k.strip_suffix(".weight") else { return Ok((k.clone(), None)) };
        let kk = format!("lora_unet_{}", base.replace('.', "_"));
        let (Ok(up), Ok(down)) = (ls.tensor(&format!("{kk}.lora_up.weight")), ls.tensor(&format!("{kk}.lora_down.weight"))) else { return Ok((k.clone(), None)) };
        let w = bs.tensor(k)?;
        let ws = w.shape().to_vec(); let us = up.shape(); let ds = down.shape();
        let r = ds[0]; let o = us[0]; let inner: usize = ds[1..].iter().product();
        if ws[0] != o || ws.iter().product::<usize>() != o * inner { return Ok((format!("!{k}"), None)); }
        let alpha = ls.tensor(&format!("{kk}.alpha")).ok().and_then(|a| to_f32(&a).ok()).and_then(|v| v.first().copied()).unwrap_or(r as f32);
        let (u, d, mut wv) = (to_f32(&up)?, to_f32(&down)?, to_f32(&w)?);
        let sc = alpha / r as f32;
        for oi in 0..o { let row = &mut wv[oi * inner..(oi + 1) * inner]; for ri in 0..r { let c = u[oi * r + ri] * sc; if c == 0.0 { continue; } let dr = &d[ri * inner..(ri + 1) * inner]; for (x, y) in row.iter_mut().zip(dr) { *x += c * y; } } }
        Ok((k.clone(), Some((ws, from_f32(&wv, Dtype::F16)))))
    }).collect::<Result<Vec<_>>>()?;
    let mut owned: HashMap<String, (Vec<usize>, Vec<u8>)> = HashMap::new();
    for (k, v) in merged { if let Some(k2) = k.strip_prefix('!') { skipped.push(k2.to_string()); continue; } if let Some(v) = v { hits += 1; owned.insert(k, v); } }
    let mut conv: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut views: Vec<(String, TensorView)> = Vec::with_capacity(names.len());
    for k in &names {
        let t = bs.tensor(k)?;
        // always f16 on disk (the base SD 1.5 UNet ships f32; loading a 3.4 GB f32 UNet next to the resident
        // models breached the VRAM floor)
        let v = match owned.get(k) { Some((s, b)) => TensorView::new(Dtype::F16, s.clone(), b)?, None if t.dtype() == Dtype::F32 => { conv.push((k.clone(), t.shape().to_vec(), from_f32(&to_f32(&t)?, Dtype::F16))); continue; } None => t };
        views.push((k.clone(), v));
    }
    for (k, s, b) in &conv { views.push((k.clone(), TensorView::new(Dtype::F16, s.clone(), b)?)); }
    std::fs::create_dir_all(dst.parent().unwrap())?;
    let tmp = dst.with_extension("partial");
    safetensors::serialize_to_file(views, None, &tmp)?;
    std::fs::rename(&tmp, &dst)?;
    let cfg = model_dir.join("unet").join("config.json");
    if cfg.is_file() { std::fs::copy(&cfg, out.join("unet").join("config.json"))?; }
    let lora_keys = ls.names().iter().filter(|n| n.ends_with(".lora_up.weight")).count();
    tracing::info!("LCM-LoRA merged into {} in {:.1}s: {hits}/{lora_keys} LoRA layers applied, {} shape-mismatched kept ({:?})", src.display(), t0.elapsed().as_secs_f64(), skipped.len(), skipped);
    Ok(out)
}
