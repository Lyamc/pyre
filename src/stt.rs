//! Speech-to-text model kind, backed by the user's whisburn crates (burn 0.21) through a thin adapter.
//!
//! whisburn keeps its own Burn-native bundles (converted `model.mpk` + tokenizer + config) under
//! `WHISBURN_MODELS_DIR/<name>`. pyre points that at `<pyre store>/stt/whisburn` unless the user
//! already set it, and lets whisburn download + convert on first use. Refs accepted:
//! a whisburn registry name (`tiny_en`), its HF repo id (`openai/whisper-tiny.en`), or a path to
//! an existing whisburn bundle directory.
use anyhow::{anyhow, bail, Result};
use axum::{extract::Multipart, http::StatusCode, response::{IntoResponse, Response}, Json};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(feature = "stt")]
use whisburn_engine::{model::registry::MODEL_REGISTRY, runtime::InferenceRuntime};

pub struct SttModel {
    pub name: String,
    pub dir: PathBuf,
    backend_desc: String,
    #[cfg(feature = "stt")]
    rt: Mutex<InferenceRuntime>,
    busy: AtomicBool,
    queued: AtomicU64,
}

/// Map a user ref to (whisburn model name, explicit bundle dir if a path was given).
#[cfg(feature = "stt")]
fn map_ref(r: &str) -> Result<(String, Option<PathBuf>)> {
    let p = Path::new(r);
    if p.is_dir() {
        let name = p.file_name().and_then(|s| s.to_str()).ok_or_else(|| anyhow!("bad path {r}"))?.to_string();
        return Ok((name, Some(p.to_path_buf())));
    }
    let r = r.strip_prefix("whisburn:").unwrap_or(r);
    for m in MODEL_REGISTRY {
        if m.name.eq_ignore_ascii_case(r) || m.hf_id.eq_ignore_ascii_case(r) {
            if !matches!(m.category, whisburn_engine::model::registry::ModelCategory::Asr) {
                bail!("whisburn model '{}' is not a speech-recognition model", m.name);
            }
            return Ok((m.name.to_string(), None));
        }
    }
    let names: Vec<_> = MODEL_REGISTRY.iter().filter(|m| matches!(m.category, whisburn_engine::model::registry::ModelCategory::Asr)).map(|m| m.name).collect();
    bail!("unknown speech-to-text model '{r}'. whisburn supports: {}", names.join(", "))
}

impl SttModel {
    #[cfg(not(feature = "stt"))]
    pub fn load(_r: &str, _backend: &str, _store_root: &Path, _offline: bool) -> Result<Self> {
        bail!("pyre was built without the `stt` feature (whisburn)")
    }

    #[cfg(feature = "stt")]
    pub fn load(r: &str, backend: &str, store_root: &Path, offline: bool) -> Result<Self> {
        let (name, explicit) = map_ref(r)?;
        // whisburn reads WHISBURN_MODELS_DIR on every lookup; it is process-global, so all
        // loaded STT models share one bundle directory (documented in the README).
        let models_dir = match &explicit {
            Some(d) => d.parent().unwrap().to_path_buf(),
            None => std::env::var_os("WHISBURN_MODELS_DIR").map(PathBuf::from).unwrap_or_else(|| store_root.join("stt").join("whisburn")),
        };
        std::fs::create_dir_all(&models_dir)?;
        // SAFETY-ish: edition 2021; set before whisburn reads it; pyre loads models one at a time.
        std::env::set_var("WHISBURN_MODELS_DIR", &models_dir);
        let dir = whisburn_core::resolve_model_dir(&name);
        let has_bundle = dir.join("model.mpk").is_file() || dir.join("model.mpk.gz").is_file() || dir.join(format!("{name}.mpk")).is_file();
        if !has_bundle {
            if offline || explicit.is_some() { bail!("no whisburn bundle for '{name}' in {} (offline or explicit path)", dir.display()); }
            tracing::info!("whisburn: downloading + converting '{name}' into {}", models_dir.display());
            whisburn_models::download_model(&name, &whisburn_models::DownloadOptions::default())
                .map_err(|e| anyhow!("whisburn download/convert failed: {e:#}"))?;
        }
        // whisburn runs on wgpu (any GPU incl. NVIDIA via Vulkan/DX12) or ndarray CPU. It has no
        // CUDA runtime, so pyre's `cuda` maps to whisburn's wgpu.
        let dev = match backend { "cpu" | "ndarray" | "flex" => Some("cpu".to_string()), _ => None };
        let rt = InferenceRuntime::new(dev, false, false);
        let t = std::time::Instant::now();
        rt.warm_model(&name).map_err(|e| anyhow!("whisburn failed to load '{name}': {e}"))?;
        let backend_desc = rt.backend_name();
        tracing::info!("whisburn '{name}' ready on {backend_desc} in {:.2}s", t.elapsed().as_secs_f64());
        Ok(Self { name, dir, backend_desc, rt: Mutex::new(rt), busy: AtomicBool::new(false), queued: AtomicU64::new(0) })
    }
    pub fn describe(&self) -> String { format!("whisburn {} on {}", self.name, self.backend_desc) }
    pub fn busy(&self) -> bool { self.busy.load(Ordering::SeqCst) }
    pub fn queued(&self) -> u64 { self.queued.load(Ordering::SeqCst) }
}

struct Req { file: Vec<u8>, filename: String, model: Option<String>, format: String, language: Option<String> }

async fn parse(mut mp: Multipart) -> Result<Req> {
    let mut q = Req { file: vec![], filename: "audio.wav".into(), model: None, format: "json".into(), language: None };
    while let Some(f) = mp.next_field().await? {
        match f.name().unwrap_or("") {
            "file" => { if let Some(n) = f.file_name() { q.filename = n.to_string(); } q.file = f.bytes().await?.to_vec(); }
            "model" => q.model = Some(f.text().await?),
            "response_format" => q.format = f.text().await?,
            "language" => { let l = f.text().await?; if !l.is_empty() { q.language = Some(l); } }
            _ => { let _ = f.bytes().await; } // prompt, temperature, timestamp_granularities: accepted, ignored
        }
    }
    if q.file.is_empty() { bail!("missing 'file' field"); }
    Ok(q)
}

pub async fn handle(app: Arc<crate::registry::App>, mp: Multipart, translate: bool) -> Response {
    use crate::server::err;
    let q = match parse(mp).await { Ok(q) => q, Err(e) => return err(StatusCode::BAD_REQUEST, format!("{e:#}")) };
    let m = match app.find(crate::registry::Kind::Stt, q.model.as_deref()) { Ok(m) => m, Err(r) => return r };
    let stt = match &m.inner { crate::registry::Inner::Stt(s) => s.clone(), _ => return err(StatusCode::INTERNAL_SERVER_ERROR, "registry mismatch") };
    if !["json", "text", "srt", "vtt", "verbose_json"].contains(&q.format.as_str()) {
        return err(StatusCode::BAD_REQUEST, format!("unsupported response_format '{}'", q.format));
    }
    let id = m.id.clone();
    stt.queued.fetch_add(1, Ordering::SeqCst);
    let res = tokio::task::spawn_blocking(move || run(&stt, &q, translate)).await;
    match res {
        Ok(Ok((r, dur, secs, fmt))) => {
            tracing::info!("stt {id}: {dur:.2}s audio in {secs:.2}s ({:.1}x realtime)", dur / secs.max(1e-6));
            render(&r, &fmt, dur, translate)
        }
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[cfg(not(feature = "stt"))]
fn run(_: &SttModel, _: &Req, _: bool) -> Result<(Transcript, f64, f64, String)> { bail!("built without stt") }
#[cfg(not(feature = "stt"))]
struct Transcript;
#[cfg(not(feature = "stt"))]
fn render(_: &Transcript, _: &str, _: f64, _: bool) -> Response { crate::server::err(StatusCode::NOT_IMPLEMENTED, "built without stt") }

#[cfg(feature = "stt")]
type Transcript = whisburn_core::TranscriptResult;

#[cfg(feature = "stt")]
fn run(stt: &SttModel, q: &Req, translate: bool) -> Result<(Transcript, f64, f64, String)> {
    let rt = stt.rt.lock().unwrap();
    stt.queued.fetch_sub(1, Ordering::SeqCst);
    stt.busy.store(true, Ordering::SeqCst);
    struct Clear<'a>(&'a AtomicBool);
    impl Drop for Clear<'_> { fn drop(&mut self) { self.0.store(false, Ordering::SeqCst); } }
    let _c = Clear(&stt.busy);
    let ext = Path::new(&q.filename).extension().and_then(|e| e.to_str()).unwrap_or("wav").to_ascii_lowercase();
    let wav = whisburn_audio::decode_bytes_to_mono_pcm(&q.file, &ext).map_err(|e| anyhow!("could not decode audio ({ext}): {e}"))?;
    let wav = wav.ensure_sample_rate(16_000).map_err(|e| anyhow!("resample: {e}"))?;
    let dur = wav.duration_secs();
    let mut o = whisburn_core::TaskOptions::default();
    o.task = if translate { whisburn_core::SpeechTask::Translate } else { whisburn_core::SpeechTask::Transcribe };
    if let Some(l) = &q.language { o.language = whisburn_core::LanguageCode::new(l.clone()); }
    // whisburn only walks past the first 30 s window / sentence in timestamp mode, so always use it.
    o.include_timestamps = true;
    let t = std::time::Instant::now();
    let r = rt.transcribe_waveform(&stt.name, wav.samples, wav.sample_rate, &o).map_err(|e| anyhow!("whisburn: {e}"))?;
    Ok((r, dur, t.elapsed().as_secs_f64(), q.format.clone()))
}

#[cfg(feature = "stt")]
fn render(r: &Transcript, fmt: &str, dur: f64, translate: bool) -> Response {
    let text = r.text.trim().to_string();
    match fmt {
        "text" => (StatusCode::OK, [("content-type", "text/plain; charset=utf-8")], text + "\n").into_response(),
        "srt" => (StatusCode::OK, [("content-type", "application/x-subrip")], whisburn_audio::format_srt(&r.segments, false)).into_response(),
        "vtt" => (StatusCode::OK, [("content-type", "text/vtt")], whisburn_audio::format_vtt(&r.segments, false)).into_response(),
        "verbose_json" => Json(json!({
            "task": if translate { "translate" } else { "transcribe" },
            "language": r.language, "duration": dur, "text": text,
            "segments": r.segments.iter().enumerate().map(|(i, s)| json!({"id": i, "start": s.start, "end": s.end, "text": s.text.trim()})).collect::<Vec<_>>(),
        })).into_response(),
        _ => Json(json!({"text": text})).into_response(),
    }
}
