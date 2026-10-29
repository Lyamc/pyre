//! Multi-model registry. Each loaded model has a kind (text-generation, image-generation,
//! speech-to-text, text-to-speech) and its own worker queue. Endpoints look models up by id
//! and kind; asking a model of the wrong kind is a clear 400.
use crate::server::{err, Llm};
use crate::storage::Ctx;
use anyhow::{Context, Result};
use axum::http::StatusCode;
use axum::response::Response;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use std::sync::{Arc, RwLock};

pub use crate::store::Kind;

pub enum Inner {
    Llm(Arc<Llm>),
    Stt(Arc<crate::stt::SttModel>),
    Image(Arc<crate::image::ImageModel>),
    Tts(Arc<crate::tts::TtsModel>),
    /// Registered but not runnable in this build (e.g. image/TTS before their backends land).
    Placeholder(String),
}

pub struct Loaded {
    pub id: String,
    pub kind: Kind,
    pub backend: String,
    pub path: String,
    pub created: u64,
    pub inner: Inner,
}

impl Loaded {
    pub fn meta(&self) -> Value {
        match &self.inner {
            Inner::Llm(l) => json!({"n_ctx": l.n_ctx, "context_length": l.n_ctx, "max_model_len": l.n_ctx, "backend": l.backend, "dtype": l.dtype, "path": self.path}),
            Inner::Stt(s) => json!({"backend": self.backend, "path": self.path, "engine": s.describe()}),
            Inner::Image(i) => json!({"backend": self.backend, "path": self.path, "engine": i.describe(), "image": i.info()}),
            Inner::Tts(t) => json!({"backend": self.backend, "path": self.path, "engine": t.describe()}),
            Inner::Placeholder(r) => json!({"backend": self.backend, "path": self.path, "status": "not-runnable", "reason": r}),
        }
    }
    pub fn busy(&self) -> bool {
        match &self.inner { Inner::Llm(l) => l.engine.busy.load(Ordering::SeqCst), Inner::Stt(s) => s.busy(), Inner::Image(i) => i.busy.load(Ordering::SeqCst), Inner::Tts(t) => t.busy.load(Ordering::SeqCst), _ => false }
    }
    pub fn queued(&self) -> u64 {
        match &self.inner { Inner::Llm(l) => l.engine.queued.load(Ordering::SeqCst), Inner::Stt(s) => s.queued(), Inner::Image(i) => i.queued.load(Ordering::SeqCst), Inner::Tts(t) => t.queued.load(Ordering::SeqCst), _ => 0 }
    }
}

#[derive(Clone, Debug)]
pub struct LoadDefaults { pub backend: String, pub dtype: String, pub ctx_size: usize, pub n_predict: usize }

pub struct App {
    pub models: RwLock<Vec<Arc<Loaded>>>,
    pub storage: Ctx,
    pub defaults: LoadDefaults,
    pub api_key: Option<String>,
    pub started: std::time::Instant,
}

fn now() -> u64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() }

impl App {
    pub fn loaded(&self) -> Vec<Arc<Loaded>> { self.models.read().unwrap().clone() }

    /// Find a model of `kind`. A name that matches a model of another kind is an error; an
    /// unknown name falls back to the first model of the right kind (OpenAI clients often send
    /// placeholder names), like llama-server.
    pub fn find(&self, kind: Kind, name: Option<&str>) -> Result<Arc<Loaded>, Response> {
        let all = self.loaded();
        if let Some(n) = name {
            if let Some(m) = all.iter().find(|m| m.id == n) {
                if m.kind != kind {
                    return Err(err(StatusCode::BAD_REQUEST, format!("model '{n}' is a {} model ({}); this endpoint needs a {} model", m.kind.as_str(), m.kind.task(), kind.task())));
                }
                return Ok(m.clone());
            }
        }
        all.iter().find(|m| m.kind == kind).cloned().ok_or_else(|| {
            let code = if matches!(kind, Kind::Image | Kind::Tts) { StatusCode::NOT_IMPLEMENTED } else { StatusCode::BAD_REQUEST };
            err(code, format!("no {} model is loaded; start pyre with {} or load one via POST /api/models/load", kind.task(),
                match kind { Kind::Llm => "-m <model>", Kind::Image => "--image-model <model>", Kind::Stt => "--stt-model <model>", Kind::Tts => "--tts-model <model>", Kind::Unknown => "--load kind=ref" }))
        })
    }

    pub fn llm(&self, name: Option<&str>) -> Result<Arc<Llm>, Response> {
        let m = self.find(Kind::Llm, name)?;
        match &m.inner { Inner::Llm(l) => Ok(l.clone()), _ => Err(err(StatusCode::INTERNAL_SERVER_ERROR, "registry mismatch")) }
    }

    pub fn insert(&self, m: Loaded) -> Result<()> {
        let mut g = self.models.write().unwrap();
        if g.iter().any(|x| x.id == m.id) { anyhow::bail!("a model with id '{}' is already loaded", m.id); }
        g.push(Arc::new(m));
        Ok(())
    }

    /// Unload: dropping the last Arc closes the worker queue; its thread exits and frees the model.
    pub fn unload(&self, id: &str) -> bool {
        let mut g = self.models.write().unwrap();
        let n = g.len();
        g.retain(|m| m.id != id);
        n != g.len()
    }
}

pub struct LoadReq<'a> { pub kind: Kind, pub r: &'a str, pub alias: Option<String>, pub backend: Option<String>, pub dtype: Option<String>, pub ctx_size: Option<usize> }

/// Resolve + load a model (blocking; run on a blocking thread from async code). Refuses loads that
/// would breach the VRAM reserve (estimate from the weight files first, then the measured result).
pub fn load(storage: &Ctx, d: &LoadDefaults, q: LoadReq) -> Result<Loaded> {
    let gpu = !matches!(q.backend.as_deref().unwrap_or(&d.backend), "cpu" | "flex" | "ndarray");
    let m = load_inner(storage, d, q)?;
    if gpu && crate::limits::headroom() == Some(0) {
        let v = crate::limits::vram();
        drop(m);
        anyhow::bail!("loading this model left only {} MiB of VRAM free, below the {} MiB reserve for other apps; it was unloaded (lower --vram-reserve / PYRE_VRAM_RESERVE, or unload another model)",
            v.map(|v| v.free >> 20).unwrap_or(0), crate::limits::reserve() >> 20);
    }
    Ok(m)
}

/// Sum of weight-file sizes under dir (a rough lower bound on the model's VRAM).
fn weight_bytes(dir: &std::path::Path) -> u64 {
    let mut n = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(p) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&p) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() { stack.push(p); continue; }
            if matches!(p.extension().and_then(|e| e.to_str()), Some("safetensors" | "gguf" | "mpk" | "bin")) { n += e.metadata().map(|m| m.len()).unwrap_or(0); }
        }
    }
    n
}

fn load_inner(storage: &Ctx, d: &LoadDefaults, q: LoadReq) -> Result<Loaded> {
    let backend = q.backend.clone().unwrap_or_else(|| d.backend.clone());
    if q.kind == Kind::Stt {
        // STT bundles are whisburn-format (converted .bpk), resolved by the whisburn adapter. Runs on the
        // requested backend (cuda/wgpu/vulkan), so it shares the LLM's device and kernel caches.
        let backend = if matches!(backend.as_str(), "cpu" | "flex" | "ndarray") { "cpu".to_string() } else { backend };
        let s = crate::stt::SttModel::load(q.r, &backend, &storage.store.root, storage.offline).context("loading speech-to-text model")?;
        let s = Arc::new(s);
        s.warm_in_background();
        let id = q.alias.clone().unwrap_or_else(|| q.r.to_string());
        return Ok(Loaded { id, kind: q.kind, backend, path: s.dir.display().to_string(), created: now(), inner: Inner::Stt(s) });
    }
    let (dir, src) = storage.resolver().resolve(q.r)?;
    tracing::info!("resolved {} -> {} (via {src:?})", q.r, dir.display());
    if !matches!(backend.as_str(), "cpu" | "flex" | "ndarray") && !crate::qn_find::is_qwen4exp(&dir) { crate::limits::check_alloc(weight_bytes(&dir), &format!("loading '{}'", q.r))?; }
    let id = q.alias.clone().unwrap_or_else(|| q.r.to_string());
    let inner = match q.kind {
        Kind::Llm => Inner::Llm(Arc::new(crate::load_llm(&dir, &id, &backend, q.dtype.as_deref().unwrap_or(&d.dtype), q.ctx_size.unwrap_or(d.ctx_size), d.n_predict)?)),
        Kind::Stt => unreachable!(),
        Kind::Image => Inner::Image(Arc::new(crate::image::ImageModel::load(&dir, &backend).context("loading image model")?)),
        Kind::Tts => Inner::Tts(Arc::new(crate::tts::TtsModel::load(&dir, &backend).context("loading text-to-speech model")?)),
        Kind::Unknown => anyhow::bail!("unknown model kind for {}", q.r),
    };
    Ok(Loaded { id, kind: q.kind, backend, path: dir.display().to_string(), created: now(), inner })
}
