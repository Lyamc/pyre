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
            Inner::Llm(l) => json!({"n_ctx": l.n_ctx, "backend": l.backend, "dtype": l.dtype, "path": self.path}),
            Inner::Stt(s) => json!({"backend": self.backend, "path": self.path, "engine": s.describe()}),
            Inner::Placeholder(r) => json!({"backend": self.backend, "path": self.path, "status": "not-runnable", "reason": r}),
        }
    }
    pub fn busy(&self) -> bool {
        match &self.inner { Inner::Llm(l) => l.engine.busy.load(Ordering::SeqCst), Inner::Stt(s) => s.busy(), _ => false }
    }
    pub fn queued(&self) -> u64 {
        match &self.inner { Inner::Llm(l) => l.engine.queued.load(Ordering::SeqCst), Inner::Stt(s) => s.queued(), _ => 0 }
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

/// Resolve + load a model (blocking; run on a blocking thread from async code).
pub fn load(storage: &Ctx, d: &LoadDefaults, q: LoadReq) -> Result<Loaded> {
    let backend = q.backend.clone().unwrap_or_else(|| d.backend.clone());
    if q.kind == Kind::Stt {
        // STT bundles are whisburn-format (converted .mpk), resolved by the whisburn adapter.
        let backend = if matches!(backend.as_str(), "cpu" | "flex" | "ndarray") { "cpu".to_string() } else { "wgpu".to_string() };
        let s = crate::stt::SttModel::load(q.r, &backend, &storage.store.root, storage.offline).context("loading speech-to-text model")?;
        let id = q.alias.clone().unwrap_or_else(|| q.r.to_string());
        return Ok(Loaded { id, kind: q.kind, backend, path: s.dir.display().to_string(), created: now(), inner: Inner::Stt(Arc::new(s)) });
    }
    let (dir, src) = storage.resolver().resolve(q.r)?;
    tracing::info!("resolved {} -> {} (via {src:?})", q.r, dir.display());
    let id = q.alias.clone().unwrap_or_else(|| q.r.to_string());
    let inner = match q.kind {
        Kind::Llm => Inner::Llm(Arc::new(crate::load_llm(&dir, &id, &backend, q.dtype.as_deref().unwrap_or(&d.dtype), q.ctx_size.unwrap_or(d.ctx_size), d.n_predict)?)),
        Kind::Stt => unreachable!(),
        Kind::Image => Inner::Placeholder("image generation (Stable Diffusion on Burn) is not implemented yet; see README roadmap".into()),
        Kind::Tts => Inner::Placeholder("text-to-speech is not implemented yet; see README roadmap".into()),
        Kind::Unknown => anyhow::bail!("unknown model kind for {}", q.r),
    };
    Ok(Loaded { id, kind: q.kind, backend, path: dir.display().to_string(), created: now(), inner })
}
