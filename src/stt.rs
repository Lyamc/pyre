//! Speech-to-text model kind. Backed by whisburn (see `stt_whisburn` when enabled).
use anyhow::{bail, Result};
use std::path::Path;

pub struct SttModel {}

impl SttModel {
    pub fn load(_dir: &Path, _backend: &str) -> Result<Self> { bail!("speech-to-text support is not built in yet") }
    pub fn describe(&self) -> String { String::new() }
    pub fn busy(&self) -> bool { false }
    pub fn queued(&self) -> u64 { 0 }
}

pub async fn handle(app: std::sync::Arc<crate::registry::App>, _mp: axum::extract::Multipart, _translate: bool) -> axum::response::Response {
    match app.find(crate::registry::Kind::Stt, None) {
        Err(r) => r,
        Ok(_) => crate::server::err(axum::http::StatusCode::NOT_IMPLEMENTED, "speech-to-text is not wired up yet"),
    }
}
