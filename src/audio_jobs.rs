//! Live-progress audio jobs (SSE) for the web UI.
//! OpenAI-compatible `/v1/audio/transcriptions` is unchanged for API clients.
use crate::registry::{App, Kind};
use crate::server::err;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use futures::StreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

type S = State<Arc<App>>;

struct Job { cancel: Arc<AtomicBool> }

fn jobs() -> &'static Mutex<HashMap<String, Job>> {
    static J: std::sync::OnceLock<Mutex<HashMap<String, Job>>> = std::sync::OnceLock::new();
    J.get_or_init(|| Mutex::new(HashMap::new()))
}

fn emit(tx: &mpsc::UnboundedSender<Value>, v: Value) { let _ = tx.send(v); }

async fn list_models(State(app): S) -> Response {
    #[cfg(feature = "stt")]
    {
        use pyre_whisper_engine::model::registry::{ModelCategory, MODEL_REGISTRY};
        let loaded: Vec<String> = app.loaded().iter().filter(|m| m.kind == Kind::Stt).map(|m| m.id.clone()).collect();
        let root = app.storage.store.root.join("stt").join("whisburn");
        let models: Vec<Value> = MODEL_REGISTRY.iter()
            .filter(|m| m.category == ModelCategory::Asr && m.burn_ready)
            .map(|m| {
                let present = root.join(m.name).join("model.bpk").is_file()
                    || root.join(m.name).join(format!("{}.bpk", m.name)).is_file();
                json!({
                    "id": m.name, "hf": m.hf_id, "description": m.description, "tagline": m.tagline,
                    "vram_mb": m.vram_mb, "present": present,
                    "loaded": loaded.iter().any(|id| id == m.name || id.contains(m.name)),
                    "size": format!("{:?}", m.size).to_ascii_lowercase(),
                    "quality": format!("{:?}", m.quality).to_ascii_lowercase(),
                })
            }).collect();
        return Json(json!({
            "models": models, "loaded": loaded,
            "formats": ["json","txt","srt","vtt","verbose_json","whisburn_json"],
            "defaults": { "model": loaded.first().cloned().unwrap_or_else(|| "base".into()),
                "format": "verbose_json", "language": "", "timestamps": true,
                "beam_size": 5, "max_tokens": 448, "sentences": false, "diarize": false, "task": "transcriptions" }
        })).into_response();
    }
    #[cfg(not(feature = "stt"))]
    { err(StatusCode::NOT_IMPLEMENTED, "built without stt") }
}

async fn cancel_job(Json(b): Json<Value>) -> Response {
    let id = b["id"].as_str().unwrap_or("");
    let ok = jobs().lock().unwrap().get(id).map(|j| { j.cancel.store(true, Ordering::SeqCst); true }).unwrap_or(false);
    Json(json!({"ok": ok, "id": id})).into_response()
}

async fn start_job_async(State(app): S, mp: Multipart) -> Response {
    #[cfg(not(feature = "stt"))]
    { return err(StatusCode::NOT_IMPLEMENTED, "built without stt"); }
    #[cfg(feature = "stt")]
    {
        let id = format!("job-{}-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0), std::process::id());
        let cancel = Arc::new(AtomicBool::new(false));
        jobs().lock().unwrap().insert(id.clone(), Job { cancel: cancel.clone() });
        let (tx, rx) = mpsc::unbounded_channel::<Value>();
        emit(&tx, json!({"stage":"queued","job":id,"message":"job accepted"}));
        let parsed = match crate::stt::parse_multipart(mp).await {
            Ok(q) => q,
            Err(e) => { jobs().lock().unwrap().remove(&id); return err(StatusCode::BAD_REQUEST, format!("{e:#}")); }
        };
        let translate = parsed.translate_task;
        let app2 = app.clone();
        let id2 = id.clone();
        let cancel2 = cancel.clone();
        let tx2 = tx.clone();
        let _ = std::thread::Builder::new().name("audio-job".into()).spawn(move || {
            let out = crate::stt::run_job(app2, parsed, translate, cancel2, &tx2, &id2);
            match out {
                Ok(v) => emit(&tx2, json!({"stage":"done","job":id2,"result":v})),
                Err(e) => {
                    let msg = e.to_string();
                    let stage = if msg.contains("cancelled") { "cancelled" } else { "error" };
                    emit(&tx2, json!({"stage":stage,"job":id2,"message":msg}));
                }
            }
            jobs().lock().unwrap().remove(&id2);
            #[cfg(feature = "stt")]
            pyre_whisper_engine::job_hooks::clear();
        });
        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(|v| {
            Ok::<_, std::convert::Infallible>(SseEvent::default().data(v.to_string()))
        });
        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    }
}

pub fn routes(r: Router<Arc<App>>) -> Router<Arc<App>> {
    r.route("/api/audio/models", axum::routing::get(list_models))
        .route("/api/audio/jobs", axum::routing::post(start_job_async).layer(axum::extract::DefaultBodyLimit::max(128 << 20)))
        .route("/api/audio/jobs/cancel", axum::routing::post(cancel_job))
}
