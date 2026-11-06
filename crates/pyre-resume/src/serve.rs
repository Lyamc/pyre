//! Reference OpenAI-compatible server used by the restart harness and as the
//! contract the pyre binary follows: request id on every chunk and on response
//! headers, SSE pings, an explicit terminal frame, and resume from the journal.
//!
//! The generator is injected so tests never load a model.

use crate::drain::Drain;
use crate::journal::{PlanError, Record, Session, Snapshot, Status, Store};
use crate::protocol::{code_for, data_frame, frames_failed, frames_finished, role_frame, Frame};
use axum::extract::{Path, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use tokio_stream::wrappers::UnboundedReceiverStream;

#[derive(Clone, Debug)]
pub struct Job {
    pub prompt_ids: Vec<u32>,
    pub max_tokens: usize,
    pub is_resume: bool,
    pub prior_text: String,
}

#[derive(Clone, Debug)]
pub enum MockEv {
    Token { text: String, id: u32 },
    Done { finish_reason: String },
    Error(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    /// Emit `finish_reason: stop`.
    Stop,
    /// Close the event channel with no terminal event (worker disappeared).
    Drop,
    /// Stay quiet until the sender is dropped.
    Hang,
    Error,
}

#[derive(Clone, Debug)]
pub struct Script {
    pub fresh: Vec<String>,
    pub fresh_end: End,
    pub resume: Vec<String>,
    pub resume_end: End,
    pub gap: Duration,
}

impl Script {
    /// Three tokens then silence. A resume appends `done` and stops.
    /// This is the shape of a watchdog kill mid-generation.
    pub fn hang_then_finish() -> Self {
        Self {
            fresh: vec!["one ".into(), "two ".into(), "three ".into()],
            fresh_end: End::Hang,
            resume: vec!["done".into()],
            resume_end: End::Stop,
            gap: Duration::from_millis(30),
        }
    }
}

pub type GenFn = Arc<dyn Fn(Job, UnboundedSender<MockEv>) + Send + Sync>;

pub fn scripted(script: Script) -> GenFn {
    Arc::new(move |job, tx| {
        let (parts, end) = if job.is_resume { (&script.resume, script.resume_end) } else { (&script.fresh, script.fresh_end) };
        let n = parts.len().min(job.max_tokens.max(1));
        for (i, p) in parts.iter().take(n).enumerate() {
            if !script.gap.is_zero() {
                std::thread::sleep(script.gap);
            }
            if tx.send(MockEv::Token { text: p.clone(), id: (if job.is_resume { 200 } else { 100 }) + i as u32 }).is_err() {
                return;
            }
        }
        match end {
            End::Stop => { let _ = tx.send(MockEv::Done { finish_reason: "stop".into() }); }
            End::Error => { let _ = tx.send(MockEv::Error("inference panicked: mock".into())); }
            End::Drop => {}
            End::Hang => {
                while !tx.is_closed() {
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
    })
}

pub struct AppState {
    pub store: Store,
    pub drain: Arc<Drain>,
    pub gen: GenFn,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/chat/completions", post(chat))
        .route("/v1/generations/{id}", get(generation))
        .route("/v1/generations/{id}/resume", post(resume_by_id))
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .with_state(state)
}

pub struct Running {
    pub addr: SocketAddr,
    pub state: Arc<AppState>,
    shutdown: Option<oneshot::Sender<()>>,
    join: Option<tokio::task::JoinHandle<()>>,
}

impl Running {
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(j) = self.join.take() {
            let _ = j.await;
        }
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
    }
}

pub async fn spawn(dir: PathBuf, gen: GenFn) -> Running {
    let store = Store::open(&dir).expect("store");
    let _ = store.recover();
    let state = Arc::new(AppState { store, drain: Drain::new(), gen });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel();
    let app = router(Arc::clone(&state));
    let join = tokio::spawn(async move {
        let _ = axum::serve(listener, app).with_graceful_shutdown(async move { let _ = rx.await; }).await;
    });
    Running { addr, state, shutdown: Some(tx), join: Some(join) }
}

/// Bind an ephemeral port, publish it to `port_file`, and serve until the process dies.
pub async fn listen_ephemeral(dir: PathBuf, port_file: PathBuf, gen: GenFn) -> std::io::Result<()> {
    let store = Store::open(&dir)?;
    let n = store.recover()?;
    eprintln!("pyre-resume: recovered {n} journal(s)");
    let state = Arc::new(AppState { store, drain: Drain::new(), gen });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    std::fs::write(&port_file, port.to_string())?;
    eprintln!("pyre-resume: listening on 127.0.0.1:{port}");
    axum::serve(listener, router(state)).await?;
    Ok(())
}

#[derive(Deserialize)]
struct Req {
    #[serde(default)]
    messages: Vec<Value>,
    model: Option<String>,
    resume: Option<String>,
    replay: Option<bool>,
    #[serde(rename = "continue", default)]
    cont: Option<bool>,
    replay_from: Option<usize>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
}

async fn generation(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match st.store.snapshot(&id) {
        Ok(s) => Json(s).into_response(),
        Err(PlanError::NotFound) => json_err(StatusCode::NOT_FOUND, "generation not found", "generation_not_found", None),
        Err(PlanError::BadId) => json_err(StatusCode::BAD_REQUEST, "invalid generation id", "invalid_request_error", None),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), "server_error", None),
    }
}

async fn resume_by_id(State(st): State<Arc<AppState>>, Path(id): Path<String>, Json(mut req): Json<Req>) -> Response {
    req.resume = Some(id);
    if !req.stream {
        req.stream = true;
    }
    chat(State(st), Json(req)).await
}

async fn chat(State(st): State<Arc<AppState>>, Json(req): Json<Req>) -> Response {
    if let Some(id) = req.resume.clone() {
        return resume(st, &req, &id).await;
    }
    let model = req.model.clone().unwrap_or_else(|| "mock".into());
    let id = crate::new_id("chatcmpl");
    let (prompt_ids, _) = prompt_of(&req.messages);
    let max_tokens = req.max_tokens.unwrap_or(32).max(1);
    let meta = crate::journal::OpenMeta {
        id: id.clone(),
        resume_token: id.clone(),
        model: model.clone(),
        created: now(),
        endpoint: "chat".into(),
        prompt_ids: prompt_ids.clone(),
        max_tokens,
        stop: vec![],
        sampling: json!({"temperature": 0.0}),
        messages: Value::Array(req.messages.clone()),
        tools: None,
        extra: json!({}),
    };
    let session = match Session::create(&st.store, meta) {
        Ok(s) => s,
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), "server_error", None),
    };
    let job = Job { prompt_ids, max_tokens, is_resume: false, prior_text: String::new() };
    begin(st, session, job, req.stream, vec![]).await
}

async fn resume(st: Arc<AppState>, req: &Req, id: &str) -> Response {
    let replay = req.replay.unwrap_or(true);
    let plan = match st.store.plan(id, replay, req.cont) {
        Ok(p) => p,
        Err(PlanError::NotFound) => return json_err(StatusCode::NOT_FOUND, "generation not found", "generation_not_found", Some(id)),
        Err(PlanError::BadId) => return json_err(StatusCode::BAD_REQUEST, "invalid generation id", "invalid_request_error", Some(id)),
        Err(PlanError::Busy) => {
            let snap = st.store.snapshot(id).ok();
            return busy(id, snap);
        }
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), "server_error", Some(id)),
    };
    let env = plan.record.meta.envelope();
    let replay_frames = replay_frames(&env, &plan.record.text, replay, req.replay_from);
    if plan.cont.is_none() {
        let mut frames = vec![role_frame(&env)];
        frames.extend(replay_frames);
        frames.extend(terminal_from_record(&plan.record));
        return respond_frames(&plan.record.meta.id, plan.record.status, req.stream, frames);
    }
    let cont = plan.cont.unwrap();
    let (session, _) = match Session::resume_append(&st.store, id) {
        Ok(s) => s,
        Err(PlanError::Busy) => return busy(id, st.store.snapshot(id).ok()),
        Err(PlanError::NotFound) => return json_err(StatusCode::NOT_FOUND, "generation not found", "generation_not_found", Some(id)),
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), "server_error", Some(id)),
    };
    let job = Job { prompt_ids: cont.prompt_ids, max_tokens: cont.max_tokens, is_resume: true, prior_text: cont.generated_text };
    begin(st, session, job, req.stream, replay_frames).await
}

fn replay_frames(env: &crate::protocol::Envelope, text: &str, replay: bool, replay_from: Option<usize>) -> Vec<Frame> {
    if !replay || text.is_empty() {
        return vec![];
    }
    let from = char_floor(text, replay_from.unwrap_or(0));
    let suffix = &text[from..];
    if suffix.is_empty() {
        vec![]
    } else {
        vec![data_frame(env, json!({"content": suffix}), true)]
    }
}

fn terminal_from_record(rec: &Record) -> Vec<Frame> {
    let env = rec.meta.envelope();
    match rec.status {
        Status::Finished => frames_finished(&env, rec.finish_reason.as_deref().unwrap_or("stop"), json!({})),
        Status::Error => frames_failed(&env, Status::Error, rec.message.as_deref().unwrap_or("generation failed"), code_for(Status::Error, false)),
        Status::Aborted | Status::Running => {
            frames_failed(&env, Status::Aborted, rec.message.as_deref().unwrap_or("generation aborted"), code_for(Status::Aborted, false))
        }
    }
}

async fn begin(st: Arc<AppState>, session: Session, job: Job, stream: bool, replay: Vec<Frame>) -> Response {
    let id = session.meta.id.clone();
    let (shut, guard) = st.drain.track();
    let (tx_ev, rx_ev) = unbounded_channel();
    let gen = Arc::clone(&st.gen);
    std::thread::spawn(move || gen(job, tx_ev));
    let (tx_out, rx_out) = unbounded_channel();
    let pump = pump(session, rx_ev, shut, tx_out, replay);
    if !stream {
        pump.await;
        drop(guard);
        let rec = match st.store.load(&id) {
            Ok(Some(r)) => r,
            _ => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "journal missing after generation", "server_error", Some(&id)),
        };
        return json_completion(&rec);
    }
    tokio::spawn(pump);
    let body = UnboundedReceiverStream::new(rx_out).map(|f: Frame| Ok::<_, Infallible>(Event::default().data(f.data)));
    let body = Hold { inner: body, _guard: guard };
    let sse = Sse::new(body).keep_alive(keepalive());
    stamp(sse.into_response(), &id, "running")
}

async fn pump(mut session: Session, mut rx: UnboundedReceiver<MockEv>, mut shut: tokio::sync::broadcast::Receiver<()>, out: UnboundedSender<Frame>, replay: Vec<Frame>) {
    let env = session.env();
    if out.send(role_frame(&env)).is_err() {
        return;
    }
    for f in replay {
        if out.send(f).is_err() {
            return;
        }
    }
    loop {
        tokio::select! {
            biased;
            s = shut.recv() => {
                match s {
                    Ok(()) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        if let Ok(frames) = session.close_failed(Status::Aborted, "server shutting down", "server_shutdown") {
                            for f in frames { let _ = out.send(f); }
                        }
                        return;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        std::future::pending::<()>().await;
                    }
                }
            }
            ev = rx.recv() => {
                match ev {
                    Some(MockEv::Token { text, id }) => {
                        let _ = session.note_token(&text, &[id]);
                        if out.send(data_frame(&session.env(), json!({"content": text}), false)).is_err() {
                            return;
                        }
                    }
                    Some(MockEv::Done { finish_reason }) => {
                        let text = session.text().to_string();
                        let n = session.ids().len();
                        let prompt_n = session.meta.prompt_ids.len();
                        let reason = if crate::is_success_finish(&finish_reason) { finish_reason } else { "stop".into() };
                        if let Ok(frames) = session.close_finished(&reason, &text, "", vec![], prompt_n, n, json!({})) {
                            for f in frames { let _ = out.send(f); }
                        }
                        return;
                    }
                    Some(MockEv::Error(m)) => {
                        if let Ok(frames) = session.close_failed(Status::Error, &m, "generation_error") {
                            for f in frames { let _ = out.send(f); }
                        }
                        return;
                    }
                    None => {
                        if let Ok(frames) = session.close_failed(Status::Aborted, "generation interrupted before a terminal event", "generation_aborted") {
                            for f in frames { let _ = out.send(f); }
                        }
                        return;
                    }
                }
            }
        }
    }
}

fn respond_frames(id: &str, status: Status, stream: bool, frames: Vec<Frame>) -> Response {
    if !stream {
        let text = frames.iter().filter_map(|f| serde_json::from_str::<Value>(&f.data).ok()).fold(String::new(), |mut acc, v| {
            if let Some(c) = v.pointer("/choices/0/delta/content").and_then(|c| c.as_str()) {
                acc.push_str(c);
            }
            acc
        });
        let finish = frames.iter().rev().filter_map(|f| serde_json::from_str::<Value>(&f.data).ok()).find_map(|v| {
            v.pointer("/choices/0/finish_reason").and_then(|x| x.as_str().map(str::to_string)).filter(|s| s != "null")
        });
        let body = json!({
            "id": id,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "pyre": {"resume_token": id, "status": status.as_str()},
        });
        return stamp(Json(body).into_response(), id, status.as_str());
    }
    let (tx, rx) = unbounded_channel();
    for f in frames {
        let _ = tx.send(f);
    }
    drop(tx);
    let body = UnboundedReceiverStream::new(rx).map(|f: Frame| Ok::<_, Infallible>(Event::default().data(f.data)));
    stamp(Sse::new(body).keep_alive(keepalive()).into_response(), id, status.as_str())
}

fn json_completion(rec: &Record) -> Response {
    let content = rec.content.clone().unwrap_or_else(|| rec.text.clone());
    let status_code = if rec.status == Status::Finished { StatusCode::OK } else { StatusCode::INTERNAL_SERVER_ERROR };
    let body = json!({
        "id": rec.meta.id,
        "object": "chat.completion",
        "created": rec.meta.created,
        "model": rec.meta.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": rec.finish_reason}],
        "pyre": {"resume_token": rec.meta.resume_token, "status": rec.status.as_str(), "message": rec.message, "resumable": rec.snapshot().resumable},
    });
    stamp((status_code, Json(body)).into_response(), &rec.meta.id, rec.status.as_str())
}

fn busy(id: &str, snap: Option<Snapshot>) -> Response {
    let body = json!({
        "error": {"message": "generation is still running on its original stream", "type": "server_error", "code": "generation_busy", "resume_token": id},
        "pyre": snap,
    });
    stamp((StatusCode::CONFLICT, Json(body)).into_response(), id, "running")
}

fn json_err(code: StatusCode, message: impl Into<String>, err_code: &str, id: Option<&str>) -> Response {
    let message = message.into();
    let mut err = json!({"message": message, "type": "server_error", "code": err_code});
    if let Some(id) = id {
        err["resume_token"] = json!(id);
    }
    (code, Json(json!({"error": err}))).into_response()
}

pub fn stamp(mut res: Response, id: &str, status: &str) -> Response {
    let h = res.headers_mut();
    if let Ok(v) = HeaderValue::from_str(id) {
        h.insert(header::HeaderName::from_static("x-request-id"), v.clone());
        h.insert(header::HeaderName::from_static("x-resume-token"), v);
    }
    if let Ok(v) = HeaderValue::from_str(status) {
        h.insert(header::HeaderName::from_static("x-pyre-status"), v);
    }
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    h.insert(header::HeaderName::from_static("x-accel-buffering"), HeaderValue::from_static("no"));
    res
}

fn keepalive() -> KeepAlive {
    KeepAlive::new().interval(Duration::from_secs(crate::ping_secs())).text("pyre-ping")
}

fn prompt_of(messages: &[Value]) -> (Vec<u32>, String) {
    let mut t = String::new();
    for m in messages {
        if let Some(c) = m.get("content").and_then(|c| c.as_str()) {
            t.push_str(c);
        }
    }
    let ids = t.chars().map(|c| c as u32).collect();
    (ids, t)
}

fn char_floor(s: &str, mut i: usize) -> usize {
    if i > s.len() {
        i = s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

struct Hold<S> {
    inner: S,
    _guard: crate::drain::TrackGuard,
}

impl<S: Stream + Unpin> Stream for Hold<S> {
    type Item = S::Item;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().inner).poll_next(cx)
    }
}

/// Used by the binary-style restart harness: `PYRE_RESUME_CHILD=1`.
pub fn child_main() {
    let dir = PathBuf::from(std::env::var("PYRE_RESUME_DIR").expect("PYRE_RESUME_DIR"));
    let port_file = PathBuf::from(std::env::var("PYRE_PORT_FILE").expect("PYRE_PORT_FILE"));
    let rt = tokio::runtime::Runtime::new().expect("rt");
    if let Err(e) = rt.block_on(listen_ephemeral(dir, port_file, scripted(Script::hang_then_finish()))) {
        eprintln!("pyre-resume child: {e}");
        std::process::exit(1);
    }
}

/// Block until `fut` completes. Exposed so callers can shut a child down in tests.
pub async fn listen_until(dir: PathBuf, gen: GenFn, fut: impl Future<Output = ()> + Send + 'static) -> SocketAddr {
    let store = Store::open(&dir).expect("store");
    let _ = store.recover();
    let state = Arc::new(AppState { store, drain: Drain::new(), gen });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router(state)).with_graceful_shutdown(fut).await;
    });
    addr
}
