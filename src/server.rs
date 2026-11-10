//! HTTP API: OpenAI-compatible (/v1/*) and llama.cpp-native endpoints.
use crate::engine::{EngineHandle, Event, GenRequest, Job, Timings};
use crate::sampler::SamplingParams;
use crate::template::ChatTemplate;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::{self, Stream, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};
use tokio_stream::wrappers::UnboundedReceiverStream;

/// One loaded text-generation model (its own worker queue).
pub struct Llm {
    pub engine: EngineHandle,
    pub tok: Arc<Tokenizer>,
    pub template: ChatTemplate,
    pub model_id: String,
    pub model_path: String,
    pub n_ctx: usize,
    pub backend: String,
    pub dtype: String,
    pub default_max_tokens: usize,
}

impl Llm {
    /// Context cap for this request. Qwen-Next follows the live setting (`/api/qwen/context`).
    pub fn n_ctx_now(&self) -> usize {
        if self.dtype == "gguf" { crate::qn_ctx::live_cap(self.n_ctx) } else { self.n_ctx }
    }
}
use crate::registry::App;
type S = State<Arc<App>>;
type AppState = Llm;

pub fn router(st: Arc<App>) -> Router {
    crate::apps::routes(crate::api::routes(Router::new()))
        .route("/v1/models", get(models))
        .route("/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/chat/completions", post(chat))
        .route("/v1/generations/{id}", get(generation))
        .route("/v1/generations/{id}/resume", post(resume_alias))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/embeddings", post(embeddings))
        .route("/completion", post(completion))
        .route("/completions", post(completion))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/props", get(props))
        .route("/slots", get(slots))
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .layer(middleware::from_fn_with_state(st.clone(), auth))
        .with_state(st)
}

pub fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    let m = msg.into();
    (code, Json(json!({"error": {"code": code.as_u16(), "message": m, "type": if code == StatusCode::UNAUTHORIZED {"authentication_error"} else {"invalid_request_error"}}}))).into_response()
}

/// CORS (like llama-server: any origin) + optional API-key check. Preflight never needs the key.
async fn auth(State(st): S, req: Request, next: Next) -> Response {
    use axum::http::{header, HeaderValue, Method};
    if req.method() == Method::OPTIONS {
        let mut r = StatusCode::NO_CONTENT.into_response();
        let ask = req.headers().get(header::ACCESS_CONTROL_REQUEST_HEADERS).cloned();
        let h = r.headers_mut();
        h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
        h.insert(header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static("GET, POST, PUT, DELETE, OPTIONS"));
        h.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, ask.unwrap_or(HeaderValue::from_static("authorization, content-type, x-api-key")));
        h.insert(header::ACCESS_CONTROL_MAX_AGE, HeaderValue::from_static("86400"));
        return r;
    }
    let mut r = auth_inner(st, req, next).await;
    let h = r.headers_mut();
    h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    h.insert(header::ACCESS_CONTROL_EXPOSE_HEADERS, HeaderValue::from_static("x-request-id, x-resume-token, x-pyre-status"));
    r
}

async fn auth_inner(st: Arc<App>, req: Request, next: Next) -> Response {
    if let Some(key) = &st.api_key {
        let p = req.uri().path();
        // health + the static UI shell are public; the UI then sends the key on API calls
        let public = p == "/health" || p == "/v1/health" || crate::ui::is_static(p);
        if !public {
            let ok = req.headers().get("authorization").and_then(|v| v.to_str().ok())
                .map(|v| v.strip_prefix("Bearer ").unwrap_or(v) == key).unwrap_or(false)
                || req.headers().get("x-api-key").and_then(|v| v.to_str().ok()) == Some(key.as_str());
            if !ok { return err(StatusCode::UNAUTHORIZED, "Invalid API Key"); }
        }
    }
    next.run(req).await
}

fn now() -> u64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() }
fn rid(p: &str) -> String { format!("{p}-{:016x}", rand::random::<u64>()) }

async fn health() -> Json<Value> { Json(json!({"status": "ok"})) }

async fn models(State(st): S) -> Json<Value> {
    Json(json!({"object": "list", "data": st.model_cards()}))
}

/// Common sampling fields shared by every generation endpoint (OpenAI + llama.cpp names).
#[derive(Deserialize, Default, Clone)]
struct Gen {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    n_predict: Option<i64>,
    stop: Option<Value>,
    seed: Option<u64>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<Value>,
    /// speculative decoding: draft length (0 = off); or `speculative: {"enabled": bool, "draft_len"|"n": k}`
    spec: Option<usize>,
    speculative: Option<Value>,
}

impl Gen {
    fn params(&self) -> SamplingParams {
        let d = SamplingParams::default();
        SamplingParams {
            temperature: self.temperature.unwrap_or(d.temperature),
            top_p: self.top_p.unwrap_or(d.top_p),
            top_k: self.top_k.unwrap_or(d.top_k),
            min_p: self.min_p.unwrap_or(d.min_p),
            repeat_penalty: self.repeat_penalty.unwrap_or(d.repeat_penalty),
            repeat_last_n: self.repeat_last_n.unwrap_or(d.repeat_last_n),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            seed: self.seed,
            spec: self.spec.or_else(|| self.speculative.as_ref().and_then(|v| match v {
                Value::Bool(false) => Some(0),
                Value::Number(n) => n.as_u64().map(|n| n as usize),
                Value::Object(o) => if o.get("enabled").and_then(|e| e.as_bool()) == Some(false) { Some(0) } else { o.get("draft_len").or(o.get("n")).and_then(|n| n.as_u64()).map(|n| n as usize) },
                _ => None })),
        }
    }
    fn stops(&self) -> Vec<String> {
        match &self.stop {
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
            _ => vec![],
        }
    }
    fn max(&self, st: &AppState, prompt_len: usize) -> usize {
        let room = st.n_ctx_now().saturating_sub(prompt_len);
        let m = self.max_completion_tokens.or(self.max_tokens)
            .or(self.n_predict.and_then(|n| if n < 0 { None } else { Some(n as usize) }))
            .unwrap_or(st.default_max_tokens.min(room));
        m.min(room)
    }
    fn include_usage(&self) -> bool {
        self.stream_options.as_ref().and_then(|o| o.get("include_usage")).and_then(|v| v.as_bool()).unwrap_or(false)
    }
}

fn start(st: &AppState, prompt: Vec<u32>, g: &Gen) -> UnboundedReceiver<Event> { start_tb(st, prompt, g, None, None, false) }

fn start_tb(st: &AppState, prompt: Vec<u32>, g: &Gen, think_budget: Option<usize>, auto_bias: Option<i32>, auto_tools: bool) -> UnboundedReceiver<Event> {
    let (tx, mut rx_in) = unbounded_channel();
    let (tx_out, rx) = unbounded_channel();
    let max_tokens = g.max(st, prompt.len());
    let mid = crate::metrics::begin(&st.model_id, prompt.len(), st.n_ctx_now(), max_tokens);
    st.engine.submit(Job::Gen(GenRequest { prompt, max_tokens, params: g.params(), stop: g.stops(), tx, think_budget, auto_bias, auto_tools }));
    // Forward engine events to the handler while recording live metrics (/api/metrics).
    // Dropping `rx_in` when the client goes away still cancels the job as before.
    tokio::spawn(async move {
        let mut done = false;
        while let Some(e) = rx_in.recv().await {
            match &e {
                Event::Token { .. } => crate::metrics::token(mid),
                Event::Done { timings, finish_reason, .. } => { crate::metrics::finish_ex(mid, "ok", Some(timings), Some(finish_reason), None); done = true; }
                Event::Error(e) => { crate::metrics::finish_ex(mid, "error", None, None, Some(e)); done = true; }
                _ => {}
            }
            if tx_out.send(e).is_err() { break; }
        }
        if !done { crate::metrics::finish(mid, "cancelled", None); }
    });
    rx
}

async fn collect(mut rx: UnboundedReceiver<Event>) -> Result<(String, &'static str, Timings, String), String> {
    while let Some(e) = rx.recv().await {
        match e {
            Event::Done { finish_reason, text, timings, stopping_word, .. } => return Ok((text, finish_reason, timings, stopping_word)),
            Event::Error(m) => return Err(m),
            _ => {}
        }
    }
    Err("worker dropped request".into())
}

fn usage(t: &Timings) -> Value {
    let p = t.prompt_n + t.cached_n;
    json!({"prompt_tokens": p, "completion_tokens": t.predicted_n, "total_tokens": p + t.predicted_n,
        "prompt_tokens_details": {"cached_tokens": t.cached_n}})
}

fn encode(st: &AppState, text: &str, special: bool) -> Result<Vec<u32>, Response> {
    st.tok.encode(text, special).map(|e| e.get_ids().to_vec()).map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))
}

type SseStream = Sse<axum::response::sse::KeepAliveStream<std::pin::Pin<Box<dyn Stream<Item = Result<SseEvent, Infallible>> + Send>>>>;
fn sse(s: impl Stream<Item = Value> + Send + 'static, done: bool) -> SseStream {
    let s = s.map(|v| Ok(SseEvent::default().data(v.to_string())));
    let s: std::pin::Pin<Box<dyn Stream<Item = Result<SseEvent, Infallible>> + Send>> = if done {
        Box::pin(s.chain(stream::once(async { Ok(SseEvent::default().data("[DONE]")) })))
    } else { Box::pin(s) };
    Sse::new(s).keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(pyre_resume::ping_secs())).text("pyre-ping"))
}

/// `x-request-id` / `x-resume-token` go out with the headers, before the first token,
/// so a client whose socket dies during prefill still knows which journal to resume.
fn stamp(mut res: Response, id: &str, status: &str) -> Response {
    use axum::http::{header::HeaderName, HeaderValue};
    let h = res.headers_mut();
    if let Ok(v) = HeaderValue::from_str(id) {
        h.insert(HeaderName::from_static("x-request-id"), v.clone());
        h.insert(HeaderName::from_static("x-resume-token"), v);
    }
    if let Ok(v) = HeaderValue::from_str(status) {
        h.insert(HeaderName::from_static("x-pyre-status"), v);
    }
    h.insert(HeaderName::from_static("cache-control"), HeaderValue::from_static("no-cache"));
    h.insert(HeaderName::from_static("x-accel-buffering"), HeaderValue::from_static("no"));
    res
}

fn frame_is_terminal(v: &Value) -> bool {
    v.pointer("/choices/0/finish_reason").and_then(|x| x.as_str()).is_some() || v.get("stop").and_then(|s| s.as_bool()) == Some(true)
}

/// If the body ends without a finish_reason / native `stop: true`, append an explicit abort.
/// `sse(..., true)` still writes the `[DONE]` sentinel.
fn ensure_terminal(s: impl Stream<Item = Value> + Send + 'static, abort: impl Fn() -> Vec<Value> + Send + 'static) -> impl Stream<Item = Value> + Send {
    let saw = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let saw2 = saw.clone();
    let marked = s.inspect(move |v| { if frame_is_terminal(v) { saw2.store(true, std::sync::atomic::Ordering::Relaxed); } });
    let tail = stream::once(async move {
        if saw.load(std::sync::atomic::Ordering::Relaxed) { Vec::new() } else { abort() }
    }).flat_map(stream::iter);
    marked.chain(tail)
}

fn frames_to_values(frames: Vec<pyre_resume::Frame>) -> Vec<Value> {
    frames.into_iter().filter(|f| !f.is_done()).filter_map(|f| serde_json::from_str(&f.data).ok()).collect()
}

/// Generation with a thinking budget (effort.rs). Budget None/0 = plain `start` (0 is handled by the prefilled empty
/// think block). Otherwise phase 1 runs with the budget; if it stops with "think_budget", the close text is streamed,
/// appended to the sequence (prompt + generated ids + close ids, so the cached state is strictly extended) and phase 2
/// generates the answer with the remaining max_tokens. The client sees one stream and one merged Done.
/// `auto`: Some(bias) = reasoning effort auto; the engine reports its choice as Event::Info (kept in `info`).
fn start_eff(st: &Arc<Llm>, ids: Vec<u32>, g: &Gen, budget: Option<usize>, auto: Option<i32>, auto_tools: bool, forced: Arc<std::sync::atomic::AtomicBool>, info: Arc<std::sync::Mutex<Value>>) -> UnboundedReceiver<Event> {
    let tb = budget.filter(|b| *b > 0);
    if tb.is_none() && auto.is_none() { return start(st, ids, g) }
    let max = g.max(st, ids.len());
    let mut rx1 = start_tb(st, ids.clone(), g, tb, auto, auto_tools);
    let (tx, rx) = unbounded_channel();
    let (st, mut g2) = (st.clone(), g.clone());
    tokio::spawn(async move {
        while let Some(e) = rx1.recv().await {
            let (text, t1, tokens) = match e {
                Event::Done { finish_reason: "think_budget", text, timings, tokens, .. } => (text, timings, tokens),
                Event::Info(v) => { *info.lock().unwrap() = v; continue; }
                other => { if tx.send(other).is_err() { return; } continue; }
            };
            // auto chose none: close the (empty) think block right away; otherwise a budget force-close
            let auto_none = info.lock().unwrap().pointer("/auto/level").and_then(|l| l.as_str()) == Some("none");
            // (taking an accepted speculative window into the output can already contain the close: then nothing is added)
            let closed = text.contains("</think>");
            if !auto_none && !closed { forced.store(true, std::sync::atomic::Ordering::Relaxed); crate::effort::FORCED.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
            let close = if closed { String::new() } else if auto_none { "\n</think>\n\n".to_string() } else { crate::effort::close_text() };
            let cids = st.tok.encode(close.as_str(), false).map(|e| e.get_ids().to_vec()).unwrap_or_default();
            if !auto_none { tracing::info!("reasoning: think budget {:?} reached ({} tokens), force-closing and continuing", tb.or(info.lock().unwrap().pointer("/auto/budget").and_then(|b| b.as_u64()).map(|b| b as usize)), tokens.len()); }
            if tx.send(Event::Token { text: close.clone(), ids: cids.clone() }).is_err() { return; }
            let mut ids2 = ids; ids2.extend_from_slice(&tokens); ids2.extend_from_slice(&cids);
            let rest = max.saturating_sub(tokens.len() + cids.len());
            let mut all = tokens; all.extend_from_slice(&cids);
            if rest == 0 {
                let _ = tx.send(Event::Done { finish_reason: "length", text: format!("{text}{close}"), timings: t1, stopping_word: String::new(), tokens: all });
                return;
            }
            g2.max_tokens = Some(rest); g2.max_completion_tokens = None; g2.n_predict = None;
            crate::effort::CONTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut rx2 = start_tb(&st, ids2, &g2, None, None, false);
            while let Some(e2) = rx2.recv().await {
                match e2 {
                    Event::Done { finish_reason, text: t2, timings: m2, stopping_word, tokens: k2 } => {
                        let pn = t1.predicted_n + cids.len() + m2.predicted_n; let pms = t1.predicted_ms + m2.prompt_ms + m2.predicted_ms;
                        let tm = Timings { prompt_n: t1.prompt_n, prompt_ms: t1.prompt_ms, prompt_per_second: t1.prompt_per_second, cached_n: t1.cached_n,
                            predicted_n: pn, predicted_ms: pms, predicted_per_second: if pms > 0.0 { pn as f64 / (pms / 1e3) } else { 0.0 } };
                        all.extend_from_slice(&k2);
                        let _ = tx.send(Event::Done { finish_reason, text: format!("{text}{close}{t2}"), timings: tm, stopping_word, tokens: all });
                        return;
                    }
                    other => { if tx.send(other).is_err() { return; } }
                }
            }
            return;
        }
    });
    rx
}

// ---------- reasoning split ----------
/// Splits `<think>...</think>` off a chat completion into `reasoning_content` (llama.cpp's
/// "deepseek" reasoning format). Handles templates whose generation prompt already opens
/// `<think>` (Qwen3.x thinking models): then the output starts inside the block.
/// PYRE_REASONING=none keeps everything in `content`.
struct Think { on: bool, enabled: bool, at_start: bool, strip_nl: bool, buf: String }
impl Think {
    fn new(prompt: &str) -> Self {
        let enabled = std::env::var("PYRE_REASONING").map(|v| v != "none").unwrap_or(true);
        Think { on: enabled && prompt.trim_end().ends_with("<think>"), enabled, at_start: true, strip_nl: false, buf: String::new() }
    }
    fn partial(s: &str, tag: &str) -> usize { (1..tag.len()).rev().find(|&k| s.len() >= k && s.is_char_boundary(s.len() - k) && tag.starts_with(&s[s.len() - k..])).unwrap_or(0) }
    /// Feed streamed text; returns (reasoning, content) ready to send.
    fn push(&mut self, s: &str) -> (String, String) {
        self.buf.push_str(s);
        let (mut r, mut c) = (String::new(), String::new());
        loop {
            if self.on {
                if let Some(k) = self.buf.find("</think>") { r.push_str(&self.buf[..k]); self.buf.drain(..k + 8); self.on = false; self.strip_nl = true; continue; }
                let n = self.buf.len() - Self::partial(&self.buf, "</think>");
                r.push_str(&self.buf[..n]); self.buf.drain(..n);
            } else {
                if self.enabled && self.at_start {
                    let t = self.buf.trim_start();
                    if t.starts_with("<think>") { let k = self.buf.find("<think>").unwrap(); self.buf.drain(..k + 7); self.on = true; self.at_start = false; continue; }
                    if "<think>".starts_with(t) { break; } // could still become the open tag
                }
                self.at_start = false;
                let mut out = std::mem::take(&mut self.buf);
                if self.strip_nl { let t = out.trim_start_matches(['\n', '\r']).to_string(); if !t.is_empty() { self.strip_nl = false; } out = t; }
                c.push_str(&out);
            }
            break;
        }
        if self.on { self.at_start = false; }
        (r, c)
    }
    fn finish(&mut self) -> (String, String) {
        let rest = std::mem::take(&mut self.buf);
        if self.on { (rest, String::new()) } else { (String::new(), rest) }
    }
}

fn think_delta(r: String, c: String) -> Option<Value> {
    match (r.is_empty(), c.is_empty()) {
        (true, true) => None,
        (false, true) => Some(json!({"reasoning_content": r})),
        (true, false) => Some(json!({"content": c})),
        (false, false) => Some(json!({"reasoning_content": r, "content": c})),
    }
}

/// OpenAI tool calls: the model writes `<tool_call>...</tool_call>` blocks in its content, either Qwen3-coder XML
/// (`<function=NAME><parameter=KEY>VALUE</parameter></function>`) or Hermes JSON (`{"name": .., "arguments": {..}}`).
/// Content before the first block streams as text; from `<tool_call>` on it is held back and returned as `tool_calls`
/// with finish_reason "tool_calls" (llama.cpp's behaviour). Parameter values are typed by the request's tool schemas.
/// PYRE_TOOL_PARSE=0 leaves the raw text in `content`.
struct Tools { on: bool, held: bool, buf: String, schema: Option<Value> }
impl Tools {
    fn new(tools: Option<&Value>) -> Self {
        let on = tools.map(|t| t.as_array().map(|a| !a.is_empty()).unwrap_or(false)).unwrap_or(false) && std::env::var("PYRE_TOOL_PARSE").map(|v| v != "0").unwrap_or(true);
        Tools { on, held: false, buf: String::new(), schema: tools.cloned() }
    }
    /// Feed content; returns the content that may be sent now.
    fn push(&mut self, c: &str) -> String {
        if !self.on { return c.to_string(); }
        self.buf.push_str(c);
        if self.held { return String::new(); }
        if let Some(k) = self.buf.find("<tool_call>") { self.held = true; let out: String = self.buf.drain(..k).collect(); return out; }
        let n = self.buf.len() - Think::partial(&self.buf, "<tool_call>");
        self.buf.drain(..n).collect()
    }
    /// End of output: (remaining content, tool calls).
    fn finish(&mut self) -> (String, Vec<Value>) {
        let rest = std::mem::take(&mut self.buf);
        if !self.held { return (rest, vec![]); }
        let mut calls = vec![]; let mut text = String::new(); let mut s = rest.as_str();
        while let Some(a) = s.find("<tool_call>") {
            text.push_str(&s[..a]); let body = &s[a + 11..];
            let (inner, after) = match body.find("</tool_call>") { Some(e) => (&body[..e], &body[e + 12..]), None => (body, "") };
            match self.parse(inner) { Some(c) => calls.push(c), None => { text.push_str("<tool_call>"); text.push_str(inner); } }
            s = after;
        }
        text.push_str(s);
        (text.trim().to_string(), calls)
    }
    fn ptype(&self, f: &str, k: &str) -> Option<String> {
        self.schema.as_ref()?.as_array()?.iter().find(|t| t["function"]["name"] == f)
            .and_then(|t| t["function"]["parameters"]["properties"][k]["type"].as_str().map(|x| x.to_string()))
    }
    fn parse(&self, inner: &str) -> Option<Value> {
        let t = inner.trim();
        let (name, args) = if t.starts_with('{') {
            let v: Value = serde_json::from_str(t).ok()?;
            let a = match &v["arguments"] { Value::String(x) => serde_json::from_str(x).unwrap_or(json!({})), Value::Null => json!({}), x => x.clone() };
            (v["name"].as_str()?.to_string(), a)
        } else {
            let a = t.find("<function=")? + 10; let e = a + t[a..].find('>')?;
            let name = t[a..e].trim().to_string(); let mut rest = &t[e + 1..]; let mut m = serde_json::Map::new();
            while let Some(p) = rest.find("<parameter=") {
                let q = p + 11; let Some(g) = rest[q..].find('>') else { break }; let key = rest[q..q + g].trim().to_string();
                let vs = &rest[q + g + 1..]; let ve = vs.find("</parameter>").unwrap_or(vs.len());
                let mut val = &vs[..ve]; if let Some(x) = val.strip_prefix('\n') { val = x; } if let Some(x) = val.strip_suffix('\n') { val = x; }
                let ty = self.ptype(&name, &key);
                let jv = if ty.as_deref() == Some("string") { Value::String(val.to_string()) } else { serde_json::from_str(val.trim()).unwrap_or(Value::String(val.to_string())) };
                m.insert(key, jv); rest = &vs[(ve + 12).min(vs.len())..];
            }
            (name, Value::Object(m))
        };
        if name.is_empty() { return None; }
        Some(json!({"id": rid("call"), "type": "function", "function": {"name": name, "arguments": args.to_string()}}))
    }
}

/// PYRE_REQ_LOG=<file> (diagnostic, off by default): append each chat request's messages and the final response as JSON lines.
fn reqlog(v: Value) -> bool {
    let Ok(f) = std::env::var("PYRE_REQ_LOG") else { return false };
    // PYRE_REQ_LOG=1: into the structured log (kind "reqbody"); any other value: that file, as before
    if f == "1" { crate::logbuf::event("reqbody", "DEBUG", json!({"body": v})); return true; }
    use std::io::Write;
    if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open(f) { let _ = writeln!(fh, "{}", v); }
    true
}

// ---------- OpenAI chat ----------
#[derive(Deserialize)]
struct ChatReq {
    /// Omitted on resume calls (`resume` identifies the generation).
    #[serde(default)]
    messages: Vec<Value>,
    model: Option<String>,
    tools: Option<Value>,
    /// reasoning effort (effort.rs): OpenAI `reasoning_effort`, OpenRouter `reasoning`, `chat_template_kwargs.enable_thinking`, pi's `thinking_token_budget`
    reasoning_effort: Option<Value>,
    reasoning: Option<Value>,
    chat_template_kwargs: Option<Value>,
    thinking_token_budget: Option<Value>,
    /// Continue or replay a generation recorded in the journal (`chatcmpl-...`).
    resume: Option<String>,
    /// Send the tokens already generated before any continuation. Default true.
    #[serde(default)]
    replay: Option<bool>,
    /// Generate the remainder of an aborted generation. Default: true only when it was aborted.
    #[serde(rename = "continue", default)]
    cont: Option<bool>,
    /// Byte offset into the raw generated text the client already has. Replay starts after it.
    #[serde(default)]
    replay_from: Option<usize>,
    #[serde(flatten)]
    g: Gen,
}

/// Flatten OpenAI content-part arrays to plain text for the template.
fn norm_messages(m: &[Value]) -> Vec<Value> {
    m.iter().map(|msg| {
        let mut msg = msg.clone();
        if let Some(Value::Array(parts)) = msg.get("content") {
            let t: String = parts.iter().filter_map(|p| p.get("text").and_then(|t| t.as_str())).collect::<Vec<_>>().join("");
            msg["content"] = Value::String(t);
        }
        if msg.get("content").map(|c| c.is_null()).unwrap_or(true) { msg["content"] = Value::String(String::new()); }
        // templates iterate tool_call.arguments as a mapping; OpenAI clients send it as a JSON string
        if let Some(Value::Array(tc)) = msg.get_mut("tool_calls") { for c in tc.iter_mut() {
            if let Some(Value::String(a)) = c.pointer("/function/arguments").cloned() { if let Ok(v) = serde_json::from_str::<Value>(&a) { c["function"]["arguments"] = v; } } } }
        msg
    }).collect()
}

async fn chat(State(app): S, Json(req): Json<ChatReq>) -> Response {
    if req.resume.is_some() { return resume_chat(app, req).await; }
    let st = match app.llm(req.model.as_deref()) { Ok(m) => m, Err(r) => return r };
    let mut msgs = norm_messages(&req.messages);
    let prompt = match st.template.render(&msgs, true, req.tools.as_ref()) {
        Ok(p) => p, Err(e) => return err(StatusCode::BAD_REQUEST, format!("chat template: {e:#}")),
    };
    let mut ids = match encode(&st, &prompt, false) { Ok(i) => i, Err(r) => return r };
    let mut prompt = prompt;
    // ctx-tooltrim: only above the threshold (below it the prompt is untouched)
    let lim = crate::ctx::trim_at(st.n_ctx_now());
    if crate::ctx::tooltrim_on() && ids.len() > lim {
        let mut count = |m: &[Value]| -> Option<usize> { let p = st.template.render(m, true, req.tools.as_ref()).ok()?; encode(&st, &p, false).ok().map(|i| i.len()) };
        if crate::ctx::tooltrim(&mut msgs, lim, &mut count) > 0 {
            if let Ok(p) = st.template.render(&msgs, true, req.tools.as_ref()) { if let Ok(i) = encode(&st, &p, false) { prompt = p; ids = i; } }
        }
    }
    // ctx-summarize: only above the threshold; one remembered compaction per history (prefix-cache friendly)
    let slim = crate::ctx::sum_at(st.n_ctx_now());
    if crate::ctx::summarize_on() && ids.len() > slim {
        let mut changed = false; while crate::ctx::apply_memo(&mut msgs) { changed = true; }
        let mut n = ids.len();
        if changed { if let Ok(p) = st.template.render(&msgs, true, req.tools.as_ref()) { if let Ok(i) = encode(&st, &p, false) { n = i.len(); prompt = p; ids = i; } } }
        if n > slim { if let Some((h0, end)) = crate::ctx::span(&msgs) {
            let sreq = crate::ctx::summary_request(&msgs[h0..end]);
            if let Ok(sp) = st.template.render(&sreq, true, None) { let sp = crate::ctx::no_think(sp); if let Ok(si) = encode(&st, &sp, false) { if si.len() + 2500 < st.n_ctx_now() {
                let g = Gen { temperature: Some(0.0), max_tokens: Some(2500), ..Default::default() };
                let mut th = Think::new(&sp);
                if let Ok((text, _, t, _)) = collect(start(&st, si, &g)).await {
                    let (_, mut c) = th.push(&text); let (_, c2) = th.finish(); c.push_str(&c2);
                    if !c.trim().is_empty() {
                        let v = crate::ctx::remember(&msgs, end, c.trim(), t.predicted_n); msgs.splice(h0..end, [v]);
                        if let Ok(p) = st.template.render(&msgs, true, req.tools.as_ref()) { if let Ok(i) = encode(&st, &p, false) {
                            tracing::warn!("ctx-summarize: messages {h0}..{end} -> summary ({} tok), prompt {} -> {} tokens (limit {slim})", t.predicted_n, n, i.len()); prompt = p; ids = i; } }
                    } } } } } } }
    }
    // reasoning effort: none = prefilled empty think block; low/medium/high = thinking budget; very_high = unlimited
    let plan = crate::effort::resolve(req.reasoning_effort.as_ref(), req.reasoning.as_ref(), req.chat_template_kwargs.as_ref(), req.thinking_token_budget.as_ref());
    crate::effort::count(&plan);
    let thinking = prompt.trim_end().ends_with("<think>");
    if thinking && plan.budget == Some(0) { let p = crate::ctx::no_think(prompt.clone()); if let Ok(i) = encode(&st, &p, false) { prompt = p; ids = i; } }
    let budget = if thinking { plan.budget } else { None };
    let auto = if thinking && plan.effort == crate::effort::Effort::Auto { Some(crate::effort::auto_bias(&msgs)) } else { None };
    let forced = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let info = Arc::new(std::sync::Mutex::new(Value::Null));
    let tokc = st.tok.clone();
    let ntok = move |r: &str| -> usize { if r.is_empty() { 0 } else { tokc.encode(r, false).map(|e| e.len()).unwrap_or(0) } };
    let model = req.model.clone().unwrap_or_else(|| st.model_id.clone());
    let id = rid("chatcmpl");
    let created = now();
    let mut think = Think::new(&prompt);
    let mut tools = Tools::new(req.tools.as_ref());
    let rlog = reqlog(json!({"req": &id, "stream": req.g.stream, "messages": &req.messages, "ntools": req.tools.as_ref().and_then(|t| t.as_array()).map(|a| a.len()),
        "effort": plan.effort.name(), "effort_explicit": plan.explicit, "budget": budget, "auto_bias": auto}));
    let has_tools = req.tools.as_ref().and_then(|t| t.as_array()).map(|a| !a.is_empty()).unwrap_or(false);
    let prompt_ids = ids.clone();
    let max_tokens = req.g.max(&st, prompt_ids.len());
    let rx = start_eff(&st, ids, &req.g, budget, auto, has_tools, forced.clone(), info.clone());
    let mut session = open_journal(&app, &id, &model, created, &prompt_ids, max_tokens, &req.g, Value::Array(msgs), req.tools.clone(), json!({"prompt_open_think": thinking, "think_budget": budget, "effort": plan.effort.name()}));
    let env = pyre_resume::Envelope::new(id.clone(), created, model.clone());
    if !req.g.stream {
        return match drive_until_done(rx, &mut session).await {
            Ok((text, fin, t, _)) => {
                let (mut r, mut c) = think.push(&text); let (r2, c2) = think.finish(); r.push_str(&r2); c.push_str(&c2);
                let mut c = tools.push(&c); let (c3, calls) = tools.finish(); c.push_str(&c3);
                let fin = if calls.is_empty() { fin } else { "tool_calls" };
                let mut msg = json!({"role": "assistant", "content": if calls.is_empty() || !c.trim().is_empty() { json!(c) } else { Value::Null }});
                if !calls.is_empty() { msg["tool_calls"] = json!(calls.clone()); }
                let tt = ntok(&r); crate::effort::THINK_TOKENS.fetch_add(tt as u64, std::sync::atomic::Ordering::Relaxed);
                let fc = forced.load(std::sync::atomic::Ordering::Relaxed);
                let ai = info.lock().unwrap().get("auto").cloned();
                if rlog { reqlog(json!({"resp": &id, "finish": fin, "message": &msg, "effort": plan.effort.name(), "budget": budget, "think_tokens": tt, "forced_close": fc, "auto": &ai})); }
                if !r.is_empty() { msg["reasoning_content"] = json!(r.clone()); }
                if let Some(sess) = session.as_mut() {
                    let _ = sess.close_finished(fin, &c, &r, calls, prompt_ids.len(), t.predicted_n, json!({}));
                }
                let mut u = usage(&t); u["completion_tokens_details"] = json!({"reasoning_tokens": tt});
                stamp(Json(json!({"id": id, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": msg, "finish_reason": fin, "logprobs": null}],
                "usage": u, "timings": t, "reasoning": {"effort": plan.effort.name(), "budget": budget, "thinking_tokens": tt, "forced_close": fc, "auto": ai},
                "pyre": {"resume_token": id, "status": "finished"}})).into_response(), &id, "finished") }
            Err(e) => err_resume(StatusCode::INTERNAL_SERVER_ERROR, e, &id),
        };
    }
    let inc = req.g.include_usage();
    let (tx, rx_sse) = unbounded_channel();
    let (shut, guard) = pyre_resume::drain().track();
    let effort = plan.effort.name();
    tokio::spawn(async move {
        let _ = tx.send(serde_json::from_str(&pyre_resume::role_frame(&env).data).unwrap_or(Value::Null));
        pump_chat(rx, shut, tx, env, session, think, tools, ntok, forced, info, rlog, inc, effort, budget, String::new(), String::new()).await;
    });
    let body = UnboundedReceiverStream::new(rx_sse);
    let guarded = HoldGuard { inner: body, _guard: guard };
    let abort_env = pyre_resume::Envelope::new(id.clone(), created, model.clone());
    let body = ensure_terminal(guarded, move || frames_to_values(pyre_resume::frames_failed(&abort_env, pyre_resume::Status::Aborted, "generation interrupted before a terminal event", "generation_aborted")));
    stamp(sse(body, true).into_response(), &id, "running")
}

/// Read engine events until a terminal one, recording each piece in the journal.
async fn drive_until_done(mut rx: UnboundedReceiver<Event>, session: &mut Option<pyre_resume::Session>) -> Result<(String, &'static str, Timings, String), String> {
    let (mut shut, _guard) = pyre_resume::drain().track();
    loop {
        tokio::select! {
            biased;
            s = shut.recv() => {
                match s {
                    Ok(()) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        if let Some(sess) = session.as_mut() { let _ = sess.close_failed(pyre_resume::Status::Aborted, "server shutting down", "server_shutdown"); }
                        return Err("server shutting down".into());
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => { std::future::pending::<()>().await; }
                }
            }
            ev = rx.recv() => match ev {
                Some(Event::Token { text, ids }) => { if let Some(sess) = session.as_mut() { let _ = sess.note_token(&text, &ids); } }
                Some(Event::Done { finish_reason, text, timings, stopping_word, .. }) => {
                    if finish_reason == "abort" {
                        if let Some(sess) = session.as_mut() { let _ = sess.close_failed(pyre_resume::Status::Aborted, "client disconnected", "generation_aborted"); }
                        return Err("client disconnected".into());
                    }
                    return Ok((text, finish_reason, timings, stopping_word));
                }
                Some(Event::Error(m)) => {
                    if let Some(sess) = session.as_mut() { let _ = sess.close_failed(pyre_resume::Status::Error, &m, "generation_error"); }
                    return Err(m);
                }
                Some(_) => {}
                None => {
                    if let Some(sess) = session.as_mut() { let _ = sess.close_failed(pyre_resume::Status::Aborted, "generation interrupted before a terminal event", "generation_aborted"); }
                    return Err("generation interrupted before a terminal event".into());
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn pump_chat(
    mut rx: UnboundedReceiver<Event>, mut shut: tokio::sync::broadcast::Receiver<()>, tx: tokio::sync::mpsc::UnboundedSender<Value>,
    env: pyre_resume::Envelope, mut session: Option<pyre_resume::Session>, mut think: Think, mut tools: Tools,
    ntok: impl Fn(&str) -> usize, forced: Arc<std::sync::atomic::AtomicBool>, info: Arc<std::sync::Mutex<Value>>,
    rlog: bool, inc: bool, effort: &'static str, budget: Option<usize>, mut acc_r: String, mut acc_c: String,
) {
    let send = |tx: &tokio::sync::mpsc::UnboundedSender<Value>, v: Vec<Value>| -> bool {
        v.into_iter().all(|item| tx.send(item).is_ok())
    };
    loop {
        let ev = tokio::select! {
            biased;
            s = shut.recv() => {
                match s {
                    Ok(()) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        let frames = fail_session(&env, &mut session, pyre_resume::Status::Aborted, "server shutting down", "server_shutdown");
                        let _ = send(&tx, frames);
                        return;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => { std::future::pending::<()>().await; continue; }
                }
            }
            ev = rx.recv() => ev,
        };
        match ev {
            Some(Event::Token { text, ids }) => {
                if let Some(sess) = session.as_mut() { let _ = sess.note_token(&text, &ids); }
                let (r, c) = think.push(&text);
                let c = tools.push(&c);
                acc_r.push_str(&r);
                acc_c.push_str(&c);
                if let Some(d) = think_delta(r, c) {
                    if tx.send(delta_value(&env, d, false)).is_err() { return; }
                }
            }
            Some(Event::Done { finish_reason, text, timings, tokens, .. }) => {
                note_tail(&mut session, &text, &tokens);
                if finish_reason == "abort" {
                    let _ = send(&tx, fail_session(&env, &mut session, pyre_resume::Status::Aborted, "client disconnected", "generation_aborted"));
                    return;
                }
                let (r, c) = think.finish();
                let mut c = tools.push(&c);
                let (c3, calls) = tools.finish();
                c.push_str(&c3);
                let mut v = Vec::new();
                if let Some(d) = think_delta(r.clone(), c.clone()) { v.push(delta_value(&env, d, false)); }
                acc_r.push_str(&r);
                let tt = ntok(&acc_r);
                crate::effort::THINK_TOKENS.fetch_add(tt as u64, std::sync::atomic::Ordering::Relaxed);
                let fc = forced.load(std::sync::atomic::Ordering::Relaxed);
                let ai = info.lock().unwrap().get("auto").cloned();
                let fin = if calls.is_empty() { finish_reason } else { "tool_calls" };
                if rlog { reqlog(json!({"resp": "stream", "finish": fin, "reasoning": &acc_r, "content": format!("{acc_c}{c}"), "tool_calls": &calls,
                    "effort": effort, "budget": budget, "think_tokens": tt, "forced_close": fc, "auto": &ai})); }
                if !calls.is_empty() {
                    v.push(delta_value(&env, json!({"tool_calls": calls.iter().enumerate().map(|(i, c)| { let mut c = c.clone(); c["index"] = json!(i); c }).collect::<Vec<_>>()}), false));
                }
                let extra = json!({"timings": timings, "reasoning": {"effort": effort, "budget": budget, "thinking_tokens": tt, "forced_close": fc, "auto": ai}});
                let prompt_n = session.as_ref().map(|s| s.meta.prompt_ids.len()).unwrap_or(0);
                let full_c = format!("{acc_c}{c}");
                let mut frames = if let Some(sess) = session.as_mut() {
                    match sess.close_finished(fin, &full_c, &acc_r, calls, prompt_n, timings.predicted_n, extra.clone()) {
                        Ok(f) => f,
                        Err(_) => pyre_resume::frames_finished(&env, fin, extra),
                    }
                } else {
                    pyre_resume::frames_finished(&env, fin, extra)
                };
                if inc {
                    let mut u = env.chunk(json!({}), None, pyre_resume::Status::Finished, false, None);
                    u["choices"] = json!([]);
                    u["usage"] = usage(&timings);
                    u["usage"]["completion_tokens_details"] = json!({"reasoning_tokens": tt});
                    let done = frames.pop();
                    frames.push(pyre_resume::Frame::json(&u));
                    if let Some(d) = done { frames.push(d); }
                }
                if !send(&tx, v) { return; }
                let _ = send(&tx, frames_to_values(frames));
                return;
            }
            Some(Event::Error(m)) => {
                let _ = send(&tx, fail_session(&env, &mut session, pyre_resume::Status::Error, &m, "generation_error"));
                return;
            }
            Some(_) => {}
            None => {
                let _ = send(&tx, fail_session(&env, &mut session, pyre_resume::Status::Aborted, "generation interrupted before a terminal event", "generation_aborted"));
                return;
            }
        }
    }
}

fn open_journal(app: &App, id: &str, model: &str, created: u64, prompt_ids: &[u32], max_tokens: usize, g: &Gen, messages: Value, tools: Option<Value>, extra: Value) -> Option<pyre_resume::Session> {
    if !pyre_resume::enabled() { return None; }
    let meta = pyre_resume::OpenMeta {
        id: id.to_string(), resume_token: id.to_string(), model: model.to_string(), created, endpoint: "chat".into(),
        prompt_ids: prompt_ids.to_vec(), max_tokens, stop: g.stops(), sampling: sampling_json(&g.params()), messages, tools, extra,
    };
    match pyre_resume::Session::create(&app.generations, meta) {
        Ok(s) => Some(s),
        Err(e) => { tracing::warn!("resume journal not opened for {id}: {e}"); None }
    }
}

fn sampling_json(p: &SamplingParams) -> Value {
    json!({"temperature": p.temperature, "top_p": p.top_p, "top_k": p.top_k, "min_p": p.min_p,
        "repeat_penalty": p.repeat_penalty, "repeat_last_n": p.repeat_last_n,
        "presence_penalty": p.presence_penalty, "frequency_penalty": p.frequency_penalty, "seed": p.seed, "spec": p.spec})
}

fn gen_from_saved(v: &Value, max_tokens: usize, stop: &[String]) -> Gen {
    let d = SamplingParams::default();
    let f = |k: &str, d: f32| v.get(k).and_then(|x| x.as_f64()).map(|n| n as f32).unwrap_or(d);
    let u = |k: &str, d: usize| v.get(k).and_then(|x| x.as_u64()).map(|n| n as usize).unwrap_or(d);
    Gen {
        temperature: Some(f("temperature", d.temperature)), top_p: Some(f("top_p", d.top_p)), top_k: Some(u("top_k", d.top_k)),
        min_p: Some(f("min_p", d.min_p)), repeat_penalty: Some(f("repeat_penalty", d.repeat_penalty)),
        repeat_last_n: Some(u("repeat_last_n", d.repeat_last_n)), presence_penalty: Some(f("presence_penalty", d.presence_penalty)),
        frequency_penalty: Some(f("frequency_penalty", d.frequency_penalty)), seed: v.get("seed").and_then(|x| x.as_u64()),
        spec: v.get("spec").and_then(|x| if x.is_null() { None } else { x.as_u64().map(|n| n as usize) }),
        max_tokens: Some(max_tokens), stop: Some(Value::Array(stop.iter().cloned().map(Value::String).collect())), stream: true,
        ..Default::default()
    }
}

fn err_resume(code: StatusCode, msg: impl Into<String>, id: &str) -> Response {
    let m = msg.into();
    stamp((code, Json(json!({"error": {"message": m, "type": "server_error", "code": code.as_u16(), "resume_token": id}}))).into_response(), id, "error")
}

fn delta_value(env: &pyre_resume::Envelope, delta: Value, replay: bool) -> Value {
    serde_json::from_str(&pyre_resume::data_frame(env, delta, replay).data).unwrap_or(Value::Null)
}

fn fail_session(env: &pyre_resume::Envelope, session: &mut Option<pyre_resume::Session>, status: pyre_resume::Status, message: &str, code: &str) -> Vec<Value> {
    if let Some(sess) = session.as_mut() {
        if let Ok(frames) = sess.close_failed(status, message, code) { return frames_to_values(frames); }
    }
    frames_to_values(pyre_resume::frames_failed(env, status, message, code))
}

fn note_tail(session: &mut Option<pyre_resume::Session>, full: &str, tokens: &[u32]) {
    let Some(sess) = session.as_mut() else { return };
    let have = sess.text().len();
    if full.len() > have && full.is_char_boundary(have) && full.starts_with(sess.text()) {
        let suf = full[have..].to_string();
        let n = sess.ids().len();
        let ids = if tokens.len() > n { tokens[n..].to_vec() } else { Vec::new() };
        let _ = sess.note_token(&suf, &ids);
    }
}

fn llm_exact(app: &App, name: &str) -> Result<Arc<Llm>, Response> {
    match app.find_exact(crate::registry::Kind::Llm, name) {
        Some(m) => match &m.inner {
            crate::registry::Inner::Llm(l) => Ok(l.clone()),
            _ => Err(err(StatusCode::CONFLICT, format!("model '{name}' is not a text model"))),
        },
        None => Err(err(StatusCode::CONFLICT, format!("model '{name}' is not loaded; the partial generation is still in the journal and can be resumed once it is loaded"))),
    }
}

struct HoldGuard<S> { inner: S, _guard: pyre_resume::TrackGuard }
impl<S: Stream + Unpin> Stream for HoldGuard<S> {
    type Item = S::Item;
    fn poll_next(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.get_mut().inner).poll_next(cx)
    }
}

async fn generation(State(app): S, axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    match app.generations.snapshot(&id) {
        Ok(s) => { let name = s.status.as_str(); stamp(Json(s).into_response(), &id, name) }
        Err(pyre_resume::PlanError::NotFound) => err(StatusCode::NOT_FOUND, "generation not found"),
        Err(pyre_resume::PlanError::BadId) => err(StatusCode::BAD_REQUEST, "invalid generation id"),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn resume_alias(State(app): S, axum::extract::Path(id): axum::extract::Path<String>, Json(mut req): Json<ChatReq>) -> Response {
    req.resume = Some(id);
    req.g.stream = true;
    chat(State(app), Json(req)).await
}

fn char_floor(s: &str, mut i: usize) -> usize {
    if i > s.len() { i = s.len(); }
    while i > 0 && !s.is_char_boundary(i) { i -= 1; }
    i
}

/// Feed `text` through the think/tool splitters. `emitted` is what a resume should put on the wire.
fn prime_split(text: &str, open_think: bool, tools_schema: Option<&Value>, replay: bool, replay_from: Option<usize>) -> (Think, Tools, Vec<Value>, String, String) {
    let mut think = Think::new(if open_think { "<think>" } else { "x" });
    let mut tools = Tools::new(tools_schema);
    let from = if replay { char_floor(text, replay_from.unwrap_or(0)) } else { text.len() };
    let (pre, suf) = text.split_at(from);
    let (r0, c0) = think.push(pre);
    let c0 = tools.push(&c0);
    let (r1, c1) = think.push(suf);
    let c1 = tools.push(&c1);
    let mut acc_r = r0;
    acc_r.push_str(&r1);
    let mut acc_c = c0;
    acc_c.push_str(&c1);
    let mut emitted = Vec::new();
    if replay {
        if let Some(d) = think_delta(r1, c1) { emitted.push(d); }
    }
    (think, tools, emitted, acc_r, acc_c)
}

async fn resume_chat(app: Arc<App>, req: ChatReq) -> Response {
    let id = match req.resume.clone() { Some(id) => id, None => return err(StatusCode::BAD_REQUEST, "resume id required") };
    let replay = req.replay.unwrap_or(true);
    let plan = match app.generations.plan(&id, replay, req.cont) {
        Ok(p) => p,
        Err(pyre_resume::PlanError::NotFound) => return err(StatusCode::NOT_FOUND, "generation not found"),
        Err(pyre_resume::PlanError::BadId) => return err(StatusCode::BAD_REQUEST, "invalid generation id"),
        Err(pyre_resume::PlanError::Busy) => {
            let snap = app.generations.snapshot(&id).ok();
            return stamp((StatusCode::CONFLICT, Json(json!({"error": {"message": "generation is still running on its original stream", "type": "server_error", "code": "generation_busy", "resume_token": id}, "pyre": snap}))).into_response(), &id, "running");
        }
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let rec = plan.record;
    let env = rec.meta.envelope();
    let open_think = rec.meta.extra.get("prompt_open_think").and_then(|v| v.as_bool()).unwrap_or(false);
    let (mut think, mut tools, replay_deltas, acc_r, acc_c) = prime_split(&rec.text, open_think, rec.meta.tools.as_ref(), plan.replay, req.replay_from);
    if plan.cont.is_none() {
        let mut frames = vec![serde_json::from_str(&pyre_resume::role_frame(&env).data).unwrap_or(Value::Null)];
        for d in replay_deltas { frames.push(delta_value(&env, d, true)); }
        let (r, c) = think.finish();
        let mut c = tools.push(&c);
        let (c3, calls) = tools.finish();
        c.push_str(&c3);
        if let Some(d) = think_delta(r, c) { frames.push(delta_value(&env, d, true)); }
        if !calls.is_empty() {
            frames.push(delta_value(&env, json!({"tool_calls": calls.iter().enumerate().map(|(i, c)| { let mut c = c.clone(); c["index"] = json!(i); c }).collect::<Vec<_>>()}), true));
        }
        let term = match rec.status {
            pyre_resume::Status::Finished => pyre_resume::frames_finished(&env, rec.finish_reason.as_deref().unwrap_or("stop"), json!({})),
            pyre_resume::Status::Error => pyre_resume::frames_failed(&env, pyre_resume::Status::Error, rec.message.as_deref().unwrap_or("generation failed"), "generation_error"),
            _ => pyre_resume::frames_failed(&env, pyre_resume::Status::Aborted, rec.message.as_deref().unwrap_or("generation aborted"), "generation_aborted"),
        };
        frames.extend(frames_to_values(term));
        let status = rec.status.as_str();
        if !req.g.stream {
            let content: String = frames.iter().filter_map(|v| v.pointer("/choices/0/delta/content").and_then(|c| c.as_str())).collect();
            let finish = frames.iter().rev().find_map(|v| v.pointer("/choices/0/finish_reason").and_then(|x| x.as_str())).unwrap_or("error");
            return stamp(Json(json!({"id": id, "object": "chat.completion", "created": rec.meta.created, "model": rec.meta.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
                "pyre": {"resume_token": id, "status": status}})).into_response(), &id, status);
        }
        let body = stream::iter(frames);
        let abort_env = env.clone();
        let body = ensure_terminal(body, move || frames_to_values(pyre_resume::frames_failed(&abort_env, pyre_resume::Status::Aborted, "generation interrupted before a terminal event", "generation_aborted")));
        return stamp(sse(body, true).into_response(), &id, status);
    }
    let st = match llm_exact(&app, &rec.meta.model) {
        Ok(s) => s,
        Err(_) => {
            let snap = rec.snapshot();
            let message = format!("model '{}' is not loaded; the partial generation is still in the journal", rec.meta.model);
            return stamp((StatusCode::CONFLICT, Json(json!({"error": {"message": message, "type": "server_error", "code": "model_not_loaded", "resume_token": id}, "pyre": snap}))).into_response(), &id, snap.status.as_str());
        }
    };
    let (session, _) = match pyre_resume::Session::resume_append(&app.generations, &id) {
        Ok(s) => s,
        Err(pyre_resume::PlanError::Busy) => {
            let snap = app.generations.snapshot(&id).ok();
            return stamp((StatusCode::CONFLICT, Json(json!({"error": {"message": "generation is still running on its original stream", "type": "server_error", "code": "generation_busy", "resume_token": id}, "pyre": snap}))).into_response(), &id, "running");
        }
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let cont = plan.cont.expect("cont");
    let g = gen_from_saved(&cont.sampling, cont.max_tokens, &cont.stop);
    let rx = start(&st, cont.prompt_ids, &g);
    let forced = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let info = Arc::new(std::sync::Mutex::new(Value::Null));
    let tokc = st.tok.clone();
    let ntok = move |r: &str| -> usize { if r.is_empty() { 0 } else { tokc.encode(r, false).map(|e| e.len()).unwrap_or(0) } };
    let (tx, rx_sse) = unbounded_channel();
    let (shut, guard) = pyre_resume::drain().track();
    let inc = req.g.include_usage();
    let env2 = env.clone();
    tokio::spawn(async move {
        if tx.send(serde_json::from_str(&pyre_resume::role_frame(&env2).data).unwrap_or(Value::Null)).is_err() { return; }
        for d in replay_deltas {
            if tx.send(delta_value(&env2, d, true)).is_err() { return; }
        }
        pump_chat(rx, shut, tx, env2, Some(session), think, tools, ntok, forced, info, false, inc, "resume", None, acc_r, acc_c).await;
    });
    let body = UnboundedReceiverStream::new(rx_sse);
    let guarded = HoldGuard { inner: body, _guard: guard };
    let abort_env = env.clone();
    let body = ensure_terminal(guarded, move || frames_to_values(pyre_resume::frames_failed(&abort_env, pyre_resume::Status::Aborted, "generation interrupted before a terminal event", "generation_aborted")));
    if req.g.stream {
        stamp(sse(body, true).into_response(), &id, "running")
    } else {
        // Non-streaming resume: collect the same frames the stream would have sent.
        let mut frames = Vec::new();
        let mut body = std::pin::pin!(body);
        while let Some(v) = futures::StreamExt::next(&mut body).await { frames.push(v); }
        let content: String = frames.iter().filter_map(|v| v.pointer("/choices/0/delta/content").and_then(|c| c.as_str())).collect();
        let finish = frames.iter().rev().find_map(|v| v.pointer("/choices/0/finish_reason").and_then(|x| x.as_str())).unwrap_or("error");
        let status = if pyre_resume::is_success_finish(finish) { "finished" } else { "aborted" };
        stamp(Json(json!({"id": id, "object": "chat.completion", "created": rec.meta.created, "model": rec.meta.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "pyre": {"resume_token": id, "status": status}})).into_response(), &id, status)
    }
}

// ---------- OpenAI legacy completions ----------
#[derive(Deserialize)]
struct CompReq { prompt: Value, model: Option<String>, #[serde(flatten)] g: Gen }

fn prompt_ids(st: &AppState, p: &Value) -> Result<Vec<u32>, Response> {
    match p {
        Value::String(s) => encode(st, s, true),
        Value::Array(a) if a.iter().all(|x| x.is_u64()) => Ok(a.iter().map(|x| x.as_u64().unwrap() as u32).collect()),
        Value::Array(a) => match a.first() { Some(Value::String(s)) => encode(st, s, true), _ => Err(err(StatusCode::BAD_REQUEST, "unsupported prompt")) },
        _ => Err(err(StatusCode::BAD_REQUEST, "prompt must be a string or token array")),
    }
}

async fn completions(State(app): S, Json(req): Json<CompReq>) -> Response {
    let st = match app.llm(req.model.as_deref()) { Ok(m) => m, Err(r) => return r };
    let ids = match prompt_ids(&st, &req.prompt) { Ok(i) => i, Err(r) => return r };
    let model = req.model.clone().unwrap_or_else(|| st.model_id.clone());
    let id = rid("cmpl");
    let created = now();
    let rx = start(&st, ids, &req.g);
    if !req.g.stream {
        return match collect(rx).await {
            Ok((text, fin, t, _)) => stamp(Json(json!({"id": &id, "object": "text_completion", "created": created, "model": model,
                "choices": [{"index": 0, "text": text, "finish_reason": fin, "logprobs": null}], "usage": usage(&t)})).into_response(), &id, "finished"),
            Err(e) => err_resume(StatusCode::INTERNAL_SERVER_ERROR, e, &id),
        };
    }
    let inc = req.g.include_usage();
    let env = pyre_resume::Envelope::new(id.clone(), created, model.clone());
    let body = UnboundedReceiverStream::new(rx).flat_map(move |e| {
        let mk = |text: &str, fin: Value| json!({"id": id, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": fin, "logprobs": null}]});
        let v = match e {
            Event::Token { text, .. } => vec![mk(&text, Value::Null)],
            Event::Done { finish_reason, timings, .. } => {
                let mut f = mk("", json!(finish_reason));
                if inc { f["usage"] = usage(&timings); }
                vec![f]
            }
            Event::Error(m) => frames_to_values(pyre_resume::frames_failed(&pyre_resume::Envelope::new(id.clone(), created, model.clone()), pyre_resume::Status::Error, &m, "generation_error")),
            _ => vec![],
        };
        stream::iter(v)
    });
    let env2 = env.clone();
    let body = ensure_terminal(body, move || frames_to_values(pyre_resume::frames_failed(&env2, pyre_resume::Status::Aborted, "generation interrupted before a terminal event", "generation_aborted")));
    stamp(sse(body, true).into_response(), &env.id, "running")
}

// ---------- embeddings ----------
#[derive(Deserialize)]
struct EmbReq { input: Value, model: Option<String> }

async fn embeddings(State(app): S, Json(req): Json<EmbReq>) -> Response {
    let st = match app.llm(req.model.as_deref()) { Ok(m) => m, Err(r) => return r };
    let inputs: Vec<Value> = match &req.input {
        Value::Array(a) if !a.is_empty() && !a[0].is_u64() => a.clone(),
        v => vec![v.clone()],
    };
    let mut data = vec![];
    let mut total = 0;
    for (i, inp) in inputs.iter().enumerate() {
        let ids = match prompt_ids(&st, inp) { Ok(i) => i, Err(r) => return r };
        if ids.is_empty() { return err(StatusCode::BAD_REQUEST, "empty input"); }
        let (tx, mut rx) = unbounded_channel();
        st.engine.submit(Job::Embed { tokens: ids, tx });
        match rx.recv().await {
            Some(Event::Embedding(v, n)) => { total += n; data.push(json!({"object": "embedding", "index": i, "embedding": v})); }
            _ => return err(StatusCode::INTERNAL_SERVER_ERROR, "embedding failed"),
        }
    }
    Json(json!({"object": "list", "data": data, "model": req.model.unwrap_or_else(|| st.model_id.clone()),
        "usage": {"prompt_tokens": total, "total_tokens": total}})).into_response()
}

// ---------- llama.cpp native ----------
#[derive(Deserialize)]
struct NativeReq { prompt: Value, #[serde(flatten)] g: Gen }

async fn completion(State(app): S, Json(req): Json<NativeReq>) -> Response {
    let st = match app.llm(None) { Ok(m) => m, Err(r) => return r };
    let ids = match prompt_ids(&st, &req.prompt) { Ok(i) => i, Err(r) => return r };
    let n_eval = ids.len();
    let rx = start(&st, ids, &req.g);
    let model = st.model_path.clone();
    if !req.g.stream {
        return match collect(rx).await {
            Ok((text, fin, t, sw)) => Json(json!({"content": text, "stop": true, "model": model, "tokens_predicted": t.predicted_n,
                "tokens_evaluated": n_eval, "tokens_cached": t.cached_n, "stopped_eos": fin == "stop" && sw.is_empty(), "stopped_word": !sw.is_empty(),
                "stopped_limit": fin == "length", "stopping_word": sw, "timings": t, "id_slot": 0})).into_response(),
            Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
        };
    }
    let body = UnboundedReceiverStream::new(rx).filter_map(move |e| {
        let model = model.clone();
        async move { match e {
            Event::Token { text, .. } => Some(json!({"content": text, "stop": false, "id_slot": 0})),
            Event::Done { finish_reason, timings, stopping_word, .. } => Some(json!({"content": "", "stop": true, "model": model,
                "tokens_predicted": timings.predicted_n, "tokens_evaluated": n_eval, "stopped_limit": finish_reason == "length",
                "stopping_word": stopping_word, "timings": timings, "id_slot": 0})),
            Event::Error(m) => Some(json!({"content": "", "stop": true, "error": {"message": m, "type": "server_error", "code": "generation_error"}})),
            _ => None,
        }}
    });
    let body = ensure_terminal(body, || vec![json!({"content": "", "stop": true, "error": {"message": "generation interrupted before a terminal event", "type": "server_error", "code": "generation_aborted"}})]);
    sse(body, false).into_response()
}

#[derive(Deserialize)]
struct TokReq { content: String, #[serde(default)] add_special: bool, #[serde(default)] with_pieces: bool }
async fn tokenize(State(app): S, Json(r): Json<TokReq>) -> Response {
    let st = match app.llm(None) { Ok(m) => m, Err(r) => return r };
    match st.tok.encode(r.content.as_str(), r.add_special) {
        Ok(e) if r.with_pieces => Json(json!({"tokens": e.get_ids().iter().zip(e.get_tokens()).map(|(i, p)| json!({"id": i, "piece": p})).collect::<Vec<_>>()})).into_response(),
        Ok(e) => Json(json!({"tokens": e.get_ids()})).into_response(),
        Err(e) => err(StatusCode::BAD_REQUEST, e.to_string()),
    }
}

#[derive(Deserialize)]
struct DetokReq { tokens: Vec<u32> }
async fn detokenize(State(app): S, Json(r): Json<DetokReq>) -> Response {
    let st = match app.llm(None) { Ok(m) => m, Err(r) => return r };
    match st.tok.decode(&r.tokens, false) {
        Ok(s) => Json(json!({"content": s})).into_response(),
        Err(e) => err(StatusCode::BAD_REQUEST, e.to_string()),
    }
}

async fn props(State(app): S) -> Response {
    let st = match app.llm(None) { Ok(m) => m, Err(r) => return r };
    let d = SamplingParams::default();
    Json(json!({
        "default_generation_settings": {"n_ctx": st.n_ctx_now(), "params": {"temperature": d.temperature, "top_k": d.top_k, "top_p": d.top_p,
            "min_p": d.min_p, "repeat_penalty": d.repeat_penalty, "repeat_last_n": d.repeat_last_n, "n_predict": st.default_max_tokens}},
        "total_slots": 1, "model_path": st.model_path, "model_alias": st.model_id, "chat_template": st.template.source,
        "bos_token": st.template.bos_token, "eos_token": st.template.eos_token,
        "build_info": format!("pyre {} (burn 0.22, backend {}, {})", env!("CARGO_PKG_VERSION"), st.backend, st.dtype),
        "models": app.model_cards(),
    })).into_response()
}

async fn slots(State(app): S) -> Json<Value> {
    Json(json!(app.loaded().iter().enumerate().map(|(i, m)| json!({"id": i, "model": m.id, "kind": m.kind.as_str(),
        "is_processing": m.busy(), "queued": m.queued(), "capabilities": m.capabilities().to_value_for(&m.id),
        "n_ctx": m.meta().get("n_ctx").cloned().unwrap_or(Value::Null)})).collect::<Vec<_>>()))
}
