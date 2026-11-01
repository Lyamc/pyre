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
use crate::registry::App;
type S = State<Arc<App>>;
type AppState = Llm;

pub fn router(st: Arc<App>) -> Router {
    crate::apps::routes(crate::api::routes(Router::new()))
        .route("/v1/models", get(models))
        .route("/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/chat/completions", post(chat))
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
    r.headers_mut().insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
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
    let data: Vec<Value> = st.loaded().iter().map(|m| json!({"id": m.id, "object": "model", "created": m.created, "owned_by": "pyre",
        "kind": m.kind.as_str(), "task": m.kind.task(), "meta": m.meta()})).collect();
    Json(json!({"object": "list", "data": data}))
}

/// Common sampling fields shared by every generation endpoint (OpenAI + llama.cpp names).
#[derive(Deserialize, Default)]
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
        let room = st.n_ctx.saturating_sub(prompt_len);
        let m = self.max_completion_tokens.or(self.max_tokens)
            .or(self.n_predict.and_then(|n| if n < 0 { None } else { Some(n as usize) }))
            .unwrap_or(st.default_max_tokens.min(room));
        m.min(room)
    }
    fn include_usage(&self) -> bool {
        self.stream_options.as_ref().and_then(|o| o.get("include_usage")).and_then(|v| v.as_bool()).unwrap_or(false)
    }
}

fn start(st: &AppState, prompt: Vec<u32>, g: &Gen) -> UnboundedReceiver<Event> {
    let (tx, mut rx_in) = unbounded_channel();
    let (tx_out, rx) = unbounded_channel();
    let max_tokens = g.max(st, prompt.len());
    let mid = crate::metrics::begin(&st.model_id, prompt.len(), st.n_ctx, max_tokens);
    st.engine.submit(Job::Gen(GenRequest { prompt, max_tokens, params: g.params(), stop: g.stops(), tx }));
    // Forward engine events to the handler while recording live metrics (/api/metrics).
    // Dropping `rx_in` when the client goes away still cancels the job as before.
    tokio::spawn(async move {
        let mut done = false;
        while let Some(e) = rx_in.recv().await {
            match &e {
                Event::Token(_) => crate::metrics::token(mid),
                Event::Done { timings, .. } => { crate::metrics::finish(mid, "ok", Some(timings)); done = true; }
                Event::Error(_) => { crate::metrics::finish(mid, "error", None); done = true; }
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
            Event::Done { finish_reason, text, timings, stopping_word } => return Ok((text, finish_reason, timings, stopping_word)),
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
    Sse::new(s).keep_alive(KeepAlive::default())
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

// ---------- OpenAI chat ----------
#[derive(Deserialize)]
struct ChatReq {
    messages: Vec<Value>,
    model: Option<String>,
    tools: Option<Value>,
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
    let st = match app.llm(req.model.as_deref()) { Ok(m) => m, Err(r) => return r };
    let prompt = match st.template.render(&norm_messages(&req.messages), true, req.tools.as_ref()) {
        Ok(p) => p, Err(e) => return err(StatusCode::BAD_REQUEST, format!("chat template: {e:#}")),
    };
    let ids = match encode(&st, &prompt, false) { Ok(i) => i, Err(r) => return r };
    let model = req.model.clone().unwrap_or_else(|| st.model_id.clone());
    let id = rid("chatcmpl");
    let created = now();
    let mut think = Think::new(&prompt);
    let mut tools = Tools::new(req.tools.as_ref());
    let rx = start(&st, ids, &req.g);
    if !req.g.stream {
        return match collect(rx).await {
            Ok((text, fin, t, _)) => {
                let (mut r, mut c) = think.push(&text); let (r2, c2) = think.finish(); r.push_str(&r2); c.push_str(&c2);
                let mut c = tools.push(&c); let (c3, calls) = tools.finish(); c.push_str(&c3);
                let fin = if calls.is_empty() { fin } else { "tool_calls" };
                let mut msg = json!({"role": "assistant", "content": if calls.is_empty() || !c.trim().is_empty() { json!(c) } else { Value::Null }});
                if !calls.is_empty() { msg["tool_calls"] = json!(calls); }
                if !r.is_empty() { msg["reasoning_content"] = json!(r); }
                Json(json!({"id": id, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": msg, "finish_reason": fin, "logprobs": null}],
                "usage": usage(&t), "timings": t})).into_response() }
            Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
        };
    }
    let inc = req.g.include_usage();
    let chunk = move |delta: Value, fin: Value| json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": fin, "logprobs": null}]});
    let first = stream::once({ let c = chunk.clone(); async move { c(json!({"role": "assistant", "content": ""}), Value::Null) } });
    let body = UnboundedReceiverStream::new(rx).flat_map(move |e| {
        let v: Vec<Value> = match e {
            Event::Token(s) => { let (r, c) = think.push(&s); let c = tools.push(&c); think_delta(r, c).map(|d| chunk(d, Value::Null)).into_iter().collect() }
            Event::Done { finish_reason, timings, .. } => {
                let (r, c) = think.finish(); let mut c = tools.push(&c); let (c3, calls) = tools.finish(); c.push_str(&c3);
                let mut v: Vec<Value> = think_delta(r, c).map(|d| chunk(d, Value::Null)).into_iter().collect();
                if !calls.is_empty() { v.push(chunk(json!({"tool_calls": calls.iter().enumerate().map(|(i, c)| { let mut c = c.clone(); c["index"] = json!(i); c }).collect::<Vec<_>>()}), Value::Null)); }
                let mut f = chunk(json!({}), json!(if calls.is_empty() { finish_reason } else { "tool_calls" }));
                f["timings"] = json!(timings);
                v.push(f);
                if inc { let mut u = chunk(json!({}), Value::Null); u["choices"] = json!([]); u["usage"] = usage(&timings); v.push(u); }
                v
            }
            Event::Error(m) => vec![json!({"error": {"message": m}})],
            _ => vec![],
        };
        stream::iter(v)
    });
    sse(first.chain(body), true).into_response()
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
            Ok((text, fin, t, _)) => Json(json!({"id": id, "object": "text_completion", "created": created, "model": model,
                "choices": [{"index": 0, "text": text, "finish_reason": fin, "logprobs": null}], "usage": usage(&t)})).into_response(),
            Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
        };
    }
    let inc = req.g.include_usage();
    let body = UnboundedReceiverStream::new(rx).flat_map(move |e| {
        let mk = |text: &str, fin: Value| json!({"id": id, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": fin, "logprobs": null}]});
        let v = match e {
            Event::Token(s) => vec![mk(&s, Value::Null)],
            Event::Done { finish_reason, timings, .. } => {
                let mut f = mk("", json!(finish_reason));
                if inc { f["usage"] = usage(&timings); }
                vec![f]
            }
            Event::Error(m) => vec![json!({"error": {"message": m}})],
            _ => vec![],
        };
        stream::iter(v)
    });
    sse(body, true).into_response()
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
            Event::Token(s) => Some(json!({"content": s, "stop": false, "id_slot": 0})),
            Event::Done { finish_reason, timings, stopping_word, .. } => Some(json!({"content": "", "stop": true, "model": model,
                "tokens_predicted": timings.predicted_n, "tokens_evaluated": n_eval, "stopped_limit": finish_reason == "length",
                "stopping_word": stopping_word, "timings": timings, "id_slot": 0})),
            Event::Error(m) => Some(json!({"error": {"message": m}})),
            _ => None,
        }}
    });
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
        "default_generation_settings": {"n_ctx": st.n_ctx, "params": {"temperature": d.temperature, "top_k": d.top_k, "top_p": d.top_p,
            "min_p": d.min_p, "repeat_penalty": d.repeat_penalty, "repeat_last_n": d.repeat_last_n, "n_predict": st.default_max_tokens}},
        "total_slots": 1, "model_path": st.model_path, "model_alias": st.model_id, "chat_template": st.template.source,
        "bos_token": st.template.bos_token, "eos_token": st.template.eos_token,
        "build_info": format!("pyre {} (burn 0.22, backend {}, {})", env!("CARGO_PKG_VERSION"), st.backend, st.dtype),
        "models": app.loaded().iter().map(|m| json!({"id": m.id, "kind": m.kind.as_str()})).collect::<Vec<_>>(),
    })).into_response()
}

async fn slots(State(app): S) -> Json<Value> {
    Json(json!(app.loaded().iter().enumerate().map(|(i, m)| json!({"id": i, "model": m.id, "kind": m.kind.as_str(),
        "is_processing": m.busy(), "queued": m.queued(), "n_ctx": m.meta().get("n_ctx").cloned().unwrap_or(Value::Null)})).collect::<Vec<_>>()))
}
