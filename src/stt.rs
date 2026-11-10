//! Speech-to-text model kind, backed by pyre-whisper-* (Whisper & co., vendored from the user's whisburn
//! @ d27f1fc and ported to the workspace Burn; pyre owns this code now).
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
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(feature = "stt")]
use pyre_whisper_engine::{model::registry::MODEL_REGISTRY, runtime::InferenceRuntime};

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
    let named = r.strip_prefix("whisburn:").unwrap_or(r);
    for m in MODEL_REGISTRY {
        if m.name.eq_ignore_ascii_case(named) || m.hf_id.eq_ignore_ascii_case(named) {
            if !matches!(m.category, pyre_whisper_engine::model::registry::ModelCategory::Asr) {
                bail!("whisburn model '{}' is not a speech-recognition model", m.name);
            }
            return Ok((m.name.to_string(), None));
        }
    }
    // A registry name wins over a relative directory. If the server's cwd is the
    // models folder, `base` is a directory and used to be treated as an explicit
    // bundle path ("no whisburn bundle for 'base' in models\\base").
    let p = Path::new(r);
    if p.is_dir() {
        let name = p.file_name().and_then(|s| s.to_str()).ok_or_else(|| anyhow!("bad path {r}"))?.to_string();
        return Ok((name, Some(p.to_path_buf())));
    }
    let names: Vec<_> = MODEL_REGISTRY.iter().filter(|m| matches!(m.category, pyre_whisper_engine::model::registry::ModelCategory::Asr)).map(|m| m.name).collect();
    bail!("unknown speech-to-text model '{named}'. whisburn supports: {}", names.join(", "))
}

impl SttModel {
    #[cfg(not(feature = "stt"))]
    pub fn load(_r: &str, _backend: &str, _store_root: &Path, _offline: bool) -> Result<Self> {
        bail!("pyre was built without the `stt` feature (whisburn)")
    }

    #[cfg(feature = "stt")]
    pub fn load(r: &str, backend: &str, store_root: &Path, offline: bool) -> Result<Self> {
        Self::load_with_progress(r, backend, store_root, offline, None)
    }

    /// Like [`load`], with optional whisburn prep progress (download/convert/load fractions).
    #[cfg(feature = "stt")]
    pub fn load_with_progress(
        r: &str,
        backend: &str,
        store_root: &Path,
        offline: bool,
        progress: Option<pyre_whisper_models::PrepProgressFn>,
    ) -> Result<Self> {
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
        let dir = pyre_whisper_core::resolve_model_dir(&name);
        let has_bundle = dir.join("model.bpk").is_file() || dir.join(format!("{name}.bpk")).is_file();
        if !has_bundle {
            if offline || explicit.is_some() { bail!("no whisburn bundle for '{name}' in {} (offline or explicit path)", dir.display()); }
            tracing::info!("whisburn: downloading + converting '{name}' into {}", models_dir.display());
            let mut opts = pyre_whisper_models::DownloadOptions::default();
            opts.progress = progress;
            pyre_whisper_models::download_model(&name, &opts)
                .map_err(|e| anyhow!("whisburn download/convert failed: {e:#}"))?;
        }
        // Same Burn as the LLM path, so Whisper runs on any pyre backend (cuda, wgpu, vulkan, cpu).
        let (dev, _dt, _label) = crate::backend::select(if backend == "ndarray" { "cpu" } else { backend }, "f32", None)?;
        let rt = InferenceRuntime::with_device(dev, false, false);
        let t = std::time::Instant::now();
        rt.warm_model(&name).map_err(|e| anyhow!("whisburn failed to load '{name}': {e}"))?;
        let backend_desc = rt.backend_name();
        tracing::info!("whisburn '{name}' ready on {backend_desc} in {:.2}s", t.elapsed().as_secs_f64());
        Ok(Self { name, dir, backend_desc, rt: Mutex::new(rt), busy: AtomicBool::new(false), queued: AtomicU64::new(0) })
    }
    pub fn describe(&self) -> String { format!("whisburn {} on {}", self.name, self.backend_desc) }
    /// Transcribe 1 s of silence on a background thread so the first real request doesn't pay
    /// kernel compilation/autotune (holds the model lock, so an early request just waits for it).
    /// Opt out with PYRE_STT_WARMUP=0.
    pub fn warm_in_background(self: &Arc<Self>) {
        if matches!(std::env::var("PYRE_STT_WARMUP").as_deref(), Ok("0" | "false" | "off")) { return; }
        #[cfg(feature = "stt")]
        {
            let me = self.clone();
            let _ = std::thread::Builder::new().name("stt-warmup".into()).spawn(move || {
                let rt = me.rt.lock().unwrap_or_else(|p| p.into_inner());
                let t = std::time::Instant::now();
                let o = pyre_whisper_core::TaskOptions::default();
                match rt.transcribe_waveform(&me.name, vec![0.0f32; 16_000], 16_000, &o) {
                    Ok(_) => tracing::info!("stt {} warm-up took {:.1}s", me.name, t.elapsed().as_secs_f64()),
                    Err(e) => tracing::warn!("stt {} warm-up failed: {e}", me.name),
                }
            });
        }
    }
    pub fn busy(&self) -> bool { self.busy.load(Ordering::SeqCst) }
    pub fn queued(&self) -> u64 { self.queued.load(Ordering::SeqCst) }
}

pub(crate) struct Req {
    pub file: Vec<u8>, pub filename: String, pub model: Option<String>, pub format: String, pub language: Option<String>,
    // whisburn options (pyre extensions, used by peal): decoding + whisburn-identical rendering.
    pub beam_size: Option<usize>, pub max_tokens: Option<usize>, pub timestamps: Option<bool>, pub sentences: bool, pub stream: bool,
    /// pyre extension: split by speaker with the loaded diarizer; `attribution` = turns (default) | midpoint
    pub diarize: bool, pub attribution: String,
    /// When set via jobs API: true = translations endpoint behaviour.
    pub translate_task: bool,
    /// Request opt-in. Also honored when `PYRE_STT_EXPERIMENTAL=1`.
    pub experimental: bool,
    /// Filled by `apply_route` so the client can see which model served the request.
    pub routing: Option<serde_json::Value>,
}

fn parse_bool(s: &str) -> bool { matches!(s.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on") }

/// A gated model or a gated long clip is a bad request, not an engine failure.
fn stt_error_status(msg: &str) -> StatusCode {
    if msg.contains("experimental and off by default") {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

/// Formats: OpenAI's `json`/`text`/`srt`/`vtt`/`verbose_json`, plus whisburn's own renderings
/// `txt` (timestamped lines when timestamps are on) and `whisburn_json` (whisburn's result JSON).
const FORMATS: &[&str] = &["json", "text", "srt", "vtt", "verbose_json", "txt", "whisburn_json"];

pub(crate) async fn parse_multipart(mut mp: Multipart) -> Result<Req> {
    let mut q = Req { file: vec![], filename: "audio.wav".into(), model: None, format: "json".into(), language: None, beam_size: None, max_tokens: None, timestamps: None, sentences: false, stream: false, diarize: false, attribution: "turns".into(), translate_task: false, experimental: false, routing: None };
    while let Some(f) = mp.next_field().await? {
        match f.name().unwrap_or("") {
            "file" => { if let Some(n) = f.file_name() { q.filename = n.to_string(); } q.file = f.bytes().await?.to_vec(); }
            "model" => q.model = Some(f.text().await?),
            "response_format" => q.format = f.text().await?,
            "language" => { let l = f.text().await?; if !l.is_empty() { q.language = Some(l); } }
            "beam_size" => q.beam_size = Some(f.text().await?.trim().parse().map_err(|_| anyhow!("beam_size must be an integer"))?),
            "max_tokens" => q.max_tokens = Some(f.text().await?.trim().parse().map_err(|_| anyhow!("max_tokens must be an integer"))?),
            "timestamps" => q.timestamps = Some(parse_bool(&f.text().await?)),
            "sentences" => q.sentences = parse_bool(&f.text().await?),
            "stream" => q.stream = parse_bool(&f.text().await?),
            "diarize" => q.diarize = parse_bool(&f.text().await?),
            "attribution" => q.attribution = f.text().await?.trim().to_ascii_lowercase(),
            "task" => {
                let v = f.text().await?.trim().to_ascii_lowercase();
                q.translate_task = matches!(v.as_str(), "translate" | "translations" | "translation");
            }
            "experimental" => q.experimental = parse_bool(&f.text().await?),
            _ => { let _ = f.bytes().await; } // prompt, temperature, timestamp_granularities: accepted, ignored
        }
    }
    if q.file.is_empty() { bail!("missing 'file' field"); }
    Ok(q)
}
async fn parse(mp: Multipart) -> Result<Req> { parse_multipart(mp).await }


/// Resolve an STT model for the OpenAI-compatible endpoints.
/// Known whisburn ASR ids must match exactly (no silent fallback to another loaded model);
/// placeholder names like `whisper-1` still fall back to the first loaded STT.
fn resolve_stt(app: &crate::registry::App, name: Option<&str>) -> Result<std::sync::Arc<crate::registry::Loaded>, Response> {
    use crate::server::err;
    if let Some(n) = name {
        if let Some(m) = app.find_exact(crate::registry::Kind::Stt, n) {
            return Ok(m);
        }
        #[cfg(feature = "stt")]
        {
            if map_ref(n).is_ok() {
                return Err(err(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "speech model '{n}' is not loaded; load it via POST /api/models/load                          or use POST /api/audio/jobs (downloads/converts on demand)"
                    ),
                ));
            }
        }
    }
    app.find(crate::registry::Kind::Stt, name)
}

fn apply_route(app: &crate::registry::App, q: &mut Req, translate: bool) -> Result<(), String> {
    let loaded: Vec<String> = app.loaded().iter().filter(|m| m.kind == crate::registry::Kind::Stt).map(|m| m.id.clone()).collect();
    let requested = q.model.clone().or_else(|| loaded.first().cloned()).unwrap_or_else(|| "base".into());
    match crate::speech_caps::route(&requested, &loaded, q.language.as_deref(), translate || q.translate_task) {
        Ok(d) => {
            crate::speech_caps::gate_experimental_model(&d.served, q.experimental)?;
            q.model = Some(d.served.clone());
            q.routing = Some(crate::speech_caps::decision_json(&d));
            if d.redirected {
                tracing::info!("stt route: {}", d.reason);
            }
            Ok(())
        }
        Err(msg) => Err(msg),
    }
}

pub async fn handle(app: Arc<crate::registry::App>, mp: Multipart, translate: bool) -> Response {
    use crate::server::err;
    let mut q = match parse(mp).await { Ok(q) => q, Err(e) => return err(StatusCode::BAD_REQUEST, format!("{e:#}")) };
    if let Err(msg) = apply_route(&app, &mut q, translate) { return err(StatusCode::BAD_REQUEST, msg); }
    let routing = q.routing.clone();
    let m = match resolve_stt(&app, q.model.as_deref()) { Ok(m) => m, Err(r) => return r };
    let stt = match &m.inner { crate::registry::Inner::Stt(s) => s.clone(), _ => return err(StatusCode::INTERNAL_SERVER_ERROR, "registry mismatch") };
    if !FORMATS.contains(&q.format.as_str()) {
        return err(StatusCode::BAD_REQUEST, format!("unsupported response_format '{}'", q.format));
    }
    if q.diarize {
        if crate::diar::DIAR.get().is_none() { return err(StatusCode::SERVICE_UNAVAILABLE, "diarize=true but no diarization model is loaded (start pyre with --load diar=nvidia/Nemotron-3-Diarization)"); }
        if !matches!(q.attribution.as_str(), "turns" | "midpoint") { return err(StatusCode::BAD_REQUEST, "attribution must be turns or midpoint"); }
        if q.format == "json" { q.format = "verbose_json".into(); }
    }
    let id = m.id.clone();
    let stream = q.stream;
    let opts = (q.timestamps, q.sentences);
    stt.queued.fetch_add(1, Ordering::SeqCst);
    let res = tokio::task::spawn_blocking(move || run(&stt, &q, translate)).await;
    match res {
        Ok(Ok((r, dur, secs, fmt))) => {
            tracing::info!("stt {id}: {dur:.2}s audio in {secs:.2}s ({:.1}x realtime)", dur / secs.max(1e-6));
            if stream { render_stream(&r, dur, translate, routing.as_ref()) } else { render(&r, &fmt, dur, translate, opts, routing.as_ref()) }
        }
        Ok(Err(e)) => {
            let msg = format!("{e:#}");
            err(stt_error_status(&msg), msg)
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[cfg(not(feature = "stt"))]
fn run(_: &SttModel, _: &Req, _: bool) -> Result<(Transcript, f64, f64, String)> { bail!("built without stt") }
#[cfg(not(feature = "stt"))]
struct Transcript;
#[cfg(not(feature = "stt"))]
fn render_stream(_: &Transcript, _: f64, _: bool, _: Option<&serde_json::Value>) -> Response { crate::server::err(StatusCode::NOT_IMPLEMENTED, "built without stt") }
#[cfg(not(feature = "stt"))]
fn render(_: &Transcript, _: &str, _: f64, _: bool, _: (Option<bool>, bool), _: Option<&serde_json::Value>) -> Response { crate::server::err(StatusCode::NOT_IMPLEMENTED, "built without stt") }

#[cfg(feature = "stt")]
type Transcript = pyre_whisper_core::TranscriptResult;

#[cfg(feature = "stt")]
fn run(stt: &SttModel, q: &Req, translate: bool) -> Result<(Transcript, f64, f64, String)> {
    let rt = stt.rt.lock().unwrap_or_else(|p| p.into_inner());
    stt.queued.fetch_sub(1, Ordering::SeqCst);
    stt.busy.store(true, Ordering::SeqCst);
    struct Clear<'a>(&'a AtomicBool);
    impl Drop for Clear<'_> { fn drop(&mut self) { self.0.store(false, Ordering::SeqCst); } }
    let _c = Clear(&stt.busy);
    let ext = Path::new(&q.filename).extension().and_then(|e| e.to_str()).unwrap_or("wav").to_ascii_lowercase();
    let wav = pyre_whisper_audio::decode_bytes_to_mono_pcm(&q.file, &ext).map_err(|e| anyhow!("could not decode audio ({ext}): {e}"))?;
    let wav = wav.ensure_sample_rate(16_000).map_err(|e| anyhow!("resample: {e}"))?;
    // Whisper STFT needs >= n_fft (400) samples; pad tiny clips instead of asserting in the engine.
    let mut wav = wav;
    const MIN_SAMPLES: usize = 400;
    if wav.samples.len() < MIN_SAMPLES {
        if wav.samples.is_empty() { bail!("decoded audio is empty ({ext})"); }
        wav.samples.resize(MIN_SAMPLES, 0.0);
    }
    let dur = wav.duration_secs();
    crate::speech_caps::gate_long_audio(&stt.name, dur, q.experimental).map_err(|e| anyhow!(e))?;
    let mut o = pyre_whisper_core::TaskOptions::default();
    o.task = if translate { pyre_whisper_core::SpeechTask::Translate } else { pyre_whisper_core::SpeechTask::Transcribe };
    if let Some(l) = &q.language { o.language = pyre_whisper_core::LanguageCode::new(l.clone()); }
    // whisburn only walks past the first 30 s window / sentence in timestamp mode, so always use it.
    // Callers may turn it off (whisburn CLI default for *_en models) to get whisburn-identical text.
    o.include_timestamps = q.timestamps.unwrap_or(true);
    if let Some(b) = q.beam_size { o.beam_size = b.clamp(1, 16); }
    if let Some(t) = q.max_tokens { o.max_tokens = t.clamp(1, 448); }
    o.group_into_sentences = q.sentences;
    let t = std::time::Instant::now();
    let name = stt.name.clone();
    let samples = wav.samples.clone();
    let sr = wav.sample_rate;
    let diarize = q.diarize;
    let attribution = q.attribution.clone();
    let r = catch_unwind(AssertUnwindSafe(|| -> Result<Transcript> {
        if diarize {
            let tx = |s: Vec<f32>| rt.transcribe_waveform(&name, s, 16_000, &o).map_err(|e| anyhow!("whisburn: {e}"));
            diarized(&tx, samples, &attribution)
        } else {
            rt.transcribe_waveform(&name, samples, sr, &o).map_err(|e| anyhow!("whisburn: {e}"))
        }
    }));
    let r = match r {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return Err(e),
        Err(panic) => {
            let msg = if let Some(s) = panic.downcast_ref::<&str>() { (*s).to_string() }
                else if let Some(s) = panic.downcast_ref::<String>() { s.clone() }
                else { "whisburn panicked".into() };
            bail!("stt engine panic: {msg}");
        }
    };
    Ok((r, dur, t.elapsed().as_secs_f64(), q.format.clone()))
}


/// Progress-aware STT used by /api/audio/jobs. Emits JSON stages on 	x.
#[cfg(feature = "stt")]
pub(crate) fn run_job(
    app: Arc<crate::registry::App>,
    mut q: Req,
    translate: bool,
    cancel: Arc<AtomicBool>,
    tx: &tokio::sync::mpsc::UnboundedSender<serde_json::Value>,
    job: &str,
) -> Result<serde_json::Value> {
    use serde_json::json;
    let t0 = std::time::Instant::now();
    let emit = |stage: &str, extra: serde_json::Value| {
        let mut v = json!({"stage": stage, "job": job, "elapsed_s": t0.elapsed().as_secs_f64()});
        if let Some(o) = extra.as_object() { for (k, val) in o { v[k] = val.clone(); } }
        let _ = tx.send(v);
        let stage_s = stage.to_string();
        let job_s = job.to_string();
        let msg = extra.get("message").and_then(|x| x.as_str()).unwrap_or(stage).to_string();
        let cur = extra.get("current").and_then(|x| x.as_u64());
        let tot = extra.get("total").and_then(|x| x.as_u64());
        let frac = extra.get("fraction").and_then(|x| x.as_f64());
        let eta = extra.get("eta_s").and_then(|x| x.as_f64());
        let bytes = extra.get("bytes").and_then(|x| x.as_u64());
        let bytes_total = extra.get("bytes_total").and_then(|x| x.as_u64());
        crate::activity::patch(&job_s, |j| {
            j.stage = stage_s.clone();
            j.label = msg.clone();
            j.current = cur; j.total = tot; j.fraction = frac; j.eta_s = eta;
            j.bytes = bytes; j.bytes_total = bytes_total;
            j.cancelable = true;
        });
    };
    if cancel.load(Ordering::Relaxed) { bail!("cancelled"); }
    let translate = translate || q.translate_task;
    if let Err(msg) = apply_route(&app, &mut q, translate) {
        bail!("{msg}");
    }
    if let Some(r) = &q.routing {
        emit("resolving_model", json!({"message": r["reason"], "model": r["served"], "routing": r}));
    }
    emit("resolving_model", json!({"message": "resolving speech model", "model": q.model}));

    // Exact match only — never silently fall back to another loaded STT (e.g. tiny → base).
    let m = if let Some(ref name) = q.model {
        if let Some(m) = app.find_exact(crate::registry::Kind::Stt, name) {
            m
        } else {
            emit("downloading", json!({"message": format!("loading model {name} (download/convert if needed)"), "model": name}));
            let tx_p = tx.clone();
            let job_p = job.to_string();
            let name_p = name.clone();
            let progress: pyre_whisper_models::PrepProgressFn = std::sync::Arc::new(move |p: pyre_whisper_models::PrepProgress| {
                let stage = match p.stage {
                    "download" | "convert" => "downloading",
                    _ => "loading_model",
                };
                let mut v = json!({
                    "stage": stage, "job": job_p, "model": name_p,
                    "message": p.label, "fraction": p.fraction, "prep_stage": p.stage,
                });
                if let Some(b) = p.bytes_done { v["bytes"] = json!(b); }
                if let Some(b) = p.bytes_total { v["bytes_total"] = json!(b); }
                let _ = tx_p.send(v);
            });
            let loaded = crate::registry::load(
                &app.storage,
                &app.defaults,
                crate::registry::LoadReq {
                    kind: crate::registry::Kind::Stt,
                    r: name,
                    alias: None,
                    backend: Some("cpu".into()),
                    dtype: None,
                    ctx_size: None,
                    stt_progress: Some(progress),
                },
            )?;
            let id = loaded.id.clone();
            app.insert(loaded)?;
            emit("loading_model", json!({"message": format!("model {id} ready"), "model": id}));
            app.find_exact(crate::registry::Kind::Stt, name)
                .ok_or_else(|| anyhow!("model '{name}' failed to register after load"))?
        }
    } else {
        match app.find(crate::registry::Kind::Stt, None) {
            Ok(m) => m,
            Err(_) => bail!("no speech-to-text model loaded; pick a model in the Audio tab"),
        }
    };
    if cancel.load(Ordering::Relaxed) { bail!("cancelled"); }
    let stt = match &m.inner { crate::registry::Inner::Stt(s) => s.clone(), _ => bail!("registry mismatch") };
    if !FORMATS.contains(&q.format.as_str()) { bail!("unsupported response_format '{}'", q.format); }
    if q.diarize {
        if crate::diar::DIAR.get().is_none() { bail!("diarize=true but no diarization model is loaded"); }
        if q.format == "json" { q.format = "verbose_json".into(); }
    }

    emit("decoding", json!({"message": format!("decoding {}", q.filename), "bytes": q.file.len()}));
    let cancel_h = cancel.clone();
    PROG.with(|p| {
        *p.borrow_mut() = Some(JobProg {
            ledger: pyre_whisper_core::ProgressLedger::new(),
            tx: tx.clone(),
            job: job.to_string(),
            job_at: t0,
            transcribe_at: None,
        });
    });
    pyre_whisper_engine::job_hooks::install(pyre_whisper_engine::job_hooks::JobHooks {
        cancel: cancel_h,
        on_chunk: Some(std::sync::Arc::new(|report| {
            PROG.with(|p| {
                let mut slot = p.borrow_mut();
                let Some(prog) = slot.as_mut() else { return };
                if prog.transcribe_at.is_none() {
                    prog.transcribe_at = Some(std::time::Instant::now());
                }
                if prog.diarizing() {
                    prog.ledger.set_phase("transcribing", "transcribing");
                }
                prog.ledger.observe_chunks(
                    report.chunks_done as u64,
                    report.chunks_total as u64,
                    report.samples_done as u64,
                    report.samples_total as u64,
                );
            });
            prog_publish();
        })),
    });
    let _prog_guard = ProgGuard;
    JOB_TX.with(|c| *c.borrow_mut() = Some((tx.clone(), job.to_string(), cancel.clone(), t0)));
    stt.queued.fetch_add(1, Ordering::SeqCst);
    let res = run(&stt, &q, translate);
    JOB_TX.with(|c| *c.borrow_mut() = None);
    drop(_prog_guard);
    let (r, dur, secs, fmt) = res?;
    if cancel.load(Ordering::Relaxed) { bail!("cancelled"); }
    tracing::info!("stt {}: {:.2}s audio in {:.2}s ({:.1}x realtime)", m.id, dur, secs, dur / secs.max(1e-6));
    emit("rendering", json!({"message": format!("rendering {fmt}"), "duration_s": dur, "process_s": secs}));
    let mut body = verbose(&r, dur, translate, q.routing.as_ref());
    body["response_format"] = json!(fmt);
    body["seconds"] = json!(secs);
    body["model"] = json!(m.id);
    if let Some(r) = &q.routing { body["routing"] = r.clone(); }
    Ok(body)
}

#[cfg(not(feature = "stt"))]
pub(crate) fn run_job(
    _: Arc<crate::registry::App>, _: Req, _: bool, _: Arc<AtomicBool>,
    _: &tokio::sync::mpsc::UnboundedSender<serde_json::Value>, _: &str,
) -> Result<serde_json::Value> { bail!("built without stt") }

thread_local! {
    static JOB_TX: std::cell::RefCell<Option<(
        tokio::sync::mpsc::UnboundedSender<serde_json::Value>,
        String,
        Arc<AtomicBool>,
        std::time::Instant,
    )>> = const { std::cell::RefCell::new(None) };
}

fn job_emit(stage: &str, extra: serde_json::Value) {
    JOB_TX.with(|c| {
        if let Some((tx, job, _, t0)) = c.borrow().as_ref() {
            let mut v = serde_json::json!({"stage": stage, "job": job, "elapsed_s": t0.elapsed().as_secs_f64()});
            if let Some(o) = extra.as_object() { for (k, val) in o { v[k] = val.clone(); } }
            let _ = tx.send(v);
        }
    });
}

fn job_cancelled() -> bool {
    JOB_TX.with(|c| c.borrow().as_ref().map(|(_, _, cancel, _)| cancel.load(Ordering::Relaxed)).unwrap_or(false))
}

#[cfg(feature = "stt")]
struct JobProg {
    ledger: pyre_whisper_core::ProgressLedger,
    tx: tokio::sync::mpsc::UnboundedSender<serde_json::Value>,
    job: String,
    job_at: std::time::Instant,
    transcribe_at: Option<std::time::Instant>,
}

#[cfg(feature = "stt")]
impl JobProg {
    fn diarizing(&self) -> bool {
        self.ledger.view(0.0).phase == "diarizing"
    }
}

#[cfg(feature = "stt")]
thread_local! {
    static PROG: std::cell::RefCell<Option<JobProg>> = const { std::cell::RefCell::new(None) };
}

#[cfg(feature = "stt")]
struct ProgGuard;

#[cfg(feature = "stt")]
impl Drop for ProgGuard {
    fn drop(&mut self) {
        PROG.with(|p| *p.borrow_mut() = None);
        pyre_whisper_engine::job_hooks::clear();
    }
}

#[cfg(feature = "stt")]
fn prog_publish() {
    let published = PROG.with(|p| {
        let slot = p.borrow();
        let prog = slot.as_ref()?;
        let elapsed_stt = prog.transcribe_at.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
        let view = prog.ledger.view(elapsed_stt);
        let mut v = serde_json::json!({
            "stage": view.phase,
            "phase": view.phase,
            "job": prog.job,
            "elapsed_s": prog.job_at.elapsed().as_secs_f64(),
            "message": view.message,
        });
        if view.total.is_some() || view.completed > 0 {
            v["current"] = serde_json::json!(view.completed);
            v["completed"] = serde_json::json!(view.completed);
        }
        if let Some(t) = view.total { v["total"] = serde_json::json!(t); }
        if let Some(f) = view.fraction { v["fraction"] = serde_json::json!(f); }
        if let Some(e) = view.eta_s { v["eta_s"] = serde_json::json!(e); }
        let _ = prog.tx.send(v);
        Some((prog.job.clone(), view))
    });
    if let Some((job, view)) = published {
        crate::activity::patch(&job, |j| {
            j.stage = view.phase.to_string();
            j.label = view.message.clone();
            j.current = if view.total.is_some() || view.completed > 0 { Some(view.completed) } else { None };
            j.total = view.total;
            j.fraction = view.fraction;
            j.eta_s = view.eta_s;
            j.cancelable = true;
        });
    }
}

#[cfg(feature = "stt")]
fn prog_set_phase(phase: &'static str, message: impl Into<String>) {
    let message = message.into();
    PROG.with(|p| {
        if let Some(prog) = p.borrow_mut().as_mut() {
            prog.ledger.set_phase(phase, message);
        }
    });
    prog_publish();
}

#[cfg(feature = "stt")]
fn prog_mark_transcribe() {
    PROG.with(|p| {
        if let Some(prog) = p.borrow_mut().as_mut() {
            if prog.transcribe_at.is_none() {
                prog.transcribe_at = Some(std::time::Instant::now());
            }
            prog.ledger.set_phase("transcribing", "transcribing");
        }
    });
    prog_publish();
}

/// Speaker-attributed transcript. `turns`: diarize, then run Whisper on each speaker turn (same-speaker
/// pauses < 1.0 s merged, 0.15 s padding), so every segment has exactly one speaker and overlapped
/// speech is transcribed once per speaker. `midpoint`: one Whisper pass, each segment goes to the
/// speaker active longest within it (ties/none: the turn nearest its midpoint).
#[cfg(feature = "stt")]
fn diarized(tx: &dyn Fn(Vec<f32>) -> Result<Transcript>, samples: Vec<f32>, mode: &str) -> Result<Transcript> {
    use pyre_whisper_core::TranscriptSegment as TS;
    let d = crate::diar::DIAR.get().ok_or_else(|| anyhow!("no diarization model loaded"))?;
    let t = std::time::Instant::now();
    prog_set_phase("diarizing", "finding speakers");
    if job_cancelled() { bail!("cancelled"); }
    let turns = d.turns(&samples, if mode == "midpoint" { 0.3 } else { 1.0 });
    tracing::info!("diarize: {} turns in {:.2}s", turns.len(), t.elapsed().as_secs_f64());
    prog_set_phase("diarizing", format!("found {} speaker turns", turns.len()));
    let mut out: Option<Transcript> = None;
    let mut segs: Vec<TS> = vec![];
    if mode == "midpoint" {
        if job_cancelled() { bail!("cancelled"); }
        prog_mark_transcribe();
        let mut r = tx(samples)?;
        for s in &mut r.segments {
            let ov = |a: &crate::diar::Seg| (s.end.min(a.end) - s.start.max(a.start)).max(0.0);
            let mid = (s.start + s.end) / 2.0;
            let dist = |a: &crate::diar::Seg| if mid >= a.start && mid <= a.end { 0.0 } else { (a.start - mid).abs().min((a.end - mid).abs()) };
            let best = turns.iter().max_by(|a, b| ov(a).partial_cmp(&ov(b)).unwrap().then(dist(b).partial_cmp(&dist(a)).unwrap()));
            s.speaker = best.map(|b| crate::diar::label(b.speaker));
        }
        segs = r.segments.clone();
        out = Some(r);
    } else {
        let n = samples.len();
        let mut work: Vec<(usize, usize, &crate::diar::Seg)> = Vec::new();
        for tu in &turns {
            let a = (((tu.start - 0.15) * 16_000.0).max(0.0) as usize).min(n);
            let b = (((tu.end + 0.15) * 16_000.0) as usize).min(n);
            if b <= a + 4_800 { continue; } // < 0.3 s: Whisper only hallucinates on these
            work.push((a, b, tu));
        }
        let budget: u64 = work.iter().map(|(a, b, _)| (*b - *a) as u64).sum();
        PROG.with(|p| {
            if let Some(prog) = p.borrow_mut().as_mut() {
                prog.transcribe_at = Some(std::time::Instant::now());
                prog.ledger.begin_turns(work.len() as u64, budget);
            }
        });
        prog_publish();
        for (a, b, tu) in work {
            if job_cancelled() { bail!("cancelled"); }
            PROG.with(|p| {
                if let Some(prog) = p.borrow_mut().as_mut() {
                    prog.ledger.begin_slice((b - a) as u64);
                }
            });
            let r = tx(samples[a..b].to_vec())?;
            PROG.with(|p| {
                if let Some(prog) = p.borrow_mut().as_mut() {
                    prog.ledger.finish_slice();
                }
            });
            prog_publish();
            let text = r.text.trim().to_string();
            if !text.is_empty() {
                let spk = crate::diar::label(tu.speaker);
                let mut s = TS::new(tu.start, tu.end, text.clone());
                s.speaker = Some(spk.clone());
                segs.push(s);
                job_emit("segment", serde_json::json!({
                    "text": text, "start": tu.start, "end": tu.end, "speaker": spk,
                    "partial": segs.iter().map(|x| x.text.trim()).collect::<Vec<_>>().join(" ")
                }));
            }
            if out.is_none() { out = Some(r); }
        }
    }
    let mut r = out.unwrap_or_else(|| Transcript::empty("whisper", "transcribe"));
    r.text = segs.iter().map(|s| format!("{}: {}", s.speaker.as_deref().unwrap_or("?"), s.text.trim())).collect::<Vec<_>>().join("\n");
    r.segments = segs;
    Ok(r)
}

#[cfg(feature = "stt")]
fn verbose(r: &Transcript, dur: f64, translate: bool, routing: Option<&serde_json::Value>) -> serde_json::Value {
    let mut v = json!({
        "task": if translate { "translate" } else { "transcribe" },
        "language": r.language, "duration": dur, "text": r.text.trim(), "model": r.model,
        "segments": r.segments.iter().enumerate().map(|(i, s)| {
            let mut v = json!({"id": i, "start": s.start, "end": s.end, "text": s.text.trim()});
            if let Some(sp) = &s.speaker { v["speaker"] = json!(sp); }
            v
        }).collect::<Vec<_>>(),
    });
    if let Some(route) = routing { v["routing"] = route.clone(); }
    v
}

/// `stream=true`: SSE with one `segment` event per segment, then `done` carrying verbose_json.
/// (whisburn decodes a file in one call, so the events go out once decoding finishes.)
#[cfg(feature = "stt")]
fn render_stream(r: &Transcript, dur: f64, translate: bool, routing: Option<&serde_json::Value>) -> Response {
    use axum::response::sse::{Event, Sse};
    let v = verbose(r, dur, translate, routing);
    let mut evs: Vec<Result<Event, std::convert::Infallible>> = v["segments"].as_array().unwrap().iter()
        .map(|s| Ok(Event::default().event("segment").data(s.to_string()))).collect();
    evs.push(Ok(Event::default().event("done").data(v.to_string())));
    Sse::new(futures::stream::iter(evs)).into_response()
}

#[cfg(feature = "stt")]
fn render(r: &Transcript, fmt: &str, dur: f64, translate: bool, (ts, sentences): (Option<bool>, bool), routing: Option<&serde_json::Value>) -> Response {
    use pyre_whisper_core::OutputFormat as F;
    let text = r.text.trim().to_string();
    let whis = |f: F, ct: &'static str| match pyre_whisper_audio::encode_result_with_options(r, f, ts.unwrap_or(true), sentences) {
        Ok(b) => (StatusCode::OK, [("content-type", ct)], b).into_response(),
        Err(e) => crate::server::err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    match fmt {
        "txt" => whis(F::Txt, "text/plain; charset=utf-8"),
        "whisburn_json" => whis(F::Json, "application/json"),
        "srt" if sentences => whis(F::Srt, "application/x-subrip"),
        "vtt" if sentences => whis(F::Vtt, "text/vtt"),
        "text" => (StatusCode::OK, [("content-type", "text/plain; charset=utf-8")], text + "\n").into_response(),
        "srt" => (StatusCode::OK, [("content-type", "application/x-subrip")], pyre_whisper_audio::format_srt(&r.segments, false)).into_response(),
        "vtt" => (StatusCode::OK, [("content-type", "text/vtt")], pyre_whisper_audio::format_vtt(&r.segments, false)).into_response(),
        "verbose_json" => Json(verbose(r, dur, translate, routing)).into_response(),
        "json" => {
            let mut v = json!({"text": text});
            if let Some(route) = routing { v["routing"] = route.clone(); }
            Json(v).into_response()
        }
        _ => Json(json!({"text": text})).into_response(),
    }
}

#[cfg(all(test, feature = "stt"))]
mod map_ref_tests {
    use super::map_ref;
    use std::sync::Mutex;

    static CWD: Mutex<()> = Mutex::new(());

    #[test]
    fn registry_name_is_not_a_relative_directory() {
        let _guard = CWD.lock().unwrap();
        let tmp = std::env::temp_dir().join(format!("pyre-stt-base-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("base")).unwrap();
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(&tmp).unwrap();
        let result = map_ref("base");
        std::env::set_current_dir(&prev).unwrap();
        let _ = std::fs::remove_dir_all(&tmp);
        let (name, explicit) = result.unwrap();
        assert_eq!(name, "base");
        assert!(explicit.is_none(), "registry name was treated as {explicit:?}");
    }

    #[test]
    fn long_audio_refusal_is_a_bad_request() {
        let msg = "long audio (365s) on parakeet-tdt-0.6b-v2 is experimental and off by default.";
        assert_eq!(super::stt_error_status(msg), axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(
            super::stt_error_status("stt engine panic: reshape"),
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}
